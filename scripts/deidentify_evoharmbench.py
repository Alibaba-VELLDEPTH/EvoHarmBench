#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Build a privacy-scrubbed EvoHarmBench JSONL release.

The release keeps the research fields needed to reproduce benchmark analyses while
removing direct contact details and traceable identifiers from every text field.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import html
import ipaddress
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PLACEHOLDERS = {
    "email": "[EMAIL]",
    "url": "[URL]",
    "ip_address": "[IP_ADDRESS]",
    "identity_number": "[IDENTITY_NUMBER]",
    "bank_card": "[BANK_CARD]",
    "phone": "[PHONE]",
    "social_handle": "[SOCIAL_HANDLE]",
    "account_id": "[ACCOUNT_ID]",
    "numeric_identifier": "[NUMERIC_IDENTIFIER]",
    "uuid": "[UUID]",
    "geo_coordinate": "[GEO_COORDINATE]",
    "detailed_address": "[DETAILED_ADDRESS]",
    "person_name": "[PERSON_NAME]",
    "unreadable": "[UNREADABLE]",
}
KNOWN_PLACEHOLDER_RE = re.compile(
    "|".join(re.escape(value) for value in PLACEHOLDERS.values())
)
PLACEHOLDER_NAMES = {value[1:-1] for value in PLACEHOLDERS.values()}

HORIZONTAL_GAP = r"[ \t\u00a0]*"
DOMAIN_LABEL_GAP = r"[ \t\u00a0_\-]*"
TLD_GAP = r"[ \t\u00a0_\-.·•]*"
DOMAIN_DOT = (
    rf"(?:\.|。|丶|·|•|点|點|奌|嚸|d{TLD_GAP}[o0]{TLD_GAP}t|"
    rf"d{TLD_GAP}[i1l]{TLD_GAP}a{TLD_GAP}n|"
    rf"\[(?:\.|。|点|點|奌|嚸|dot|dian)\]|\((?:\.|。|点|點|奌|嚸|dot|dian)\))"
)
SPACED_TLD = (
    rf"(?:c{TLD_GAP}[o0]{TLD_GAP}m|c{TLD_GAP}n|"
    rf"n{TLD_GAP}[e3]{TLD_GAP}t|[o0]{TLD_GAP}r{TLD_GAP}g|"
    rf"i{TLD_GAP}o|c{TLD_GAP}o|m{TLD_GAP}e|t{TLD_GAP}v|"
    rf"c{TLD_GAP}c|a{TLD_GAP}p{TLD_GAP}p|a{TLD_GAP}i|"
    rf"t{TLD_GAP}o{TLD_GAP}p|x{TLD_GAP}y{TLD_GAP}z|"
    rf"v{TLD_GAP}[i1l]{TLD_GAP}p|s{TLD_GAP}[i1l]{TLD_GAP}t{TLD_GAP}[e3]|"
    rf"s{TLD_GAP}h{TLD_GAP}o{TLD_GAP}p|"
    rf"l{TLD_GAP}i{TLD_GAP}n{TLD_GAP}k|"
    rf"l{TLD_GAP}i{TLD_GAP}v{TLD_GAP}e|"
    rf"i{TLD_GAP}n{TLD_GAP}f{TLD_GAP}o|"
    rf"b{TLD_GAP}i{TLD_GAP}z|[a-z](?:{TLD_GAP}[a-z]){{1,23}})"
)
SPACED_DOMAIN_RE = re.compile(
    rf"(?i)(?<![a-z0-9@])"
    rf"(?:[a-z0-9](?:{DOMAIN_LABEL_GAP}[a-z0-9]){{0,62}}{HORIZONTAL_GAP}{DOMAIN_DOT}{HORIZONTAL_GAP}){{1,5}}"
    rf"{SPACED_TLD}(?!{HORIZONTAL_GAP}[a-z0-9]){HORIZONTAL_GAP}"
    rf"(?::\s*\d{{2,5}})?(?:[/\\?#][^\s<>\"',，。!！？;；、）】\]}}]*)?"
)
ACCOUNT_SEPARATOR_CHAR = r"[ \t\u00a0\-—–－_.·•/\\()（）@＠❤♥:：<>=+|~]"
ACCOUNT_SEPARATOR = rf"{ACCOUNT_SEPARATOR_CHAR}*"
NUMERIC_SYMBOL_CLASS = "0-9零〇○洞一壹幺二贰两三叁四肆五伍六陆七柒八捌九玖"
NUMERIC_SYMBOL = rf"[{NUMERIC_SYMBOL_CLASS}]"
ACCOUNT_SYMBOL = rf"[a-z{NUMERIC_SYMBOL_CLASS}]"
OBFUSCATED_ACCOUNT_RE = re.compile(
    rf"(?i)(?:q{ACCOUNT_SEPARATOR}q|v{ACCOUNT_SEPARATOR}x|v{ACCOUNT_SEPARATOR}信|"
    rf"t{ACCOUNT_SEPARATOR}g|w{ACCOUNT_SEPARATOR}a|[微薇威徽嶶]{ACCOUNT_SEPARATOR}信|"
    rf"[飞飛]{ACCOUNT_SEPARATOR}[机機]|"
    rf"电{ACCOUNT_SEPARATOR}报|電{ACCOUNT_SEPARATOR}報|[扣蔻]{ACCOUNT_SEPARATOR}[扣蔻]|"
    rf"企{ACCOUNT_SEPARATOR}[鹅鵝])"
    rf"{ACCOUNT_SEPARATOR}(?:群聊|群號|群号|群|羣|裙|qun|号码|號碼|号|號|ID)?"
    rf"{ACCOUNT_SEPARATOR}[:：是为加]?{ACCOUNT_SEPARATOR}"
    rf"(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
)
SINGLE_Q_GROUP_RE = re.compile(
    rf"(?i)q{ACCOUNT_SEPARATOR}(?:群聊|群號|群号|群|羣|裙|qun){ACCOUNT_SEPARATOR}"
    rf"[:：是为加]?{ACCOUNT_SEPARATOR}(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
)
GENERIC_GROUP_RE = re.compile(
    rf"(?i)(?:群聊|群號|群号|群|羣|裙|qun){ACCOUNT_SEPARATOR}"
    rf"[:：是为加]?{ACCOUNT_SEPARATOR}(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
)
BRACKET_CANDIDATE_RE = re.compile(
    r"<[^<>\n]{5,48}>|〈[^〈〉\n]{5,48}〉|《[^《》\n]{5,48}》|"
    r"「[^「」\n]{5,48}」|『[^『』\n]{5,48}』|【[^【】\n]{5,48}】|"
    r"\([^()\n]{5,48}\)|（[^（）\n]{5,48}）|\[[^\[\]\n]{5,48}\]"
)
BRACKET_TOKEN_SEPARATOR_RE = re.compile(r"[ \t\-—–－_.·•]+")
BRACKET_CONTEXT_PREFIX_RE = re.compile(
    r"(?:邀请码|邀請碼|推广码|推廣碼|推荐码|推薦碼|注册码|註冊碼|验证码|驗證碼|口令|暗号|暗號|网址|網址|网站|網站|输入|輸入)\s*[:：]?\s*$"
)


def is_bracketed_identifier(candidate: str) -> bool:
    content = candidate[1:-1]
    if content in PLACEHOLDER_NAMES:
        return False
    token = BRACKET_TOKEN_SEPARATOR_RE.sub("", content)
    if not re.fullmatch(rf"[A-Za-z{NUMERIC_SYMBOL_CLASS}]{{5,32}}", token):
        return False
    contains_number = bool(re.search(rf"[{NUMERIC_SYMBOL_CLASS}]", token))
    all_uppercase_ascii = bool(re.search(r"[A-Z]", token)) and not re.search(r"[a-z]", token)
    return contains_number or all_uppercase_ascii
CODE_CONTEXT_RE = re.compile(
    rf"(?i)(?:邀请码|邀請碼|推广码|推廣碼|推荐码|推薦碼|注册码|註冊碼|注册渠道码|註冊渠道碼|"
    rf"验证码|驗證碼|加群验证|加群驗證|群验证|群驗證|注册|註冊|口令|暗号|暗號|"
    rf"频道码|頻道碼|群码|群碼|代码|代碼)"
    rf"{ACCOUNT_SEPARATOR}[\[【《『<〈(（]?{ACCOUNT_SEPARATOR}"
    rf"(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{3,31}}{ACCOUNT_SYMBOL}"
    rf"{ACCOUNT_SEPARATOR}[\]】》』>〉)）]?"
)
CONTACT_ACCOUNT_RE = re.compile(
    rf"(?i)[\[【《『<〈(（]?(?:加我|加微|加徽|联系我|聯繫我|联系方式|聯繫方式|客服|排名|排行|"
    rf"抖音号|抖音號|抖音|小红书号|小紅書號|微博号|微博號|"
    rf"[飞飛]{ACCOUNT_SEPARATOR}[机機]|[微薇威徽嶶]|电报|電報|账号|帳號)"
    rf"{ACCOUNT_SEPARATOR}[\[【《『<〈(（]?{ACCOUNT_SEPARATOR}"
    rf"(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
    rf"{ACCOUNT_SEPARATOR}[\]】》』>〉)）]?"
)
SITE_CODE_CONTEXT_RE = re.compile(
    rf"(?i)(?:网址|網址|网站|網站|输入|輸入|下载|下載|访问|訪問|打开|打開)"
    rf"{ACCOUNT_SEPARATOR}[\[【《『<〈(（]?{ACCOUNT_SEPARATOR}"
    rf"(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
    rf"{ACCOUNT_SEPARATOR}[\]】》』>〉)）]?"
)
PHONE_LEET_SYMBOL = r"[a-z0-9]"
CONTACT_LEET_PHONE_RE = re.compile(
    rf"(?i)[\[【《『<〈(（]?(?:电话|電話|联系电话|聯繫電話|联系|聯繫|手机|手機|手机号|手機號|微信|微|徽|电|電)"
    rf"{ACCOUNT_SEPARATOR}[\[【《『<〈(（]?{ACCOUNT_SEPARATOR}"
    rf"(?:{PHONE_LEET_SYMBOL}[ \t\u00a0\-—–－_.·•/\\()（）]{{0,3}}){{6,19}}{PHONE_LEET_SYMBOL}"
    rf"{ACCOUNT_SEPARATOR}[\]】》』>〉)）]?"
)
GROUP_HOMOPHONE_RE = re.compile(
    rf"(?i)(?:寇{ACCOUNT_SEPARATOR}君{ACCOUNT_SEPARATOR}羊|[扣寇]){ACCOUNT_SEPARATOR}"
    rf"(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
)
AIRPLANE_WRAPPED_ACCOUNT_RE = re.compile(
    rf"(?i)[\[【《『<〈(（]?[飞飛]{ACCOUNT_SEPARATOR}[\[【《『<〈(（]?{ACCOUNT_SEPARATOR}"
    rf"(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
    rf"{ACCOUNT_SEPARATOR}[\]】》』>〉)）]?{ACCOUNT_SEPARATOR}[机機][\]】》』>〉)）]?"
)
ACCOUNT_SUFFIX_CONTEXT_RE = re.compile(
    rf"(?i)(?:{ACCOUNT_SYMBOL}{ACCOUNT_SEPARATOR}){{4,31}}{ACCOUNT_SYMBOL}"
    rf"{ACCOUNT_SEPARATOR}(?:加我|联系我|聯繫我|加群|进群|進群)"
)
CONTACT_FIELD_CONTEXT_RE = re.compile(
    r"(?:电话|電話|联系|聯繫|手机|手機|微信|微|徽|嶶|客服|加我|加群|进群|進群|账号|帳號|"
    r"群号|群號|邀请码|邀請碼|验证码|驗證碼|注册码|註冊碼|抖音号|抖音號|飞机|飛機|电报|電報)"
)
REMAINING_CONTACT_TOKEN_RE = re.compile(
    r"(?i)(?<![a-z0-9_])[a-z0-9][a-z0-9_.\-]{4,31}(?![a-z0-9_])"
)
SITE_FIELD_CONTEXT_RE = re.compile(r"(?:网址|網址|网站|網站|输入|輸入|下载|下載|访问|訪問|打开|打開)")
REMAINING_SITE_TOKEN_RE = re.compile(
    rf"(?i)(?<![a-z0-9{NUMERIC_SYMBOL_CLASS}])"
    rf"(?:{ACCOUNT_SYMBOL}[ \t\u00a0\-—–－_.·•]{{0,2}}){{4,31}}{ACCOUNT_SYMBOL}"
    rf"(?![a-z0-9{NUMERIC_SYMBOL_CLASS}])"
)

# Replacement order matters: specific long identifiers are removed before more
# permissive contact/account patterns.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "email",
        re.compile(
            r"(?i)(?<![a-z0-9.+-])[a-z0-9.+-]{1,64}\s*(?:@|\(at\)|\[at\]|＠)\s*"
            r"[a-z0-9-]{1,63}(?:\s*(?:\.|\(dot\)|\[dot\]|点)\s*[a-z0-9-]{1,63})+"
        ),
    ),
    (
        "url",
        re.compile(
            r"(?i)(?<![a-z0-9_])(?:(?:https?|hxxps?|ftp)\s*:\s*/{1,2}|www\s*\.)\s*[^\s<>\"',，。!！?？;；、）】}]+"
        ),
    ),
    ("url", SPACED_DOMAIN_RE),
    (
        "url",
        re.compile(
            r"(?i)(?<![a-z0-9@])(?:[a-z0-9-]{1,63}\s*(?:\.|\[\.\]|\(\.\)|点)\s*){1,5}"
            r"(?:com|cn|net|org|io|co|me|tv|cc|app|ai|top|xyz|vip|site|shop|link|live|info|biz)"
            r"(?::\d{2,5})?(?:[/\\][^\s<>\"'，。！？；、）】}]*)?"
        ),
    ),
    (
        "ip_address",
        re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\s*\.\s*){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"),
    ),
    (
        "uuid",
        re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])"),
    ),
    (
        "identity_number",
        re.compile(r"(?<!\d)\d{6}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"),
    ),
    ("identity_number", re.compile(r"(?<!\d)\d{15}(?!\d)")),
    (
        "bank_card",
        re.compile(r"(?<!\d)(?:\d[\s\-]{0,2}){15,18}\d(?!\d)"),
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:(?:\+|00)\s*86[\s\-_.·•/\\()（）]{0,3})?1[\s\-_.·•/\\()（）]{0,3}[3-9](?:[\s\-_.·•/\\()（）]{0,3}\d){9}(?!\d)"),
    ),
    (
        "phone",
        re.compile(r"(?<!\d)0\d{2,3}[\s\-_.·•/\\()（）]{0,3}\d{7,8}(?!\d)"),
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:400|800)[\s\-_.·•/\\()（）]{0,3}\d{3}[\s\-_.·•/\\()（）]{0,3}\d{4}(?!\d)"),
    ),
    (
        "phone",
        re.compile(r"(?<![零〇一二两三四五六七八九])一[三四五六七八九](?:[零〇一二两三四五六七八九][\s\-_.·•]*){9}(?![零〇一二两三四五六七八九])"),
    ),
    (
        "geo_coordinate",
        re.compile(r"(?i)(?:经纬度|坐标|GPS)\s*[:：]?\s*[-+]?\d{1,3}\.\d{3,}\s*[,，/]\s*[-+]?\d{1,3}\.\d{3,}"),
    ),
    (
        "social_handle",
        re.compile(r"(?<![a-zA-Z0-9_@])[@＠][a-zA-Z0-9_\-.\u4e00-\u9fff]{2,32}"),
    ),
    ("account_id", OBFUSCATED_ACCOUNT_RE),
    ("account_id", SINGLE_Q_GROUP_RE),
    ("account_id", GENERIC_GROUP_RE),
    ("account_id", CODE_CONTEXT_RE),
    ("account_id", CONTACT_ACCOUNT_RE),
    ("account_id", GROUP_HOMOPHONE_RE),
    ("account_id", AIRPLANE_WRAPPED_ACCOUNT_RE),
    ("account_id", ACCOUNT_SUFFIX_CONTEXT_RE),
    ("url", SITE_CODE_CONTEXT_RE),
    ("phone", CONTACT_LEET_PHONE_RE),
    (
        "account_id",
        re.compile(
            r"(?i)(?:微信|微.?信|WeChat|VX|V信|QQ|扣扣|企鹅号|TG|Telegram|电报|WhatsApp|Line|加V|加微|账号|帐号|用户ID|群号)"
            r"\s*(?:号|号码|ID|id)?\s*[:：是为]?\s*[a-zA-Z0-9_\-.]{4,32}"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?:电话|手机|手机号|联系方式|联系号码|热线|致电)\s*[:：是为]?\s*(?:\+?\d[\s\-_.·•/\\()（）]{0,3}){6,15}\d"
        ),
    ),
    (
        "detailed_address",
        re.compile(
            r"(?:地址|住址|收货地址|开户地址|联系地址)\s*[:：]?\s*"
            r"[^\n,，。!！?？;；\[\]【】<>《》『』()（）]{4,80}"
        ),
    ),
    (
        "detailed_address",
        re.compile(
            r"(?:[\u4e00-\u9fff]{2,12}(?:省|自治区|特别行政区))?"
            r"[\u4e00-\u9fff]{2,12}市[\u4e00-\u9fff]{1,12}(?:区|县)"
            r"[^\n,，。!！?？;；\[\]【】<>《》『』()（）]{0,30}"
            r"(?:路|街|巷|道|村|小区)"
            r"[^\n,，。!！?？;；\[\]【】<>《》『』()（）]{0,20}"
            r"(?:号|栋|幢|单元|室)"
        ),
    ),
    (
        "person_name",
        re.compile(r"(?:姓名|联系人|收件人|开户人)\s*[:：]?\s*[\u3400-\u9fff·]{2,8}"),
    ),
    (
        "account_id",
        re.compile(r"(?:身份证|证件|银行卡|卡号|订单号|快递单号|账号|帐号|用户ID)\s*[:：]?\s*[A-Za-z0-9][A-Za-z0-9_\-]{5,31}"),
    ),
    # Long unlabelled digit strings can be QQ numbers, order/tracking IDs, or
    # compact timestamps. They are conservatively removed as quasi-identifiers.
    (
        "numeric_identifier",
        re.compile(
            rf"(?<![{NUMERIC_SYMBOL_CLASS}])"
            rf"(?:{NUMERIC_SYMBOL}[\s\-_.·•/\\()（）]{{0,3}}){{6,31}}{NUMERIC_SYMBOL}"
            rf"(?![{NUMERIC_SYMBOL_CLASS}])"
        ),
    ),
]

RESIDUAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"(?i)[\w.+-]+\s*(?:@|＠|\(at\)|\[at\])\s*[\w-]+\s*(?:\.|点|\(dot\)|\[dot\])\s*[a-z]{2,}"),
    "url_scheme": re.compile(r"(?i)(?:https?|hxxps?|ftp)\s*:\s*/{1,2}|www\s*\."),
    "bare_domain": re.compile(r"(?i)(?<![a-z0-9@])[a-z0-9-]+\s*(?:\.|\[\.\]|\(\.\)|点)\s*(?:com|cn|net|org|io|co|me|tv|cc|app|ai|top|xyz|vip|site|shop|link|live)(?![a-z0-9])"),
    "spaced_domain": SPACED_DOMAIN_RE,
    "cn_mobile": re.compile(r"(?<!\d)1[\s\-_.·•/\\()（）]{0,3}[3-9](?:[\s\-_.·•/\\()（）]{0,3}\d){9}(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\s*\.\s*){3}\d{1,3}(?!\d)"),
    "identity_number": re.compile(r"(?<!\d)\d{6}(?:18|19|20)\d{2}\d{4}\d{3}[0-9Xx](?!\d)"),
    "long_digit_identifier": re.compile(r"(?<!\d)(?:\d[\s-]{0,2}){14,18}\d(?!\d)"),
    "unlabelled_numeric_identifier": re.compile(
        rf"(?<![{NUMERIC_SYMBOL_CLASS}])"
        rf"(?:{NUMERIC_SYMBOL}[\s\-_.·•/\\()（）]{{0,3}}){{6,31}}{NUMERIC_SYMBOL}"
        rf"(?![{NUMERIC_SYMBOL_CLASS}])"
    ),
    "social_handle": re.compile(r"(?<![\w@])[@＠][a-zA-Z0-9_\-.\u4e00-\u9fff]{2,32}"),
    "contact_context": re.compile(r"(?i)(?:微信|WeChat|VX|V信|QQ|Telegram|WhatsApp|手机号|联系方式|联系号码)\s*[:：是为]?\s*[a-z0-9][a-z0-9_.\-]{3,}"),
    "obfuscated_account": OBFUSCATED_ACCOUNT_RE,
    "single_q_group": SINGLE_Q_GROUP_RE,
    "generic_group": GENERIC_GROUP_RE,
    "code_context": CODE_CONTEXT_RE,
    "contact_account": CONTACT_ACCOUNT_RE,
    "group_homophone": GROUP_HOMOPHONE_RE,
    "airplane_wrapped_account": AIRPLANE_WRAPPED_ACCOUNT_RE,
    "account_suffix_context": ACCOUNT_SUFFIX_CONTEXT_RE,
    "site_code_context": SITE_CODE_CONTEXT_RE,
    "contact_leet_phone": CONTACT_LEET_PHONE_RE,
}

TEXT_FIELDS = ("cluster_name", "original_text", "rewritten_text")
BRACKET_PAIRS = (("(", ")"), ("（", "）"), ("[", "]"), ("【", "】"), ("<", ">"), ("《", "》"), ("『", "』"), ("「", "」"), ("〈", "〉"))
IPV6_CANDIDATE = re.compile(r"(?i)(?<![0-9a-f:])[0-9a-f:]{3,39}(?![0-9a-f:])")
PHONE_DIGIT_MAP = str.maketrans(
    {
        "零": "0",
        "〇": "0",
        "○": "0",
        "洞": "0",
        "一": "1",
        "壹": "1",
        "幺": "1",
        "二": "2",
        "贰": "2",
        "两": "2",
        "三": "3",
        "叁": "3",
        "四": "4",
        "肆": "4",
        "五": "5",
        "伍": "5",
        "六": "6",
        "陆": "6",
        "七": "7",
        "柒": "7",
        "八": "8",
        "捌": "8",
        "九": "9",
        "玖": "9",
    }
)
PHONEISH_CANDIDATE = re.compile(
    rf"(?<![{NUMERIC_SYMBOL_CLASS}])"
    rf"(?:\+|00)?(?:{NUMERIC_SYMBOL}[\s\-_.·•/\\()（）]{{0,3}}){{10,13}}"
    rf"{NUMERIC_SYMBOL}(?![{NUMERIC_SYMBOL_CLASS}])"
)


def normalize_text(value: str) -> str:
    """Expose common obfuscation without translating or paraphrasing content."""
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", value)
    return value


def scrub_text(value: str, counts: Counter[str]) -> str:
    text = normalize_text(value)
    if "\ufffd" in text:
        n = text.count("\ufffd")
        counts["unreadable"] += n
        text = text.replace("\ufffd", PLACEHOLDERS["unreadable"])

    def replace_phoneish(match: re.Match[str]) -> str:
        candidate = match.group().translate(PHONE_DIGIT_MAP)
        digits = "".join(character for character in candidate if character.isdigit())
        if len(digits) == 13 and digits.startswith("86"):
            digits = digits[2:]
        if len(digits) == 11 and digits.startswith("1") and digits[1] in "3456789":
            counts["phone"] += 1
            return PLACEHOLDERS["phone"]
        return match.group()

    text = PHONEISH_CANDIDATE.sub(replace_phoneish, text)

    def replace_bracketed_identifier(match: re.Match[str]) -> str:
        if not is_bracketed_identifier(match.group()):
            return match.group()
        prefix = match.string[max(0, match.start() - 16) : match.start()]
        if BRACKET_CONTEXT_PREFIX_RE.search(prefix):
            return match.group()
        counts["account_id"] += 1
        return PLACEHOLDERS["account_id"]

    # This pass only accepts short, paired wrappers. It catches opaque codes
    # such as <TKWJSDM> without treating arbitrary bracketed prose as PII.
    text = BRACKET_CANDIDATE_RE.sub(replace_bracketed_identifier, text)

    def replace_ipv6(match: re.Match[str]) -> str:
        candidate = match.group()
        if ":" not in candidate:
            return candidate
        try:
            ipaddress.IPv6Address(candidate)
        except ipaddress.AddressValueError:
            return candidate
        counts["ip_address"] += 1
        return PLACEHOLDERS["ip_address"]

    text = IPV6_CANDIDATE.sub(replace_ipv6, text)
    for kind, pattern in PATTERNS:
        placeholder = PLACEHOLDERS[kind]

        def replace(
            match: re.Match[str], *, entity_kind: str = kind, replacement: str = placeholder
        ) -> str:
            counts[entity_kind] += 1
            matched = match.group()
            opening = ""
            closing = ""
            # Balanced wrappers fully contained in a match can disappear with
            # the identifier. If a rule captures only one side of a wrapper,
            # reinsert just that side so replacement never worsens a source
            # field that was already unbalanced.
            for opening_char, closing_char in BRACKET_PAIRS:
                difference = matched.count(opening_char) - matched.count(closing_char)
                if difference > 0:
                    opening += opening_char * difference
                elif difference < 0:
                    closing += closing_char * -difference
            return f"{opening}{replacement}{closing}"

        text = pattern.sub(replace, text)

    if CONTACT_FIELD_CONTEXT_RE.search(text):

        def replace_remaining_contact_token(match: re.Match[str]) -> str:
            if match.group() in PLACEHOLDER_NAMES:
                return match.group()
            counts["account_id"] += 1
            return PLACEHOLDERS["account_id"]

        text = REMAINING_CONTACT_TOKEN_RE.sub(replace_remaining_contact_token, text)

    if SITE_FIELD_CONTEXT_RE.search(text):

        def replace_remaining_site_token(match: re.Match[str]) -> str:
            if match.group() in PLACEHOLDER_NAMES:
                return match.group()
            counts["url"] += 1
            return PLACEHOLDERS["url"]

        text = REMAINING_SITE_TOKEN_RE.sub(replace_remaining_site_token, text)
    # Collapse only adjacent identical placeholders created by overlapping PII.
    text = re.sub(r"(\[[A-Z_]+\])(?:\s*\1)+", r"\1", text)
    return text


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], int]]:
    """Yield records and the count of invalid UTF-8 replacement characters."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            invalid_utf8 = line.count("\ufffd")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path.name} line {line_number}")
            yield line_number, value, invalid_utf8


def infer_category(path: Path) -> str:
    suffix = "_1000_samples"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def scan_residuals(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    findings: Counter[str] = Counter()
    for record in records:
        for field in TEXT_FIELDS:
            # A non-identifier sentinel prevents values on opposite sides of an
            # existing placeholder from being concatenated into a false match.
            text = KNOWN_PLACEHOLDER_RE.sub("断", str(record.get(field, "")))
            for kind, pattern in RESIDUAL_PATTERNS.items():
                findings[kind] += len(pattern.findall(text))
            findings["bracketed_identifier"] += sum(
                is_bracketed_identifier(match.group()) for match in BRACKET_CANDIDATE_RE.finditer(text)
            )
            if CONTACT_FIELD_CONTEXT_RE.search(text):
                findings["contact_field_token"] += sum(
                    match.group() not in PLACEHOLDER_NAMES
                    for match in REMAINING_CONTACT_TOKEN_RE.finditer(text)
                )
            if SITE_FIELD_CONTEXT_RE.search(text):
                findings["site_field_token"] += sum(
                    match.group() not in PLACEHOLDER_NAMES
                    for match in REMAINING_SITE_TOKEN_RE.finditer(text)
                )
            for match in PHONEISH_CANDIDATE.finditer(text):
                candidate = match.group().translate(PHONE_DIGIT_MAP)
                digits = "".join(character for character in candidate if character.isdigit())
                if len(digits) == 13 and digits.startswith("86"):
                    digits = digits[2:]
                if len(digits) == 11 and digits.startswith("1") and digits[1] in "3456789":
                    findings["mixed_script_mobile"] += 1
            for match in IPV6_CANDIDATE.finditer(text):
                if ":" not in match.group():
                    continue
                try:
                    ipaddress.IPv6Address(match.group())
                except ipaddress.AddressValueError:
                    continue
                findings["ipv6"] += 1
    return {
        kind: findings.get(kind, 0)
        for kind in (
            *RESIDUAL_PATTERNS,
            "bracketed_identifier",
            "contact_field_token",
            "site_field_token",
            "mixed_script_mobile",
            "ipv6",
        )
    }


def build_release(inputs: list[Path], output: Path, report_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_counts: Counter[str] = Counter()
    records_affected: dict[str, set[str]] = defaultdict(set)
    category_counts: Counter[str] = Counter()
    source_counts: dict[str, int] = {}
    released_records: list[dict[str, Any]] = []
    bracket_violations: list[tuple[str, str, str]] = []
    overredaction_violations: list[tuple[str, str, int, int]] = []
    invalid_utf8_count = 0

    for input_path in sorted(inputs, key=lambda item: item.name):
        category = infer_category(input_path)
        source_rows = 0
        for _, source, invalid_utf8 in read_jsonl(input_path):
            source_rows += 1
            invalid_utf8_count += invalid_utf8
            sample_id = f"EHB-{len(released_records) + 1:06d}"
            local_counts: Counter[str] = Counter()
            release_record: dict[str, Any] = {
                "sample_id": sample_id,
                "risk_category": category,
                "cluster_id": source.get("cluster_id"),
            }
            for field in TEXT_FIELDS:
                raw = source.get(field, "")
                if raw is None:
                    raw = ""
                if not isinstance(raw, str):
                    raw = str(raw)
                normalized_raw = normalize_text(raw)
                release_record[field] = scrub_text(raw, local_counts)
                if len(normalized_raw) >= 80 and len(release_record[field]) / len(normalized_raw) < 0.25:
                    overredaction_violations.append(
                        (sample_id, field, len(normalized_raw), len(release_record[field]))
                    )
                for opening, closing in BRACKET_PAIRS:
                    source_difference = normalized_raw.count(opening) - normalized_raw.count(closing)
                    release_difference = release_record[field].count(opening) - release_record[field].count(closing)
                    if source_difference != release_difference:
                        bracket_violations.append((sample_id, field, opening + closing))
            for kind, count in local_counts.items():
                total_counts[kind] += count
                records_affected[kind].add(sample_id)
            released_records.append(release_record)
            category_counts[category] += 1
        source_counts[input_path.name] = source_rows

    if len(released_records) != 5002:
        raise ValueError(f"Expected 5,002 benchmark rows, found {len(released_records):,}")
    if any(not record["original_text"].strip() or not record["rewritten_text"].strip() for record in released_records):
        raise ValueError("De-identification produced an empty original_text or rewritten_text")
    if bracket_violations:
        raise ValueError(
            f"De-identification introduced {len(bracket_violations)} bracket-balance violations: "
            f"{bracket_violations[:20]}"
        )
    if overredaction_violations:
        raise ValueError(
            f"De-identification shrank {len(overredaction_violations)} long fields below 25%: "
            f"{overredaction_violations[:20]}"
        )

    residuals = scan_residuals(released_records)
    nonzero_residuals = {kind: count for kind, count in residuals.items() if count}
    if nonzero_residuals:
        raise ValueError(f"Residual direct-identifier patterns remain: {nonzero_residuals}")

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in released_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    report: dict[str, Any] = {
        "release_file": output.name,
        "row_count": len(released_records),
        "source_files": source_counts,
        "category_counts": dict(sorted(category_counts.items())),
        "schema": ["sample_id", "risk_category", "cluster_id", "cluster_name", "original_text", "rewritten_text"],
        "redaction_counts": {kind: total_counts.get(kind, 0) for kind in PLACEHOLDERS},
        "records_affected": {kind: len(records_affected.get(kind, set())) for kind in PLACEHOLDERS},
        "invalid_utf8_codepoints_replaced": invalid_utf8_count,
        "bracket_balance_violations": len(bracket_violations),
        "overredaction_violations": len(overredaction_violations),
        "residual_scan_counts": residuals,
        "privacy_note": (
            "Direct identifiers and traceable contact details were replaced with typed placeholders. "
            "The benchmark still contains safety-risk language and is intended for authorized safety "
            "research and defensive evaluation."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Five *_1000_samples.jsonl files")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--report", required=True, type=Path, help="Machine-readable de-identification summary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input files: {', '.join(missing)}")
    report = build_release(args.inputs, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
