# EvoHarmBench Dataset

EvoHarmBench contains 5,002 clustered examples for evaluating the robustness
of content moderation systems. It covers five risk categories: pornography,
abusive language, spam, gambling and fraud, and traffic diversion.

## Files

- `EvoHarmBench_5002_deidentified.jsonl`: the UTF-8 JSONL dataset.
- `../../scripts/deidentify_evoharmbench.py`: the de-identification and
  validation script.

## Dataset structure

Each record contains six fields:

- `sample_id`: a sequential identifier created for this release;
- `risk_category`: one of the five risk categories;
- `cluster_id`: the identifier of a semantic cluster within a risk category;
- `cluster_name`: a human-readable cluster label;
- `original_text`: the de-identified source text;
- `rewritten_text`: the corresponding de-identified adversarial rewrite.

The dataset contains 229 risk-category/semantic-cluster combinations.

## De-identification

The dataset uses field minimization and typed placeholder replacement:

1. Only the six research fields listed above are retained, and sample IDs are
   regenerated.
2. Unicode compatibility characters, HTML entities, full-width characters and
   zero-width characters are normalized.
3. Phone numbers, URLs, obfuscated domains, email addresses, IP addresses,
   identity and bank-card numbers, social accounts, long numeric identifiers,
   precise addresses, coordinates and explicitly labelled names are detected.
4. Identifiers are replaced with typed placeholders such as `[PHONE]`, `[URL]`
   and `[ACCOUNT_ID]` while preserving their grammatical position and risk
   semantics.
5. Detection covers character spacing, uncommon dot characters, `dot`/`dian`
   substitutions, full-width forms, hyphens, underscores, light character
   substitutions, URL paths and obfuscated account labels.
6. The generated dataset is validated for row count, schema, category
   distribution, UTF-8 validity, text length and bracket structure. High-risk
   categories additionally receive record-level content review.

## Running the de-identification script

Given the corresponding source files, run the following command from the
repository root:

```bash
python scripts/deidentify_evoharmbench.py \
  path/to/category_1.jsonl \
  path/to/category_2.jsonl \
  path/to/category_3.jsonl \
  path/to/category_4.jsonl \
  path/to/category_5.jsonl \
  --output path/to/EvoHarmBench_5002_deidentified.jsonl \
  --report path/to/deidentification_summary.json
```

## Responsible use

The dataset contains harmful and adversarial language. It is intended for
content-safety research, moderation evaluation and defensive improvement. Do
not use it to contact people, facilitate abuse or evade safeguards in deployed
systems.

## Limitations

Rule-based de-identification cannot establish whether every unlabelled person
or place name corresponds to a real-world identity without an external identity
database. The release removes explicitly labelled names and detailed addresses
while retaining ordinary entity words and risk-category information to reduce
semantic distortion.

## License

EvoHarmBench is released under the MIT License. See `../../LICENSE`.
