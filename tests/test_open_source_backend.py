import json
from pathlib import Path
from types import SimpleNamespace

from examples.obscure_text import main_cluster_optimization as cluster_main
from examples.obscure_text import utils


class _FakeCompletions:
    def create(self, **kwargs):
        assert kwargs["model"] == "local-model"
        assert kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="local backend ok"))]
        )


def test_openai_compatible_backend_uses_configured_model(monkeypatch):
    monkeypatch.setattr(
        utils,
        "_client",
        SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions())),
    )
    assert utils._call_llm([{"role": "user", "content": "ping"}], model="local-model") == "local backend ok"


def test_bundled_release_has_5002_rows_and_229_cluster_pairs(monkeypatch):
    root = Path(__file__).parents[1]
    data_file = root / "data/evoharmbench/EvoHarmBench_5002_deidentified.jsonl"
    rows = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines()]
    monkeypatch.setattr(cluster_main, "DATA_FILE", str(data_file))

    assert len(rows) == 5002
    assert len({(row["risk_category"], row["cluster_id"]) for row in rows}) == 229
    assert len(cluster_main.get_all_categories()) == 229
