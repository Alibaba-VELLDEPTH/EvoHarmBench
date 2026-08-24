# EvoHarmBench

EvoHarmBench is a benchmark and evaluation pipeline for measuring the
robustness of content moderation systems under iterative, feedback-driven text
rewriting. It organizes examples into semantic clusters, learns reusable
cluster-level rewriting strategies, and evaluates both moderation bypass and
meaning preservation.

The repository includes:

- a de-identified benchmark with 5,002 examples across five risk categories;
- the cluster-level strategy evolution and evaluation pipeline;
- support for OpenAI-compatible model endpoints;
- a launcher and smoke-test configuration for locally hosted open-source
  models.

## Dataset

The benchmark is stored at
`data/evoharmbench/EvoHarmBench_5002_deidentified.jsonl`. It contains 5,002
examples spanning 229 risk-category/semantic-cluster combinations. Each record
contains:

- `sample_id`;
- `risk_category`;
- `cluster_id` and `cluster_name`;
- `original_text` and `rewritten_text`.

See `data/evoharmbench/README.md` for the schema, de-identification method and
usage notes.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` if you want to keep model settings in a local environment
file. Export the variables in your shell or load the file with your preferred
environment manager.

## Run with an open-source model

The pipeline connects to any OpenAI-compatible endpoint. For a local vLLM
server, install vLLM in a compatible GPU environment:

```bash
pip install vllm
```

Start the server with a local Hugging Face checkpoint:

```bash
MODEL_PATH=/path/to/Qwen3.5-0.8B \
SERVED_MODEL_NAME=Qwen/Qwen3.5-0.8B \
bash scripts/serve_open_source_model.sh
```

The server listens on `127.0.0.1:8000` by default. The launcher exposes
`TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN` and
`MAX_NUM_SEQS` for larger models or different GPU layouts.

In another shell, configure the endpoint and run a one-example smoke test:

```bash
export EVOHARMBENCH_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export EVOHARMBENCH_OPENAI_API_KEY=not-required
export EVOHARMBENCH_MODEL=Qwen/Qwen3.5-0.8B

python examples/obscure_text/main_cluster_optimization.py \
  --category "微信号直发-1" \
  --sample-limit 1 \
  --iterations 1 \
  --reflection-version v0 \
  --force
```

Use `--reflection-version raw` for a moderator-only baseline. The rewriting,
moderation, comparison and reflection models can be configured independently:

```bash
python examples/obscure_text/main_cluster_optimization.py \
  --transform-model Qwen/Qwen3.5-0.8B \
  --audit-model Qwen/Qwen3.5-0.8B \
  --comparison-model Qwen/Qwen3.5-0.8B \
  --reflection-model openai/Qwen/Qwen3.5-0.8B
```

Run `python examples/obscure_text/main_cluster_optimization.py --help` for all
available options.

## Evaluation outputs

Results are written under `outputs/` and grouped by reflection version, model
and semantic cluster. Existing completed clusters are skipped unless `--force`
is supplied. Use `--checkpoint-iters` to save selected intermediate
iterations.

## Responsible use

The benchmark contains harmful and adversarial language. It is intended for
authorized content-safety evaluation and defensive research. Do not use it to
contact people, facilitate abuse or evade safeguards in deployed systems.

## License

The EvoHarmBench source code and dataset are available under the MIT License in
`LICENSE`.
