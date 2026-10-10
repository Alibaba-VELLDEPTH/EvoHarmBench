<a id="overview"></a>

# EvoHarmBench: Breaking Content Moderation with Iterative Human-Like Evasion

> Ruijie Jian<sup>1,2,&#42;</sup>, Benlei Cui<sup>1,&#42;,†</sup>, Ting Ma<sup>1</sup>, Haidong Ding<sup>1</sup>, Kangwei Liu<sup>1</sup>, Ziwen Xu<sup>1</sup>, Longtao Huang<sup>1</sup>, Hui Xue<sup>1</sup>, Ziqiang Zhu<sup>1</sup>, Junjie Li<sup>1</sup>, Haiwen Hong<sup>1,†</sup>
>
> <sup>1</sup> Yuvion Team, Alibaba Group &nbsp;&nbsp; <sup>2</sup> University of Chinese Academy of Sciences<br>
> <sup>&#42;</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding authors.
>
> <p><img src="assets/yuvion.png" alt="Yuvion Team" height="32"> &nbsp;&nbsp; <img src="assets/ucas.png" alt="University of Chinese Academy of Sciences" height="32"></p>
>
> ## Abstract
>
> Existing evaluations of harmful content detection rely predominantly on static benchmarks, which struggle to reflect the interactive adversarial ecosystem of real-world content platforms where users continuously revise their expressions in response to moderation feedback. This mismatch creates a significant performance gap between offline benchmark scores and online deployment effectiveness. To the best of our knowledge, we present **EvoHarmBench**, the first dynamic adversarial evaluation framework for content moderation systems. The framework employs an iterative optimization loop that evolves evasion strategies at the semantic-cluster level, while simultaneously optimizing for evasion success and human readability. We systematically evaluate LLM-based defense models which are widely used in real world moderation systems. The evaluation covers 229 semantic sub-clusters across five violation categories, derived from 5,002 real-world adversarial samples collected from content platforms. Our experiments reveal substantial vulnerabilities even in leading commercial systems: after twelve optimization iterations, the attack success rate under readability constraints reaches 80.3% within SOTA LLM moderators. We will release the full benchmark data, evaluation framework, and code to encourage a shift from static benchmarking toward dynamic adversarial evaluation in content safety research.
>
> <p align="center"><img src="assets/framework.png" alt="Overview of EvoHarmBench from the paper: data source and categories, semantic cluster construction, and the iterative evaluation protocol." width="100%"></p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.27844"><img src="https://img.shields.io/badge/arXiv-2608.27844-b31b1b" alt="arXiv paper"></a>
  <a href="https://arxiv.org/abs/2608.27844"><img src="https://img.shields.io/badge/EMNLP_2026-Findings-2563eb" alt="Findings of EMNLP 2026"></a>
  <a href="data/evoharmbench"><img src="https://img.shields.io/badge/Dataset-5%2C002_examples-16845b" alt="Dataset: 5,002 examples"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-64748b" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> · <a href="#dataset">Dataset</a> · <a href="#quick-start">Quick Start</a> · <a href="#citation">Citation</a>
</p>

<details>
<summary><b>🔍 A human-like evasion example from the paper</b></summary>

The same harmful intent can survive several changes in spelling, sound, and character shape, even after a moderator stops detecting it.

<p align="center">
  <img src="assets/evasion-example.png" alt="Example from the paper showing repeated rewriting of an abusive message, from blocked expressions to a readable rewrite that bypasses moderation." width="400">
</p>

</details>

<a id="dataset"></a>
## 🤗 Dataset

The [dataset](data/evoharmbench/EvoHarmBench_5002_deidentified.jsonl) is included in this repository. It covers real platform content that escaped moderation and was subsequently reported by users.

| Risk category | Examples | Sub-clusters |
| :--- | ---: | ---: |
| Traffic diversion | 1,001 | 44 |
| Pornography | 1,000 | 54 |
| Abuse | 1,001 | 42 |
| Spam | 1,000 | 43 |
| Gambling and fraud | 1,000 | 46 |
| **Total** | **5,002** | **229** |

Each record contains `sample_id`, `risk_category`, `cluster_id`, `cluster_name`, `original_text`, and `rewritten_text`. Contact details and other identifiers are replaced with typed placeholders such as `[PHONE]`, `[URL]`, and `[ACCOUNT_ID]`. See the [dataset README](data/evoharmbench/README.md) for the schema and de-identification procedure.

<a id="quick-start"></a>
## 🔥 Quick Start

### 1️⃣ Install

Use Python 3.10 or later.

```bash
git clone https://github.com/Alibaba-VELLDEPTH/EvoHarmBench.git
cd EvoHarmBench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Connect a model

The pipeline supports OpenAI-compatible endpoints. For a locally hosted model:

```bash
export EVOHARMBENCH_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export EVOHARMBENCH_OPENAI_API_KEY=not-required
```

For an authenticated endpoint, supply its base URL and your own API key. The optional [.env.example](.env.example) lists the supported environment variables; export them in your shell or load them with your preferred environment manager.

<details>
<summary><b>Host an open-source model with vLLM</b></summary>

In a compatible GPU environment, install vLLM and start the model server:

```bash
pip install vllm

MODEL_PATH=/path/to/Qwen3.5-0.8B \
SERVED_MODEL_NAME=Qwen/Qwen3.5-0.8B \
bash scripts/serve_open_source_model.sh
```

The default address is `127.0.0.1:8000`. The launcher also accepts `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, and `MAX_NUM_SEQS`.

</details>

### 3️⃣ Run a small example

This one-example, one-iteration run uses the same small model for all four roles:

```bash
python examples/obscure_text/main_cluster_optimization.py \
  --category "微信号直发-1" \
  --sample-limit 1 \
  --iterations 1 \
  --reflection-version v1 \
  --transform-model Qwen/Qwen3.5-0.8B \
  --audit-model Qwen/Qwen3.5-0.8B \
  --comparison-model Qwen/Qwen3.5-0.8B \
  --reflection-model openai/Qwen/Qwen3.5-0.8B
```

### 4️⃣ Configure an evaluation

Set `--iterations 12` for a 12-iteration strategy search, remove `--sample-limit 1` to evaluate the full sub-cluster, and use `--all-categories` to run all sub-clusters. The four model roles can be configured independently with the flags above.

To rerun the small example's sub-cluster at full size, add `--force`; this replaces its saved run.

For the paper's evaluation, rewriting, comparison, and reflection use **DeepSeek-V3.2-Exp**, with different models serving as the moderator.

Use `--reflection-version raw` for a moderator-only baseline. Run `python examples/obscure_text/main_cluster_optimization.py --help` for all options.

Results are saved under `outputs/`, grouped by reflection version, model, and sub-cluster. Completed sub-clusters are skipped unless `--force` is specified. Use `--checkpoint-iters` to save intermediate iterations. The optimization score is the sub-cluster mean of `bypass × comparison score`; it is distinct from the thresholded ASR@Readable metric reported in the paper.

<a id="citation"></a>
## ✍️ Citation

If you use EvoHarmBench in your research, please cite:

```bibtex
@misc{jian2026evoharmbench,
  title={EvoHarmBench: Breaking Content Moderation with Iterative Human-Like Evasion},
  author={Ruijie Jian and Benlei Cui and Ting Ma and Haidong Ding and Kangwei Liu and Ziwen Xu and Longtao Huang and Hui Xue and Ziqiang Zhu and Junjie Li and Haiwen Hong},
  year={2026},
  eprint={2608.27844},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.27844}
}
```

## 🙏 Acknowledgments

This work was supported by Alibaba Security.

## 📜 License and Use

The code and dataset are released under the [MIT License](LICENSE). The benchmark contains harmful and adversarial language and is intended for content-safety evaluation and defensive research.
