# MedJudge: Medical Multimodal Reward Modeling

> **MedJudge: Medical Multimodal Reward Modeling**
> Yunhong He\*, Kai Zhang¹, Jiarong Qian², Zhengqing Yuan³, Yanfang Ye³, Jing Huang², Lichao Sun¹\*†
>
> ¹Lehigh University&nbsp;&nbsp;²Rice University&nbsp;&nbsp;³University of Pennsylvania&nbsp;&nbsp;⁴University of Notre Dame
>
> <sub>\*Equal contribution&nbsp;&nbsp;†Corresponding author</sub>

This repository contains the official implementation of **MedJudge**, a framework
for building and evaluating **medical multimodal judges / reward models** that
score and compare candidate answers to medical visual questions.

---

## Overview

Reward models and LLM-as-a-judge pipelines are central to aligning multimodal
models, but in the medical domain a judge must reason over both the image and
clinically grounded knowledge. MedJudge studies how well general, medical, and
reward-specialized vision–language models can act as **pairwise judges** of
medical answers, and how three training paradigms improve them:

| Paradigm | Data | LLaMA-Factory stage | Judge output |
|----------|------|---------------------|--------------|
| **SFT**   | `medjudge_sft`     | `sft` | A direct `A` / `B` preference (decision-only judge) |
| **SFT-R** | `medjudge_sft_rea` | `sft` | A clinical explanation followed by a `Final preference: A/B` (reasoning judge) |
| **BT**    | `medjudge_rm`      | `rm`  | A scalar reward used to rank `chosen` over `rejected` (Bradley–Terry reward model) |

Each candidate pair is drawn from medical VQA sources and contrasts a correct
response against a perturbed one (generated, knowledge-graph-contradicting,
paraphrase, or subtle factual errors).

---

## Repository Layout

```
.
├── LLaMA-Factory/                 # Training framework (LoRA SFT / RM)
│   └── data/
│       ├── dataset_info.json      # Registers the medjudge_* datasets
│       └── medjudge/              # Train/test splits (sft, sft_rea, rm)
├── scripts/
│   ├── prepare_medjudge_llamafactory.py   # Build ShareGPT-format data
│   ├── convert_pairwise_to_sharegpt.py
│   ├── convert_test_pairwise_to_sharegpt.py
│   ├── make_ablation_splits.py            # Error-type ablation splits
│   ├── inference_qwen3vl_pairwise.py      # Batched pairwise inference
│   ├── run_inference_qwen3vl_pairwise.sh
│   ├── evaluate_pairwise.py               # Accuracy / BLEU / BERTScore
│   ├── metrics.py                         # Accuracy / Macro-F1 / AUC-ROC / AP / UCO
│   ├── best_of_4.py                       # Best-of-N generation & judging
│   └── config/                            # One config per model × {sft, sft_rea, rm}
│       ├── <model>_medjudge_sft.yaml
│       ├── <model>_medjudge_sft_rea.yaml
│       ├── <model>_medjudge_rm.yaml
│       ├── run_train.sh                   # Train a single config
│       └── run_all_models.sh              # Train all models for one paradigm
└── Medthink_Code/                 # Upstream MedThink reference code
```

---

## Models

The benchmark covers eight trainable open models and two closed-source zero-shot
baselines.

---

## Installation

```bash
# 1. Clone and create an environment
conda create -n medjudge python=3.10 -y
conda activate medjudge

# 2. Install the bundled LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# 3. (Optional) UCO metric dependencies — scispaCy + UMLS linker
pip install scispacy spacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

> Set the absolute paths in `scripts/config/*.yaml` (`dataset_dir`, `output_dir`)
> and in the `run_*.sh` scripts to your environment. The released configs use
> `/path/to/...` placeholders.

---

## Data

The datasets are registered in `LLaMA-Factory/data/dataset_info.json` and stored
under `LLaMA-Factory/data/medjudge/`:

| Dataset | File | Format |
|---------|------|--------|
| `medjudge_sft` / `medjudge_sft_test`         | `medjudge_sft_*.json`     | ShareGPT messages, assistant = `A`/`B` |
| `medjudge_sft_rea` / `medjudge_sft_rea_test` | `medjudge_sft_rea_*.json` | ShareGPT messages, assistant = explanation + `Final preference` |
| `medjudge_rm` / `medjudge_rm_test`           | `medjudge_rm_*.json`      | ShareGPT ranking with `chosen` / `rejected` |

To build the datasets from the raw pairwise sources:

```bash
python scripts/prepare_medjudge_llamafactory.py \
  --medthink-root /path/to/Medthink \
  --output-dir   /path/to/Medthink/LLaMA-Factory/data/medjudge
```

### Synthetic pairwise data — per-category intermediate results

The intermediate, **per-category** synthetic pairwise data (one folder per error
strategy, before they are filtered, de-duplicated, and merged) lives under:

```
Medthink_Dataset/Merged_Dataset/Pairwise_Dataset/
├── Generated/          # SLP  -> candidate_type "omission"
├── Subtle_Error/       # SSP  -> candidate_type "subtle_error"
├── Paraphrase_Error/   # MGA  -> candidate_type "paraphrase_error"
├── KG_Contradiction/   # SCM  -> candidate_type "kg_contradiction"
└── Final_Merged/       # all categories combined + train/val/test splits
                        # (the RP "random" pairs are bundled here / in the
                        #  per-error files as pairwise_type "random")
```


---

## Training

Train a single model/paradigm:

```bash
cd scripts/config
# usage: run_train.sh <config.yaml> <num_gpus>
bash run_train.sh medgemma_4b_it_medjudge_sft.yaml 4
```

Train every model for one paradigm (`sft` | `sft_rea` | `rm`):

```bash
cd scripts/config
bash run_all_models.sh sft 4
bash run_all_models.sh sft_rea 4
bash run_all_models.sh rm 4
```

All configs use LoRA (`rank=16`, `alpha=32`, `lora_target=all`), `bf16`, cosine
schedule, and DeepSpeed when available.

---

## Inference & Evaluation

Run pairwise inference, then score the predictions:

```bash
# 1. Generate predictions (writes a JSONL with a `predict` field per example)
bash scripts/run_inference_qwen3vl_pairwise.sh

# 2. Compute the headline metrics
python scripts/metrics.py \
  --predictions  outputs/<model>/<paradigm>/test_predictions.jsonl \
  --ground-truth LLaMA-Factory/data/medjudge/medjudge_sft_test.json \
  --output       outputs/<model>/<paradigm>/metrics.json \
  --compute-uco --uco-model en_core_sci_sm --uco-threshold 0.7
```

`metrics.py` reports **Accuracy**, **Macro-F1**, **AUC-ROC**, **Average
Precision (AP)**, and **UCO**. For RM-style outputs it consumes continuous
`score_a`/`score_b` (or `prob_a`/`prob_b`); for generative judges it parses the
final `A`/`B` decision.

### UCO — UMLS Concept Overlap

UCO is a deterministic, **model-agnostic** measure of medical semantic
consistency that avoids the circularity of LLM-as-a-judge scoring. Both the
prediction and the reference are grounded into UMLS concepts with a scispaCy
biomedical pipeline (`en_core_sci_sm`, UMLS linker, confidence threshold
`τ = 0.7`, top concept per mention), and UCO is the **Jaccard overlap** of the
two CUI sets:

```
UCO = |C_gt ∩ C_pred| / |C_gt ∪ C_pred|
```

When both concept sets are empty the union is empty and UCO is defined as `1`
(trivial agreement); when exactly one set is empty UCO is `0`.

---

## Citation

If you use MedJudge, please cite:

```bibtex
@article{he2026medjudge,
  title  = {MedJudge: Medical Multimodal Reward Modeling},
  author = {He, Yunhong and Zhang, Kai and Qian, Jiarong and Yuan, Zhengqing and Ye, Yanfang and Huang, Jing and Sun, Lichao},
  year   = {2026}
}
```

## License

Released under the terms in [`LICENSE`](LICENSE).
