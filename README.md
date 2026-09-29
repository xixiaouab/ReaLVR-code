<div align="center">

<img src="assets/realvr-robot.png" width="76" alt="ReaLVR robot with a magnifying glass">

# ReaLVR

### Rethinking Latent Visual Reasoning: Grounding Latent Reasoning in Visual Evidence

*“Reasoning like humans. Grounded in what you see.”*

Xi Xiao · Tianchen Zhao · Youngeun Kim · Zhuowei Li · Linghan Xu · Jiaye Wu · Zheng Zhang · Xiang Xu · Xuanbai Chen · Farhan Tejani · Jakub Zablocki · Julia Xu · Yifan Xing

University of Alabama at Birmingham · Amazon AGI

[![Project](https://img.shields.io/badge/Project-Website-2b6cb0?style=flat-square)](https://xixiaouab.github.io/projects/ReaLVR/)
[![Models](https://img.shields.io/badge/🤗%20Models-Coming%20Soon-555?style=flat-square)](https://huggingface.co/MarkShaw99/ReaLVR)
[![License: MIT](https://img.shields.io/badge/License-MIT-555?style=flat-square)](LICENSE)

[Overview](#overview) · [Method](#method) · [Installation](#installation) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

<img src="assets/evidence-credit.gif" width="100%" alt="Animated ReaLVR walkthrough: image and question become tokens, a free-running latent trajectory is generated, and answer contrast selects latent tokens for visual supervision.">

*From images and questions to latent reasoning grounded in visual evidence.*

</div>

## Overview

ReaLVR grounds continuous latent reasoning in the visual evidence needed to answer a question. It adds visual supervision to the model’s own free-running latent trajectory, alongside the existing reinforcement learning objective.

The method learns both **what visual information to preserve** and **where to apply supervision**. The architecture and inference procedure remain unchanged.

## Method

| Step | What Happens |
| :--- | :--- |
| **Generate the trajectory** | Roll out a differentiable autoregressive latent span with the current model, before supplying answer tokens. |
| **Identify relevant evidence** | Contrast a relevant visual prototype with mismatched visual prototypes using a cosine margin. |
| **Assign latent credit** | Compare attention from correct and model-generated wrong answer readouts over the same latent span. |
| **Train with visual supervision** | Apply detached credit weights to the visual margin and backpropagate through latent generation. |

The two supervision branches are used during training only. The [project page](https://xixiaouab.github.io/projects/ReaLVR/) includes the method figures, benchmark results, and evidence sensitivity analysis.

## Installation

```bash
conda create -n realvr python=3.11 -y
conda activate realvr
pip install -r requirements.txt
```

## Data

**Stage 1.** A JSON list of LLaVA-style records with one box per `<lvr>` placeholder:

```json
{
  "image": "flickr30k/2618322793.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the child on the swing wearing?"},
    {"from": "gpt", "value": "<lvr>\n<answer> Dark blue denim shorts. </answer>"}
  ],
  "bboxes": [[0.382, 0.456, 0.718, 0.656]]
}
```

Each `<lvr>` becomes `<|lvr_start|>`, one `<|lvr|>` per visual token inside its box, and
`<|lvr_end|>`; the model is trained to reconstruct those visual tokens. `--data_path` can also be a
JSON list of `{"ds_name", "data_path", "image_folder"}` entries to mix several datasets.

**Stage 2.** A JSON list of records:

```json
{
  "image": "relative/path.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion ... Options: A. ... B. ..."},
    {"from": "gpt", "value": "<answer>B</answer>"}
  ],
  "bboxes": [[x1, y1, x2, y2]]
}
```

`bboxes` (pixels or normalized) mark the visual evidence; without it the whole image is used.
The paper uses a mixture of ViRL39K and Visual-CoT.

## Training

Both stages use DeepSpeed ZeRO-3 (`scripts/zero3.json`). Set `NNODES`, `NODE_RANK`, `MASTER_ADDR`
and `GPUS_PER_NODE` for multi-node runs.

**Stage 1** (from `Qwen/Qwen2.5-VL-7B-Instruct`):

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct DATA=data/stage1.json IMAGE_FOLDER=data/images \
OUTPUT=checkpoints/stage1 bash scripts/stage1_sft.sh
```

**Stage 2** (from a Stage-1 checkpoint):

```bash
MODEL=checkpoints/stage1 DATA=data/stage2.json IMAGE_FOLDER=data/images \
OUTPUT=checkpoints/stage2 bash scripts/stage2_realvr.sh
```

| Argument | Default | Meaning |
|---|---|---|
| `--lvr_steps` | 8 | latent length K (training and inference) |
| `--evidence_weight` | 0.2 | weight of the evidence loss (0 gives plain GRPO) |
| `--evidence_margin` | 0.5 | target cosine margin |
| `--credit_eta` | 0.3 | uniform share of the position weights |
| `--num_negatives` | 16 | negative prototypes per example |

## Evaluation

```bash
python -m eval.evaluate --checkpoint checkpoints/stage2 --benchmark mmvp \
    --data-dir data/benchmarks --output-dir results/mmvp
```

Benchmarks: `mmvp`, `blink`, `hrbench4k`, `hrbench8k`, `mme_realworld_lite`. The expected layout of
`--data-dir` is listed in `python -m eval.evaluate --help`. By default each question gets one greedy
answer with K = 8 latent steps at the processor's image-size limit. `--max-pixels` changes that limit,
and `--num-samples N` samples N answers per question and selects one (`--selection`); the settings of
every run are written to its `summary.json`. `--start-index` and `--max-samples` split a benchmark into
shards; `python -m eval.merge_results` combines their summaries.

## Tests

```bash
pytest tests/
```

`tests/test_trainer.py` runs Stage-2 steps end to end on CPU with a tiny random model; it needs the
Qwen2.5-VL processor files locally (`REALVR_TEST_PROCESSOR=/path/to/Qwen2.5-VL-7B-Instruct`).

## Citation

```bibtex
@misc{xiao2026realvr,
  title={Rethinking Latent Visual Reasoning: Grounding Latent Reasoning in Visual Evidence},
  author={Xiao, Xi and Zhao, Tianchen and Kim, Youngeun and Li, Zhuowei and Xu, Linghan and Wu, Jiaye and Zhang, Zheng and Xu, Xiang and Chen, Xuanbai and Tejani, Farhan and Zablocki, Jakub and Xu, Julia and Xing, Yifan},
  year={2026},
  note={Preprint},
  url={https://xixiaouab.github.io/projects/ReaLVR/}
}
```

## Contact

[Xi Xiao](mailto:xxiao@uab.edu) · [Tianchen Zhao](mailto:tianchz@amazon.com)

Work done during an internship at Amazon AGI.

## Acknowledgements

This code builds on [Latent Visual Reasoning](https://github.com/VincentLeebang/lvr),
[Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune), [TRL](https://github.com/huggingface/trl)
and [InternVL](https://github.com/OpenGVLab/InternVL).

## License

This repository is licensed under the [MIT License](LICENSE). Code adapted from other projects keeps its
original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
