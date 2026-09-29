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

[Overview](#overview) · [Method](#method) · [Release](#release) · [Citation](#citation)

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

<details>
<summary><b>Reported Results</b></summary>

The preprint evaluates six backbones across three model families, up to 235B total parameters. On Qwen2.5-VL-7B, ReaLVR reaches a **63.7% five-task mean**, compared with **60.4% for LVR-RL**.

| Backbone | MMVP | BLINK | HRBench-4K | HRBench-8K | MME-RealWorld | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B + ReaLVR | 72.0 | 55.8 | 71.8 | 66.6 | 52.2 | **63.7** |

Accuracy (%), three-seed mean. See the [project page](https://xixiaouab.github.io/projects/ReaLVR/#results) for the comparison and evaluation scope.

</details>

## Release

> [!NOTE]
> **Code and model checkpoints are coming soon.** This repository hosts the project overview and animation ahead of the code release.

| Resource | Location |
| :--- | :--- |
| Project and figures | [Project Website](https://xixiaouab.github.io/projects/ReaLVR/) |
| Code | This repository · Coming soon |
| Model checkpoints | [Hugging Face](https://huggingface.co/MarkShaw99/ReaLVR) · Coming soon |

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

## License

This repository is licensed under the [MIT License](LICENSE).
