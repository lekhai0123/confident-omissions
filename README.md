# Confident omissions in semantic segmentation

Code and aggregated results for the manuscript

> **Beyond mIoU: Understanding and detecting confident object omissions in semantic segmentation**
> Khai-Thanh Le. Submitted to *Computer Vision and Image Understanding*.

Semantic segmenters sometimes erase a whole road user: almost none of its pixels receive the correct class, and they
are absorbed into a background class with high confidence (a *confident omission into background*). This repository
contains the pipeline that measures this failure on Cityscapes, synthetic corruptions, ACDC, BDD100K and WildDash2,
the detector-guided verifier that detects it at run time, and the analysis scripts that produce every table and
figure of the paper.

## Repository structure

| Path | Content |
|---|---|
| `notebook/pilot.py` | Main pipeline: segmenters (SegFormer, Mask2Former, DeepLabV3+), calibration, object-level failure statistics, detector proposals (RT-DETR / Faster R-CNN, normal + gamma pass), the 25 proposal features, depth-based baseline, cost measurement (`--mode p4`, `--mode cost`, `--mode maag_train`). |
| `notebook/test_pilot.py` | Unit tests (`pytest notebook/test_pilot.py`). |
| `analysis/` | Analysis scripts (one per experiment; see the table below). |
| `data_prep/prepare_wilddash.py` | Converts WildDash2 panoptic labels to Cityscapes-style instance IDs (labels themselves are not included). |
| `results/` | Aggregated results (JSON / CSV) behind the tables and figures. Per-object results are not included because they are derived from ACDC and WildDash2 annotations, whose licences do not permit redistribution. |

## Tables and figures

| Paper | Script | Result file |
|---|---|---|
| Table 1 (prevalence) | `analysis/analyse_v9.py`, `analysis/e15_bdd.py`, `analysis/e23_wilddash.py`, `analysis/e25_cnn_unseen.py` | `results/e23_wilddash.json`, `results/e25_cnn_unseen.json` |
| Table 2, Appendix B (synthetic vs. real rank agreement) | `analysis/e13_rank.py`, `analysis/e28_e13_subset.py`, `analysis/e24_imagecorruptions.py` | `results/e28_*.json`, `results/e24_imagecorruptions.json` |
| Table 4 (comparison with existing methods) | `analysis/e18_comparison.py`, `analysis/e23_wilddash.py` | `results/e18_comparison.json`, `results/e23_wilddash.json` |
| Table 5 (generalisation, label-free thresholds) | `analysis/e15_bdd.py`, `analysis/e14_calibration.py`, `analysis/e25_cnn_unseen.py`, `analysis/e26_detswap.py`, `analysis/e_suite_v8.py` (leave-one-model-out) | `results/e21_lomo_v11.json`, `results/e25_*.json`, `results/e26_*.json` |
| Table 6 (ablation) | `analysis/e19_ablation.py` | `results/e19_ablation.json` |
| Uncertain regions of ACDC (Sec. 6.5) | `analysis/e22_invalid_sensitivity.py` | `results/e22_invalid_sensitivity.json` |
| Connected components vs. instances (Sec. 6.1) | `analysis/e29_cc_merge.py` | `results/e29_cc_merge.json` |
| Appendix A (definition sensitivity) | `analysis/definition_sensitivity.py` | `results/definition_sensitivity*.csv` |
| Appendix C (hit criterion) | `analysis/e27_matching.py` | `results/e27_matching.json` |
| Fig. 1 | `analysis/fig_cobg_share.py` | `results/fig_cobg_share.csv` |
| Figs. 3-4 | `analysis/e9_figure_split.py` | - |
| Cost (Sec. 6.6) | `notebook/pilot.py --mode cost` | - |

Other scripts in `analysis/` are earlier or auxiliary analyses (leakage audit `e11_audit.py`, correlated failures
`e16_correlated.py`, threshold transfer `e6_transfer.py`, ...).

## Setup

Python 3.12 or 3.13, a CUDA GPU (all experiments ran on a single NVIDIA Tesla T4).

```bash
pip install -r requirements.txt
pytest notebook/test_pilot.py
```

Segmenter and detector weights are downloaded from the Hugging Face hub (`nvidia/segformer-*-finetuned-cityscapes-*`,
`facebook/mask2former-swin-*-cityscapes-semantic`, `PekingU/rtdetr_r50vd_coco_o365`) and from the OpenMMLab model zoo
(DeepLabV3+ Cityscapes checkpoints) on first use.

## Data

No images or annotations are distributed here. Obtain the datasets from their providers and pass their roots to
`pilot.py` (`--data-root`, `--acdc-root`, `--bdd-root`, `--wd-root`):

- Cityscapes: <https://www.cityscapes-dataset.com>
- ACDC: <https://acdc.vision.ee.ethz.ch>
- BDD100K (10K semantic segmentation set): <https://bdd100k.com>
- WildDash2: <https://wilddash.cc> (convert the panoptic labels with `data_prep/prepare_wilddash.py`)

## Licence

Code: MIT (see `LICENSE`). The datasets, model checkpoints and third-party components keep their own licences.
