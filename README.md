# Confident omissions in semantic segmentation

Code and aggregated results for the manuscript

> **Beyond mIoU: Understanding and detecting confident object omissions in semantic segmentation**
> Khai-Thanh Le. Manuscript, 2026.

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
| Fig. 1 (example) | `analysis/fig_teaser.py` | - |
| Table 1 (prevalence) | `analysis/table1_prevalence.py` | `results/table1_prevalence.json` |
| Fig. 2 (omitted vs. confident share) | `analysis/fig_cobg_share.py` | `results/fig_cobg_share.csv` |
| Table 2, Appendix B (synthetic vs. real rank agreement) | `analysis/e13_rank.py`, `analysis/e28_e13_subset.py`, `analysis/e32_build_15models.py`, `analysis/e24_imagecorruptions.py` | `results/e32_*.json`, `results/e28_*.json`, `results/e24_imagecorruptions.json` |
| Table B.2 (difference of rank correlations, repeatability) | `analysis/e35_rank_stability.py` | `results/e35_rank_stability.json` |
| Sec. 4.3 (information in the segmenter) | `analysis/e10_e12.py` | - |
| Sec. 4.4 (correlated failures, coverage by source) | `analysis/e16_correlated.py`, `analysis/e36_source_coverage.py` | `results/e16_correlated.json`, `results/e36_source_coverage.json` |
| Fig. 3 (method) | drawn by hand from the assets of `analysis/fig3_assets.py` | - |
| Table 4 (comparison) | `analysis/e18_comparison.py`, `analysis/e23_wilddash.py`, `analysis/e34_review_offline.py` (logistic regression, box-mask disagreement) | `results/e18_comparison.json`, `results/e23_wilddash.json`, `results/e34_review_offline.json` |
| Fig. 4 (free-response curves) | `analysis/e30_froc.py`, `analysis/plot_froc.py` | `results/fig_froc.json` |
| Table 5 (generalisation, label-free thresholds) | `analysis/e14_calibration.py`, `analysis/e15_bdd.py`, `analysis/e25_cnn_unseen.py`, `analysis/e26_detswap.py`, `analysis/e31_cnn_acdc.py`, `analysis/e33_detswap_acdc.py`, `analysis/e_suite_v8.py` | `results/e21_lomo_v11.json`, `results/e25_*.json`, `results/e26_*.json`, `results/e31_*.json`, `results/e33_*.json` |
| Sec. 6.2-6.3 (per class and size, threshold on a later batch, alarm load) | `analysis/e34_review_offline.py` | `results/e34_review_offline.json` |
| Table 6 (ablation) | `analysis/e19_ablation.py`, `analysis/e34_review_offline.py` (rarity-matched target) | `results/e19_ablation.json` |
| Uncertain regions of ACDC (Sec. 6.5) | `analysis/e22_invalid_sensitivity.py` | `results/e22_invalid_sensitivity.json` |
| Connected components vs. instances (Sec. 6.1) | `analysis/e29_cc_merge.py` | `results/e29_cc_merge.json` |
| Figs. 5-6 | `analysis/e9_figure_split.py` | - |
| Cost (Sec. 6.6) | `notebook/pilot.py --mode cost` | - |
| Appendix A (definition sensitivity) | `analysis/definition_sensitivity.py` | `results/definition_sensitivity*.csv` |
| Appendix C (hit criterion) | `analysis/e27_matching.py` | `results/e27_matching.json` |
| Graphical abstract | `analysis/graphical_abstract.py` | - |

`notebook/pilot.py --mode relabel` re-runs only the segmenters on the stored detector boxes. It computes the joint
definition check, the box-level maximum softmax probability and objects as connected components. Its inputs come
from `analysis/make_relabel_ref.py`.

Other scripts in `analysis/` are earlier or auxiliary analyses, such as the leakage audit (`e11_audit.py`) and
threshold transfer (`e6_transfer.py`). Settings, seeds, software versions and the lists of evaluated frames are
described in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

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
