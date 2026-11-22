# Reproducibility notes

Everything needed to rebuild the tables from the datasets, apart from the datasets themselves.

## Environment

There are two environments.

**GPU runs** (`notebook/pilot.py`: segmentation, detection, proposals, features, timing) ran as Kaggle script
kernels on one NVIDIA Tesla T4 (16 GB) with the Kaggle image's packages:

- Python 3.13.16
- PyTorch 2.11.0+cu130
- transformers 5.20.2

**Analysis** (`analysis/*.py`: training of the verifier and all baselines, bootstrap statistics, tables) ran on CPU
with the versions pinned in `requirements.txt` (Python 3.12.9, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1,
SciPy 1.18.1; the torch / transformers pins there are only needed to run `pilot.py` locally). Gradient-boosting
results depend on the scikit-learn version as well as on the seed; another scikit-learn version can shift recall
by a few points (compare the seed spread reported in Sec. 6.4); use the pinned version to
reproduce the tables exactly.

Segmenter and detector inference runs in float16 autocast. Unit tests (`pytest notebook/test_pilot.py`) run on CPU.

The verifier's feature vector has 26 columns: the 25 features of Table 3 plus `src_owl`, the indicator of an
open-vocabulary detector that is not used in the reported runs and is therefore always 0; `p_fill` is constant 1 for
box proposals.

Analysis scripts also run with Python 3.13. A re-run on another GPU reproduces the stored
per-object statistics up to float16 rounding: on a GTX 1080 Ti, the largest difference in the correctly predicted
pixel share of an object was 0.4%, and no object changed label.

## Model weights

| Role | Identifier |
|---|---|
| Segmenters (Hugging Face) | `nvidia/segformer-b{0..5}-finetuned-cityscapes-1024-1024`; `nvidia/segformer-b0-finetuned-cityscapes-{512-1024,640-1280,768-768}`; `facebook/maskformer-resnet101-cityscapes`; `facebook/mask2former-swin-{tiny,small,large}-cityscapes-semantic`; `shi-labs/oneformer_cityscapes_swin_large`; `tue-mps/cityscapes_semantic_eomt_large_1024` (rank analysis only) |
| Segmenters (OpenMMLab) | `deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth`, `deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth`; `pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth`, `fcn_r101-d8_512x1024_80k_cityscapes_20200606_113038-3fb937eb.pth` (rank analysis only; URLs in `notebook/pilot.py`, `MMSEG_URLS`) |
| Detector | `PekingU/rtdetr_r50vd_coco_o365` (main); torchvision `fasterrcnn_resnet50_fpn_v2` with `COCO_V1` weights (detector swap) |
| Depth (baseline) | `depth-anything/Depth-Anything-V2-Small-hf`; the foreground/background network is trained with `pilot.py --mode maag_train` |

Mask2Former-Swin-B was excluded because its public checkpoint loads with a temperature above 5, which indicates
that the weights were not loaded. The run log records this.

## Frames

`results/manifests/` lists the frames that were used. These are file stems only; no image or label content is
included.

| File | Content |
|---|---|
| `cityscapes_calibration_frames.csv` | The 50 Cityscapes validation frames used for temperature scaling and the Mahalanobis Gaussians. |
| `cityscapes_eval_frames.csv` | The 450 evaluation frames. 443 of them contain at least one eligible thing object. |
| `acdc_frames.csv` | 2006 adverse frames and 1003 reference frames from the train and val splits. |
| `bdd100k_frames.csv` | 3000 frames: 1000 val and 2000 train. |
| `wilddash2_frames.csv` | 2760 frames: the first 2760 of the seed-0 permutation of the 4256 public frames. |

## Settings and seeds

**Image order.** `numpy.random.default_rng(0).permutation`. For Cityscapes, the first 50 frames of the permutation
of the sorted validation list form the calibration set.

**Resolution.**
- Cityscapes, ACDC and WildDash2 are processed at their native resolution.
- BDD100K is upscaled by a factor of 1.5.
- The minimum object area is 64 px at scale 1, i.e. 144 px for BDD100K.
- Proposal features and the matching of proposals to objects use a half-resolution grid. Each box is rasterised on
  that grid, and each mask is subsampled by taking every second pixel.

**Objects.**
- Instances are non-crowd things.
- On BDD100K, an object is a connected component of a thing class (`scipy.ndimage.label`, 4-connectivity).
- An object is *omitted* when at most 5% of its pixels carry the correct class. It is *confident* when at least 50%
  of its pixels are wrong with a calibrated confidence of at least 0.9.
- The *majority class* is the most frequent predicted class (`numpy.unique` + `argmax`). A tie goes to the lower
  train id, which means a stuff class.
- The joint criterion counts pixels that are both predicted as stuff and confident. It is computed by
  `object_records` and by the `relabel` mode.

**Proposals.**
- RT-DETR runs at a score threshold of 0.05, on the normal and on the gamma-corrected (x^0.5) image.
- At most 60 detector boxes per image and monitored segmenter (normal and gamma passes together; 51-57 on average) are kept, and these are the only candidates that the verifier and all compared methods rank. The cap on the total number of regions (`--max-props`, 90; 110 in the ACDC run v11) only limits regions from other sources (the second segmenter, depth-based and ML-rule regions), which are taken after the detector boxes and do not enter the main comparison; it never truncates the detector boxes.

**Matching.**
- Each proposal is assigned to the object of which it covers the largest fraction (`proposal_records`).
- A proposal hits that object when it covers at least 50% of it.
- Further hits on the same object are not counted as false alarms (`analysis/e18_comparison.py`, `found_at`).
- One-to-one matching and IoU criteria are in `analysis/e27_matching.py`.

**Verifier.**
- `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)`.
- Training uses all positives and a 25% random sample of the negatives (`numpy.random.default_rng(0)`).

**Bootstrap.**
- Paired differences: 2000 resamples of source images, seed 0 (`analysis/e10_e12.py`, `paired_ci`).
- Recall intervals: 500 resamples.
- Rank correlations: 1000 resamples. The intervals of rank-correlation differences are bias-corrected
  (`analysis/e35_rank_stability.py`, `analysis/e40_rank_extended.py`). In `e40_rank_extended.py` both correlations
  are resampled: mIoU is pooled from per-image confusion counts (`pilot.py --mode confusion --save-confusion`). For
  the 15 original configurations these counts come from a re-run with the same corruption seeds
  (`seed * 1000003 + k * 101 + condition index`, k = position of the frame in the original run); deterministic
  corruptions reproduce the stored predictions, while Gaussian and shot noise, fog, darkening and motion blur are a
  new draw with the same parameters (condition-level mIoU changes by at most 3.6 points). A two-level bootstrap
  also resamples the segmenters (draws with fewer than five distinct segmenters are skipped).
