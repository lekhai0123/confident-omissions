# Reproducibility notes

Everything needed to rebuild the tables from the datasets, apart from the datasets themselves.

## Environment

All experiments ran as Kaggle script kernels on one NVIDIA Tesla T4 (16 GB):

- Python 3.13.16
- PyTorch 2.11.0+cu130
- transformers 5.20.2
- scikit-learn and scipy as in `requirements.txt`

Segmenter and detector inference runs in float16 autocast. Unit tests (`pytest notebook/test_pilot.py`) run on CPU.

Analysis scripts run on CPU with Python 3.12 (Windows) or 3.13. A re-run on another GPU reproduces the stored
per-object statistics up to float16 rounding: on a GTX 1080 Ti, the largest difference in the correctly predicted
pixel share of an object was 0.4%, and no object changed label.

## Model weights

| Role | Identifier |
|---|---|
| Segmenters (Hugging Face) | `nvidia/segformer-b{0..5}-finetuned-cityscapes-1024-1024`; `nvidia/segformer-b0-finetuned-cityscapes-{512-1024,640-1280,768-768}`; `facebook/maskformer-resnet101-cityscapes`; `facebook/mask2former-swin-{tiny,small,large}-cityscapes-semantic` |
| Segmenters (OpenMMLab) | `deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth`, `deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth` (URLs in `notebook/pilot.py`, `MMSEG_URLS`) |
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
- At most 60 detector boxes and at most 90 proposals (110 in the ACDC run) are kept. Detector boxes are taken first.

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
  (`analysis/e35_rank_stability.py`).
