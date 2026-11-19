"""Offline unit tests for pilot.py (no model download, no data)."""

import os

import numpy as np
import torch

import pilot


def test_labels_and_instances():
    inst = np.zeros((10, 10), dtype=np.int32)
    inst[:, :] = 7                    # road (labelId 7 -> trainId 0)
    inst[0:3, 0:3] = 24000            # person instance 0, area 9
    inst[5:10, 5:10] = 26001          # car instance 1, area 25
    inst[0, 9] = 24                   # person crowd region (id < 1000): not an object
    labels = pilot.labels_from_instances(inst)
    assert labels[0, 0] == 11 and labels[6, 6] == 13 and labels[9, 0] == 0 and labels[0, 9] == 11
    objs = pilot.instance_objects(inst, min_area=10)
    assert [(i, t) for i, t, _ in objs] == [(26001, 13)]


def test_object_records_flags_confident_omission():
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[5:10, 5:10] = 13
    mask = labels == 13
    pred = np.zeros_like(labels)            # the car is predicted as road everywhere
    conf = np.full(labels.shape, 0.95, dtype=np.float32)
    ent = np.full(labels.shape, 0.5, dtype=np.float32)
    ent[mask] = 0.1                          # less uncertain inside the car than elsewhere
    rows = pilot.object_records(pred, conf, ent, labels, [(26001, 13, mask)], conf_thr=0.9, omit_frac=0.05)
    r = rows[0]
    assert r["omitted"] and r["confident_omission"] and r["predicted_as"] == "road"
    assert r["entropy_percentile"] == 0.0


def test_corruptions_keep_shape_and_range():
    x = torch.rand(1, 3, 32, 64)
    gen = torch.Generator().manual_seed(0)
    for name in ("clean", "gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg"):
        y = pilot.corrupt(x, name, 3, gen)
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1


def test_temperature_recovers_scale():
    torch.manual_seed(0)
    labels = torch.randint(0, 5, (4000,))
    logits = torch.randn(4000, 5)
    logits[torch.arange(4000), labels] += 2.0
    t0 = pilot.fit_temperature(logits, labels)
    t3 = pilot.fit_temperature(logits * 3.0, labels)   # the same model, three times over-confident
    assert abs(t3 / t0 - 3.0) < 0.1


def test_ml_rule_and_true_class_probability():
    probs = torch.zeros(19, 4, 4)
    probs[2] = 0.9                      # building dominates
    probs[13] = 0.1                     # car is second everywhere
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:2, :2] = True
    prior = torch.full((19,), 0.01)
    prior[2] = 0.5                      # building is very frequent, car rare
    pred_ml = (probs / prior[:, None, None]).argmax(0)
    row = pilot.object_extras(probs, pred_ml, [mask], [13], omit_frac=0.05)[0]
    assert row["frac_true_ml"] == 1.0 and not row["omitted_ml"]          # 0.1/0.01 > 0.9/0.5
    assert abs(row["mean_p_true"] - 0.1) < 1e-6 and row["frac_true_in_top2"] == 1.0


def test_class_shift_and_quality():
    pred = torch.full((8, 8), 2)
    h = pilot.class_histogram(pred)
    assert float(h[2]) == 1.0
    assert pilot.js_divergence(h, h) < 1e-6 and pilot.js_divergence(h, torch.ones(19) / 19) > 0.3
    q = pilot.quality_features(torch.full((1, 3, 16, 16), 0.5))
    assert abs(q["q_brightness"] - 0.5) < 1e-6 and q["q_contrast"] < 1e-6 and q["q_sharpness"] < 1e-6


# ---------------------------------------------------------------- v6 / P4
def test_new_corruptions_keep_shape_and_range():
    x = torch.rand(1, 3, 32, 64)
    g, r = torch.Generator().manual_seed(0), np.random.default_rng(0)
    for name in ("shot_noise", "defocus_blur", "motion_blur", "fog", "dark", "gaussian_noise", "jpeg"):
        y = pilot.corrupt_v6(x, name, 3, g, r)
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1, name
        assert not torch.allclose(y, x), name
    assert float(pilot.corrupt_v6(x, "dark", 5, g, r).mean()) < 0.2 * float(x.mean()) + 0.02
    pf = pilot.plasma_fractal(64, 2.0, np.random.default_rng(1))
    assert pf.shape == (64, 64) and pf.min() == 0 and abs(pf.max() - 1) < 1e-9
    for k in (pilot.disk_kernel(3, 0.1), pilot.motion_kernel(10, 3, 30.0)):
        assert abs(k.sum() - 1) < 1e-5
    assert pilot.parse_conditions("clean,fog-3,gaussian_noise-5") == [("clean", 0), ("fog", 3), ("gaussian_noise", 5)]


def test_feature_density_scores_outliers_higher():
    gen = torch.Generator().manual_seed(0)
    feat = torch.randn(1, 8, 20, 20) * 0.1
    labels = torch.zeros(40, 40, dtype=torch.long)
    labels[:, 20:] = 1
    feat[:, 0, :, 10:] += 3.0                       # class 1 lives at +3 on dim 0
    dens = pilot.FeatureDensity(per_class=200)
    dens.add(feat, labels, gen)
    assert dens.fit() == 8
    test = feat.clone()
    test[:, 3, :5, :5] += 5.0                        # an out-of-distribution patch
    pred_low = torch.zeros(20, 20, dtype=torch.long)
    pred_low[:, 10:] = 1
    dmin, dpred = dens.score(test, pred_low)
    assert float(dmin[:5, :5].mean()) > 5 * float(dmin[10:, 12:].mean())
    wrong = torch.ones_like(pred_low)                # predicted class 1 where features are class 0
    _, dwrong = dens.score(feat, wrong)
    assert float(dwrong[:, :10].mean()) > float(dpred[:, :10].mean()) + 1


def test_depth_tophat_and_object_stats():
    disp = torch.zeros(40, 80)
    disp[:, :] = torch.linspace(0, 0.5, 40)[:, None]  # smooth ground plane
    disp[10:16, 30:36] += 0.6                          # a near object
    th = pilot.depth_tophat(disp, (15,))
    assert float(th[12, 32]) > 0.3 and float(th[30, 5]) < 0.05
    maps = torch.zeros(2, 10, 10)
    maps[0, :2, :2] = 1.0                              # signal 0 flags the object; signal 1 flags elsewhere
    maps[1, 8:, 8:] = 1.0
    obj = torch.zeros(10, 10, dtype=torch.bool)
    obj[:2, :2] = True
    st = pilot.object_signal_stats([obj], maps, ["a", "b"], budgets=(0.04,))
    assert st[0]["a_mean"] == 1.0 and st[0]["b_mean"] == 0.0
    assert st[0]["a_hit04"] == 0.0 or st[0]["a_hit04"] == 1.0
    m = torch.zeros(9, 9, dtype=torch.bool)
    m[3, 3] = True                                     # 1-px object lost by stride-2 sampling
    assert bool(pilot.downsample_mask(m).any())
    p = torch.softmax(torch.randn(5, 4, 4), 0)
    assert float(pilot.js_map(p, p).max()) < 1e-6 and float(pilot.js_map(p, p.flip(0)).mean()) > 0


def test_panoptic_to_instances():
    pan = np.zeros((6, 6, 3), dtype=np.uint8)
    pan[:, :] = (7, 0, 0)                    # segment id 7 = road
    pan[0:2, 0:2] = (1, 1, 0)                # id 257 = person instance
    pan[4:6, 4:6] = (2, 0, 0)                # id 2 = crowd of cars
    pan[5, 0] = (0, 0, 0)                    # unlabelled
    segs = [{"id": 7, "category_id": 7, "iscrowd": 0}, {"id": 257, "category_id": 24, "iscrowd": 0},
            {"id": 2, "category_id": 26, "iscrowd": 1}]
    inst = pilot.panoptic_to_instances(pan, segs)
    assert inst[0, 0] == 24000 and inst[3, 3] == 7 and inst[5, 5] == 26 and inst[5, 0] == 0
    labels = pilot.labels_from_instances(inst)
    assert labels[0, 0] == 11 and labels[3, 3] == 0 and labels[5, 5] == 13 and labels[5, 0] == 255
    assert [(i, t) for i, t, _ in pilot.instance_objects(inst, 1)] == [(24000, 11)]


def test_proposals_from_detector_and_other_model():
    pred = torch.zeros(20, 20, dtype=torch.long)            # this model: all road
    other = torch.zeros(20, 20, dtype=torch.long)
    other[2:8, 2:8] = 13                                     # another model sees a car here
    boxes = [(0.9, "person", 10, 10, 16, 18)]
    props = pilot.proposal_masks(pred, {"m2": other}, boxes, min_area=4)
    assert [m["source"] for _, m in props] == ["det", "xmodel"]
    probs = torch.softmax(torch.randn(19, 20, 20), 0)
    stack = torch.rand(2, 20, 20)
    gt = torch.zeros(20, 20, dtype=torch.bool)
    gt[2:8, 2:8] = True
    recs = pilot.proposal_records(props, probs, pred, {"m2": other}, stack, ["a", "b"], [gt], [26000])
    assert recs[1]["gt_instance_id"] == 26000 and abs(recs[1]["gt_cover"] - 1) < 1e-6 and abs(recs[1]["gt_iou"] - 1) < 1e-6
    assert recs[0]["gt_cover"] == 0 and recs[1]["p_self_stuff_frac"] == 1.0 and recs[1]["p_n_other_thing_mean"] == 1.0


def test_box_merge_and_crops():
    a = (0.9, "car", 0, 0, 10, 10, "rtdetr")
    a_dup = (0.5, "car", 1, 0, 10, 10, "rtdetr_gamma")
    b = (0.2, "person", 20, 20, 30, 40, "owl")
    b_dup = (0.15, "person", 0, 0, 10, 10, "owl")          # duplicates a across groups
    out = pilot.merge_boxes([[a, a_dup], [b, b_dup]], [5, 5])
    assert out == [a, b]
    assert pilot.merge_boxes([[a, b]], [1]) == [a]          # quota
    c = pilot.square_crop((90, 10, 100, 20), H=50, W=100, factor=1.5, min_side=30)
    assert c[2] - c[0] == c[3] - c[1] == 30 and c[2] <= 100 and c[0] >= 0
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[2:5, 3:7] = True
    assert pilot.mask_box(m) == (3, 2, 7, 5)


class _ZoomSeg:
    """Dummy segmenter: predicts 'car' (13) everywhere -> zoom sees a thing."""

    def __call__(self, x, want_feat=True):
        lg = torch.zeros(x.shape[0], pilot.N_CLASSES, *x.shape[-2:])
        lg[:, 13] = 5.0
        return lg, None


def test_second_opinion_zoom_features():
    x = torch.rand(1, 3, 64, 96)
    pred_h = torch.zeros(32, 48, dtype=torch.long)           # all road at half-res
    m1 = torch.zeros(32, 48, dtype=torch.bool)
    m1[5:10, 5:12] = True
    m2 = torch.zeros(32, 48, dtype=torch.bool)
    m2[20:30, 30:40] = True
    props = [(m1, {"source": "det"}), (m2, {"source": "xmodel"})]
    pilot.second_opinion(props, x, pred_h, _ZoomSeg(), 1.0, None, {}, zoom_k=1, zoom_size=32)
    done = [p[1]["zoom_done"] for p in props]
    assert sorted(done) == [0.0, 1.0]
    z = [p[1] for p in props if p[1]["zoom_done"] == 1.0][0]
    assert z["zoom_thing_frac"] == 1.0 and z["zoom_top"] == 13 and z["zoom_thing_prob"] > 0.9


def test_semantic_to_instances_components():
    tid = np.full((10, 12), 0, dtype=np.uint8)       # road
    tid[1:3, 1:3] = 13                               # car A
    tid[6:9, 8:11] = 13                              # car B (separate component)
    tid[0, 11] = 255
    inst = pilot.semantic_to_instances(tid)
    ids = sorted(set(inst[inst >= 1000].tolist()))
    assert len(ids) == 2 and all(i // 1000 == 26 for i in ids)     # car labelId 26
    assert inst[0, 0] == 7 and inst[0, 11] == 0                      # road labelId 7, ignore -> 0
    assert (pilot.labels_from_instances(inst) == np.where(tid == 255, 255, tid)).all()


def test_wilddash_convert_and_load(tmp_path):
    import importlib.util
    from PIL import Image
    spec = importlib.util.spec_from_file_location(
        "prepare_wilddash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_prep", "prepare_wilddash.py"))
    wd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wd)
    ids = np.zeros((8, 10), dtype=np.int64)
    ids[:, :] = 1                     # road segment
    ids[1:3, 1:3] = 2                 # van -> car instance
    ids[5:7, 6:9] = 3                 # pickup -> truck instance
    ids[0, 9] = 4                     # billboard -> ignore
    ids[7, 0] = 5                     # road marking -> road
    rgb = np.stack([ids % 256, (ids // 256) % 256, ids // 65536], -1).astype(np.uint8)
    segs = [{"id": 1, "category_id": 7}, {"id": 2, "category_id": 35}, {"id": 3, "category_id": 34},
            {"id": 4, "category_id": 36}, {"id": 5, "category_id": 38}]
    inst = wd.convert(rgb, segs)
    assert inst[1, 1] == 26000 and inst[5, 6] == 27000 and inst[0, 9] == 0 and inst[7, 0] == 7 and inst[0, 0] == 7
    root = tmp_path / "wd"
    (root / "labels").mkdir(parents=True)
    (root / "images").mkdir()
    Image.fromarray(inst).save(root / "labels" / "us0001_100000_instanceIds.png")
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(root / "images" / "us0001_100000.jpg")
    (root / "validation.txt").write_text("us0001_100000\n")
    items = pilot.list_wd(str(root))
    assert len(items) == 1 and items[0]["cond"] == "wd_val"
    x, inst2, inv = pilot.load_wd(items[0], 1.0)
    assert x.shape == (1, 3, 8, 10) and inv is None and (inst2 == inst.astype(np.int32)).all()
    lab = pilot.labels_from_instances(inst2)
    assert lab[1, 1] == 13 and lab[5, 6] == 14 and lab[0, 9] == 255   # car / truck trainIds, ignore


def test_mmseg_deeplabv3plus_structure():
    """Key names and shapes match the OpenMMLab checkpoint layout (no download)."""
    bb, head = pilot._ResNetV1cD8(), pilot._DLV3PHead()
    kb, kh = set(bb.state_dict()), set(head.state_dict())
    for k in ("stem.6.weight", "stem.7.running_var", "layer1.0.downsample.0.weight", "layer3.22.conv3.weight",
              "layer4.2.bn3.bias"):
        assert k in kb
    for k in ("image_pool.1.conv.weight", "aspp_modules.0.bn.weight", "aspp_modules.3.depthwise_conv.conv.weight",
              "bottleneck.conv.weight", "c1_bottleneck.bn.running_mean", "sep_bottleneck.1.pointwise_conv.conv.weight",
              "conv_seg.bias"):
        assert k in kh
    assert head.state_dict()["bottleneck.conv.weight"].shape == (512, 2560, 3, 3)
    assert bb.layer3[0].conv2.dilation == (1, 1) and bb.layer3[1].conv2.dilation == (2, 2)
    assert bb.layer4[0].conv2.dilation == (2, 2) and bb.layer4[2].conv2.dilation == (4, 4)
    bb.eval(); head.eval()
    with torch.no_grad():
        c1, c4 = bb(torch.rand(1, 3, 64, 128))
        logits, feat = head(c1, c4)
    assert c1.shape[-2:] == (16, 32) and c4.shape[-2:] == (8, 16) and logits.shape == (1, 19, 16, 32) and feat.shape[1] == 512
