import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
d = sys.argv[1]
NAMES="road sidewalk building wall fence pole light sign vegetation terrain sky person rider car truck bus train motorcycle bicycle".split()
for tag in ("segformer-b0","segformer-b2"):
    img = pd.read_csv(f"{d}/images_{tag}.csv")
    print("=====", tag)
    # dominant predicted class per condition
    hc=[f"hist_{i}" for i in range(19)]
    for c in sorted(img.condition.unique()):
        g=img[img.condition==c]; m=g[hc].mean().to_numpy()
        top=np.argsort(m)[::-1][:3]
        dom=g[hc].to_numpy().max(1)
        print(f"  {c:17s} top classes {[(NAMES[t],round(m[t],2)) for t in top]} share imgs dominant>=60%: {(dom>=.6).mean():.2f} pos rate {g.has_confident_omission.mean():.2f}")
    # within-condition AUROC table
    S={"entropy":("mean_entropy",1),"max_ent":("max_entropy",1),"margin":("one_minus_mean_margin",1),"js_shift":("class_shift_js",1),"blur":("q_sharpness",-1)}
    rows=[]
    for c in sorted(img.condition.unique()):
        g=img[img.condition==c]; y=g.has_confident_omission.astype(int)
        if 3<=y.sum()<=len(y)-3:
            rows.append({"cond":c,"npos":int(y.sum()),**{k:round(roc_auc_score(y,s*g[col]),3) for k,(col,s) in S.items()}})
    t=pd.DataFrame(rows); print(t.to_string(index=False))
    nn=t[~t.cond.str.startswith("gaussian_noise")&(t.cond!="clean")]
    print("  mean within-condition AUROC (corrupted, no noise):", nn[list(S)].mean().round(3).to_dict())
    # pooled corrupted without noise
    g=img[(img.condition!="clean")&~img.condition.str.startswith("gaussian_noise")]; y=g.has_confident_omission.astype(int)
    print("  pooled corrupted no-noise, npos", int(y.sum()), {k:round(roc_auc_score(y,s*g[col]),3) for k,(col,s) in S.items()})
    # circularity: does JS just detect the condition? AUROC of js for corrupted-vs-clean
    g=img[~img.condition.str.startswith("gaussian_noise")]; y=(g.condition!="clean").astype(int)
    print("  JS shift as corruption detector (any non-noise corruption vs clean):", round(roc_auc_score(y,g.class_shift_js),3))
