"""Face detection/alignment (InsightFace) + identity diversity (ElasticFace embeddings).

Usage (Colab):
  python src/evaluate_faces.py --config configs/experiment.yaml \
      --weights /content/drive/MyDrive/models/295672backbone.pth \
      --elasticface_repo /content/ElasticFace

Pipeline (the SAME for every image, report it):
  1. InsightFace buffalo_l detector (SCRFD, det_thresh=0.5, det_size 640) -> highest-score face.
  2. Alignment: insightface norm_crop with the 5 detected landmarks -> 112x112.
  3. ElasticFace IResNet-100 (pretrained, MS1MV2) -> 512-d embedding, L2-normalised.
     Preprocessing as in ArcFace-style models: RGB, (x/255 - 0.5)/0.5.
Measures
  - Detection outcome per image (no face / one face / several faces) -> failures are REPORTED.
  - Identity diversity: mean pairwise cosine distance (1 - cos) between embeddings of the same
    prompt+config across seeds. Higher = more different faces. Embedding similarity is used
    only between generated images; it is not proof of identity and nobody is identified.
  - Identity distance to reference: cosine distance to the reference image with the same
    prompt and seed (matched initial noise). Descriptive, compares final outputs only.
Rule: safety-checker-flagged (black) images are counted but excluded.
Outputs: <out_dir>/metrics/face_*.csv, identity_*.csv, embeddings.npz
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd, torch, yaml
from PIL import Image
from stats_utils import boot_means, boot_diff


def load_elasticface(weights, repo, device):
    sys.path.insert(0, repo)
    try:
        from backbones.iresnet import iresnet100
    except Exception as e:
        raise SystemExit(f"Could not import iresnet100 from {repo}/backbones: {e}\n"
                         "Check the repo path and the file names in <repo>/backbones.")
    try:
        net = iresnet100(num_features=512)
    except TypeError:
        net = iresnet100()
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    res = net.load_state_dict(ckpt, strict=False)
    print("missing keys:", res.missing_keys, "| unexpected keys:", res.unexpected_keys)
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit("Weights do not match the backbone exactly - stop and check the model/weights.")
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--elasticface_repo", default="/content/ElasticFace")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--save_crops", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out = cfg["paths"]["out_dir"]
    img_dir, met_dir = os.path.join(out, "images"), os.path.join(out, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    crop_dir = os.path.join(met_dir, "aligned_crops")
    if args.save_crops:
        os.makedirs(crop_dir, exist_ok=True)
    device = "cuda"

    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    import insightface, onnxruntime
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    net = load_elasticface(args.weights, args.elasticface_repo, device)

    m = pd.read_json(os.path.join(out, "manifest.jsonl"), lines=True).drop_duplicates("file", keep="last")
    m["config"] = m.scheduler + "_" + m.steps.astype(str)

    n_faces, det_score, embs = {}, {}, {}
    for i, r in enumerate(m.itertuples()):
        if r.nsfw_flagged:
            continue
        rgb = np.asarray(Image.open(os.path.join(img_dir, r.file)).convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        faces = app.get(bgr)
        n_faces[r.file] = len(faces)
        if not faces:
            det_score[r.file] = np.nan
            continue
        f = max(faces, key=lambda x: x.det_score)
        det_score[r.file] = float(f.det_score)
        crop_bgr = face_align.norm_crop(bgr, landmark=f.kps, image_size=112)
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        if args.save_crops:
            Image.fromarray(crop_rgb).save(os.path.join(crop_dir, r.file))
        x = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            e = net(x).float()
        embs[r.file] = torch.nn.functional.normalize(e, dim=1)[0].cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(m)}")

    m["n_faces"] = m.file.map(n_faces)            # NaN for flagged (not processed)
    m["det_score"] = m.file.map(det_score)
    m["embedded"] = m.file.isin(embs)
    m[["file", "role", "scheduler", "steps", "prompt_id", "seed", "config", "nsfw_flagged",
       "n_faces", "det_score", "embedded"]].to_csv(os.path.join(met_dir, "face_per_image.csv"), index=False)
    np.savez(os.path.join(met_dir, "embeddings.npz"), files=np.array(list(embs)), emb=np.stack(list(embs.values())))

    counts = m.groupby(["role", "config"]).agg(
        generated=("file", "count"), flagged=("nsfw_flagged", "sum"),
        no_face=("n_faces", lambda s: int((s == 0).sum())),
        multi_face=("n_faces", lambda s: int((s > 1).sum())),
        embedded=("embedded", "sum"), mean_det_score=("det_score", "mean")).reset_index()
    counts.to_csv(os.path.join(met_dir, "face_counts.csv"), index=False)

    # identity diversity per (config, prompt)
    em = m[m.embedded]
    div_rows = []
    for (cname, pid), g in em.groupby(["config", "prompt_id"]):
        E = np.stack([embs[f] for f in g.sort_values("seed").file])
        if len(E) < 2:
            div_rows.append(dict(config=cname, prompt_id=pid, id_div=np.nan, n_images=len(E)))
            continue
        iu = np.triu_indices(len(E), k=1)
        d = 1.0 - (E @ E.T)[iu]
        div_rows.append(dict(config=cname, prompt_id=pid, id_div=float(d.mean()), n_images=len(E)))
    div = pd.DataFrame(div_rows)
    div.to_csv(os.path.join(met_dir, "identity_diversity_per_group.csv"), index=False)

    # identity distance to reference
    ref = cfg.get("reference", {})
    refdf = pd.DataFrame()
    if ref.get("enabled"):
        refs = em[em.role == "reference"].set_index(["prompt_id", "seed"]).file.to_dict()
        rows = []
        for r in em[em.role == "main"].itertuples():
            rf = refs.get((r.prompt_id, r.seed))
            d = np.nan if rf is None else float(1.0 - embs[r.file] @ embs[rf])
            rows.append(dict(file=r.file, config=r.config, prompt_id=r.prompt_id, seed=r.seed, id_dist_to_ref=d))
        refdf = pd.DataFrame(rows)
        refdf.to_csv(os.path.join(met_dir, "identity_to_reference_per_image.csv"), index=False)

    # summaries
    sched_names = list(cfg["schedulers"])
    cols = [f"{s}_{n}" for s in sched_names for n in cfg["steps"]]
    if ref.get("enabled"):
        cols.append(f'{ref["scheduler"]}_{ref["steps"]}')
    cols = [c for c in cols if c in set(m.config)]
    div_piv = div.pivot_table(index="prompt_id", columns="config", values="id_div").reindex(columns=cols)
    div_sum = boot_means(div_piv, args.boot)
    div_sum.to_csv(os.path.join(met_dir, "identity_diversity_summary.csv"), index=False)
    ref_sum = None
    if len(refdf):
        rp = refdf.pivot_table(index=["prompt_id", "seed"], columns="config", values="id_dist_to_ref")
        rp = rp.reindex(columns=[c for c in cols if c in rp.columns])
        ref_sum = boot_means(rp, args.boot)
        ref_sum.to_csv(os.path.join(met_dir, "identity_to_reference_summary.csv"), index=False)

    A, B = sched_names[0], sched_names[1]
    st = cfg["steps"]; lo, mid, hi = st[0], st[1], st[-1]
    comps = [(f"{B}_{n}", f"{A}_{n}") for n in st] + [
        (f"{B}_{mid}", f"{A}_{hi}"), (f"{B}_{lo}", f"{A}_{hi}"),
        (f"{A}_{mid}", f"{A}_{hi}"), (f"{B}_{mid}", f"{B}_{hi}")]
    rows = []
    for metric, piv in [("id_div", div_piv)] + ([("id_dist_to_ref", rp)] if ref_sum is not None else []):
        for a, b in comps:
            if a in piv.columns and b in piv.columns:
                rows.append(dict(metric=metric, **boot_diff(piv, a, b, args.boot)))
    pd.DataFrame(rows).to_csv(os.path.join(met_dir, "identity_paired_differences.csv"), index=False)

    pd.set_option("display.width", 170)
    print("\n== face detection / embedding counts ==\n", counts.round(3).to_string(index=False))
    print("\n== identity diversity (1-cos, mean over prompts, 95% CI) ==\n", div_sum.round(4).to_string(index=False))
    if ref_sum is not None:
        print("\n== identity distance to reference (1-cos) ==\n", ref_sum.round(4).to_string(index=False))
    print("\n== paired differences (a - b) ==\n", pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nsaved to {met_dir} | insightface {insightface.__version__}, onnxruntime {onnxruntime.__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
