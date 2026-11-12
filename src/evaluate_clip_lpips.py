"""CLIP prompt alignment + LPIPS diversity (+ LPIPS distance to the reference).

Usage (Colab):  python src/evaluate_clip_lpips.py --config configs/experiment.yaml

Measures
- CLIP similarity: Measures how closely a generated image matches its
  original text prompt by comparing their CLIP embeddings (cosine similarity).
  Higher scores indicate stronger image-text agreement, but do not prove
  that specific visual attributes are correctly represented.
- LPIPS diversity: Measures the average perceptual difference between
  all pairs of generated images belonging to the same prompt and
  configuration, using the AlexNet-based LPIPS model.
  Higher scores indicate greater visual diversity, not necessarily
  better image quality.
- LPIPS to reference: Measures the perceptual difference between each
  main generated image and its corresponding 100-step DDIM reference,
  using the same prompt and initial seed.
  Lower scores indicate more visually similar final images.
  This is a comparison of final outputs, not of intermediate
  generation trajectories.

Rule: images flagged by the safety checker (saved as black) are kept in the counts but
excluded from all metrics.
Outputs: <out_dir>/metrics/*.csv
"""
import argparse, itertools, os
import numpy as np, pandas as pd, torch, yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import lpips
import transformers
from stats_utils import boot_means, boot_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out = cfg["paths"]["out_dir"]
    img_dir, met_dir = os.path.join(out, "images"), os.path.join(out, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    device = "cuda"

    m = pd.read_json(os.path.join(out, "manifest.jsonl"), lines=True).drop_duplicates("file", keep="last")
    m["config"] = m.scheduler + "_" + m.steps.astype(str)
    ok = m[~m.nsfw_flagged].reset_index(drop=True)

    sched_names = list(cfg["schedulers"])
    cols = [f"{s}_{n}" for s in sched_names for n in cfg["steps"]]
    ref = cfg.get("reference", {})
    if ref.get("enabled"):
        cols.append(f'{ref["scheduler"]}_{ref["steps"]}')
    cols = [c for c in cols if c in set(m.config)]

    cache = {}
    def load(file):
        if file not in cache:
            cache[file] = torch.from_numpy(np.asarray(Image.open(os.path.join(img_dir, file)).convert("RGB"))).permute(2, 0, 1).clone()
        return cache[file]

    # ---------------- CLIP ----------------
    print("CLIP ...", args.clip_model)
    clip = CLIPModel.from_pretrained(args.clip_model, dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    sims = []
    with torch.no_grad():
        for i in range(0, len(ok), args.batch):
            chunk = ok.iloc[i:i + args.batch]
            imgs = [Image.open(os.path.join(img_dir, f)).convert("RGB") for f in chunk.file]
            texts = [cfg["prompts"][p] for p in chunk.prompt_id]
            inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True)
            inp = {k: (v.to(device).half() if k == "pixel_values" else v.to(device)) for k, v in inp.items()}
            o = clip(**inp)
            ie, te = o.image_embeds.float(), o.text_embeds.float()
            ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
            sims += (ie * te).sum(-1).cpu().tolist()
    ok["clip_cos"] = sims
    del clip; torch.cuda.empty_cache()
    ok[["file", "role", "scheduler", "steps", "prompt_id", "seed", "config", "clip_cos"]].to_csv(
        os.path.join(met_dir, "clip_per_image.csv"), index=False)

    # ---------------- LPIPS ----------------
    print("LPIPS ...")
    net = lpips.LPIPS(net="alex").to(device).eval()
    def to_in(t):  # uint8 CxHxW -> float [-1,1]
        return t.float().div(127.5).sub(1.0)

    def lp(pairs_a, pairs_b):
        with torch.no_grad():
            a = torch.stack([to_in(x) for x in pairs_a]).to(device)
            b = torch.stack([to_in(x) for x in pairs_b]).to(device)
            return net(a, b).flatten().cpu().numpy()

    div_rows = []
    for (cfgname, pid), g in ok.groupby(["config", "prompt_id"]):
        files = g.sort_values("seed").file.tolist()
        pairs = list(itertools.combinations(files, 2))
        if len(pairs) == 0:
            div_rows.append(dict(config=cfgname, prompt_id=pid, lpips_div=np.nan, n_images=len(files)))
            continue
        d = lp([load(a) for a, _ in pairs], [load(b) for _, b in pairs])
        div_rows.append(dict(config=cfgname, prompt_id=pid, lpips_div=float(d.mean()), n_images=len(files)))
    div = pd.DataFrame(div_rows)
    div.to_csv(os.path.join(met_dir, "lpips_diversity_per_group.csv"), index=False)

    ref_rows = []
    if ref.get("enabled"):
        refs = ok[ok.role == "reference"].set_index(["prompt_id", "seed"]).file.to_dict()
        for _, r in ok[ok.role == "main"].iterrows():
            rf = refs.get((r.prompt_id, r.seed))
            if rf is None:
                ref_rows.append(dict(file=r.file, config=r.config, prompt_id=r.prompt_id, seed=r.seed, lpips_to_ref=np.nan))
                continue
            d = lp([load(r.file)], [load(rf)])[0]
            ref_rows.append(dict(file=r.file, config=r.config, prompt_id=r.prompt_id, seed=r.seed, lpips_to_ref=float(d)))
    refdf = pd.DataFrame(ref_rows)
    if len(refdf):
        refdf.to_csv(os.path.join(met_dir, "lpips_to_reference_per_image.csv"), index=False)

    # ---------------- counts ----------------
    counts = m.groupby(["role", "config"]).agg(generated=("file", "count"), flagged=("nsfw_flagged", "sum")).reset_index()
    ev = ok.groupby(["role", "config"]).size().rename("evaluated").reset_index()
    counts = counts.merge(ev, on=["role", "config"], how="left").fillna({"evaluated": 0})
    counts.to_csv(os.path.join(met_dir, "counts.csv"), index=False)

    # ---------------- summaries (bootstrap over matched units) ----------------
    clip_piv = ok.pivot_table(index=["prompt_id", "seed"], columns="config", values="clip_cos").reindex(columns=cols)
    div_piv = div.pivot_table(index="prompt_id", columns="config", values="lpips_div").reindex(columns=cols)
    clip_sum = boot_means(clip_piv, args.boot)
    div_sum = boot_means(div_piv, args.boot)
    clip_sum.to_csv(os.path.join(met_dir, "clip_summary.csv"), index=False)
    div_sum.to_csv(os.path.join(met_dir, "lpips_diversity_summary.csv"), index=False)

    ref_sum = None
    if len(refdf):
        ref_piv = refdf.pivot_table(index=["prompt_id", "seed"], columns="config", values="lpips_to_ref")
        ref_piv = ref_piv.reindex(columns=[c for c in cols if c in ref_piv.columns])
        ref_sum = boot_means(ref_piv, args.boot)
        ref_sum.to_csv(os.path.join(met_dir, "lpips_to_reference_summary.csv"), index=False)

    # paired differences
    A, B = sched_names[0], sched_names[1]
    st = cfg["steps"]; lo, mid, hi = st[0], st[1], st[-1]
    comps = [(f"{B}_{n}", f"{A}_{n}") for n in st] + [
        (f"{B}_{mid}", f"{A}_{hi}"), (f"{B}_{lo}", f"{A}_{hi}"),
        (f"{A}_{mid}", f"{A}_{hi}"), (f"{B}_{mid}", f"{B}_{hi}")]
    rows = []
    for metric, piv in [("clip_cos", clip_piv), ("lpips_div", div_piv)]:
        for a, b in comps:
            if a in piv.columns and b in piv.columns:
                rows.append(dict(metric=metric, **boot_diff(piv, a, b, args.boot)))
    pd.DataFrame(rows).to_csv(os.path.join(met_dir, "paired_differences.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n== counts ==\n", counts.to_string(index=False))
    print("\n== CLIP cosine (mean, 95% CI over (prompt,seed) units) ==\n", clip_sum.round(4).to_string(index=False))
    print("\n== LPIPS diversity (mean over prompts, 95% CI) ==\n", div_sum.round(4).to_string(index=False))
    if ref_sum is not None:
        print("\n== LPIPS to reference ==\n", ref_sum.round(4).to_string(index=False))
    print("\n== paired differences (a - b) ==\n", pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nsaved to {met_dir}  | transformers {transformers.__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
