"""Image grids, failure-case figure, combined results table and summary plots.
No GPU needed; reads what the earlier scripts saved.

Usage (Colab):  python src/make_figures.py --config configs/experiment.yaml --faces_dir metrics_pad50
Outputs in <out_dir>/figures/:
  grid_<prompt>.png          rows = seeds, columns = configs (+ reference); red border = flagged
                             by the safety checker, orange border = no face detected
  failure_cases.png          failed images (top) vs same prompt/seed at the second scheduler, mid steps (bottom)
  failure_cases.csv
  results_table.csv / .md    one row per config: runtime, CLIP, LPIPS and identity metrics, failures
  summary_plots.png          runtime vs convergence / CLIP, and diversity vs steps
"""
import argparse, os
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PRETTY = {"ddim": "DDIM", "dpmpp_2m": "DPM++ 2M"}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def split_cfg(c):
    s, n = c.rsplit("_", 1)
    return s, int(n)


def pretty(c):
    s, n = split_cfg(c)
    return f"{PRETTY.get(s, s)} {n}"


def img_path(img_dir, role, cfgname, pid, seed):
    s, n = split_cfg(cfgname)
    return os.path.join(img_dir, f"{role}_{s}_s{n}_{pid}_seed{seed}.png")


def make_grid(pid, cols, ref_cfg, seeds, img_dir, flagged, noface, T=192):
    hdr, lab = 36, 70
    W, H = lab + T * len(cols), hdr + T * len(seeds)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f = font(16)
    for j, c in enumerate(cols):
        d.text((lab + j * T + 6, 8), pretty(c) + (" (ref)" if c == ref_cfg else ""), fill="black", font=f)
    for i, seed in enumerate(seeds):
        d.text((6, hdr + i * T + T // 2 - 8), f"seed {seed}", fill="black", font=f)
        for j, c in enumerate(cols):
            role = "reference" if c == ref_cfg else "main"
            p = img_path(img_dir, role, c, pid, seed)
            x0, y0 = lab + j * T, hdr + i * T
            if os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize((T, T)), (x0, y0))
            fn = os.path.basename(p)
            if fn in flagged:
                d.rectangle([x0, y0, x0 + T - 1, y0 + T - 1], outline=(220, 30, 30), width=5)
            elif fn in noface:
                d.rectangle([x0, y0, x0 + T - 1, y0 + T - 1], outline=(255, 140, 0), width=5)
    return canvas


def fmt(row, k="mean", lo="ci_lo", hi="ci_hi", nd=3):
    if row is None or pd.isna(row[k]):
        return ""
    return f"{row[k]:.{nd}f} [{row[lo]:.{nd}f}, {row[hi]:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--faces_dir", default="metrics_pad50")
    ap.add_argument("--face_set", default="common", help="'common' or 'all' identity summaries")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["paths"]["out_dir"]
    img_dir, fig_dir = os.path.join(out, "images"), os.path.join(out, "figures")
    met, fmet = os.path.join(out, "metrics"), os.path.join(out, args.faces_dir)
    os.makedirs(fig_dir, exist_ok=True)

    sn = list(cfg["schedulers"])
    main_cols = [f"{s}_{n}" for s in sn for n in cfg["steps"]]
    ref = cfg.get("reference", {})
    ref_cfg = f'{ref["scheduler"]}_{ref["steps"]}' if ref.get("enabled") else None
    cols = main_cols + ([ref_cfg] if ref_cfg else [])

    face = pd.read_csv(os.path.join(fmet, "face_per_image.csv"))
    flagged = set(face[face.nsfw_flagged].file)
    noface = set(face[(~face.nsfw_flagged) & (face.n_faces == 0)].file)

    # ---- grids ----
    for pid in cfg["prompts"]:
        make_grid(pid, cols, ref_cfg, cfg["seeds"], img_dir, flagged, noface).save(os.path.join(fig_dir, f"grid_{pid}.png"))
    print("saved grids:", len(cfg["prompts"]))

    # ---- failure cases ----
    fails = face[face.role == "main"][lambda d: d.file.isin(flagged | noface)].copy()
    fails["failure"] = np.where(fails.file.isin(flagged), "flagged (black image)", "no face detected")
    fails[["file", "config", "prompt_id", "seed", "failure"]].to_csv(os.path.join(fig_dir, "failure_cases.csv"), index=False)
    if len(fails):
        T, lab, hdr = 224, 0, 54
        cmp_cfg = f"{sn[1]}_{cfg['steps'][1]}" if len(sn) > 1 else main_cols[0]
        W, H = T * len(fails), hdr + 2 * T
        canvas = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(canvas); f = font(14)
        for k, r in enumerate(fails.itertuples()):
            d.text((k * T + 4, 4), f"{pretty(r.config)} | {r.prompt_id} seed {r.seed}", fill="black", font=f)
            d.text((k * T + 4, 24), r.failure[:34], fill=(200, 30, 30), font=f)
            p0 = img_path(img_dir, "main", r.config, r.prompt_id, r.seed)
            p1 = img_path(img_dir, "main", cmp_cfg, r.prompt_id, r.seed)
            for row, p in enumerate([p0, p1]):
                if os.path.exists(p):
                    canvas.paste(Image.open(p).convert("RGB").resize((T, T)), (k * T, hdr + row * T))
        canvas.save(os.path.join(fig_dir, "failure_cases.png"))
        print("failure cases:", len(fails))

    # ---- combined table ----
    tm = pd.read_csv(os.path.join(out, "timing", "timing_summary.csv"))
    tm["config"] = tm.scheduler + "_" + tm.steps.astype(str)
    tm = tm.set_index("config")
    S = lambda name, d=met: pd.read_csv(os.path.join(d, name)).set_index("config")
    clip, ldiv, lref = S("clip_summary.csv"), S("lpips_diversity_summary.csv"), S("lpips_to_reference_summary.csv")
    idiv = S(f"identity_diversity_summary_{args.face_set}.csv", fmet)
    iref = S(f"identity_to_reference_summary_{args.face_set}.csv", fmet)
    counts = pd.read_csv(os.path.join(fmet, "face_counts.csv")).set_index("config")

    rows = []
    for c in cols:
        g = lambda df: df.loc[c] if c in df.index else None
        rows.append({
            "config": pretty(c),
            "runtime_median_s": f"{tm.loc[c, 'median']:.2f}" if c in tm.index else "",
            "CLIP cos": fmt(g(clip), nd=4),
            "LPIPS diversity": fmt(g(ldiv), nd=3),
            "LPIPS to ref": fmt(g(lref), nd=3),
            f"ID diversity ({args.face_set})": fmt(g(idiv), nd=3),
            f"ID dist to ref ({args.face_set})": fmt(g(iref), nd=3),
            "flagged": int(counts.loc[c, "flagged"]), "no face": int(counts.loc[c, "no_face"]),
            "generated": int(counts.loc[c, "generated"]),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(fig_dir, "results_table.csv"), index=False)
    with open(os.path.join(fig_dir, "results_table.md"), "w") as fh:
        fh.write("| " + " | ".join(tab.columns) + " |\n|" + "---|" * len(tab.columns) + "\n")
        for _, r in tab.iterrows():
            fh.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    print(tab.to_string(index=False))

    # ---- plots ----
    colors = {sn[0]: "#1f77b4", sn[1]: "#d62728"} if len(sn) > 1 else {sn[0]: "#1f77b4"}
    marks = {sn[0]: "o", sn[1]: "s"} if len(sn) > 1 else {sn[0]: "o"}

    def panel(ax, summ, ylabel, title, refline=None):
        for s in sn:
            xs, ys, lo, hi, ns = [], [], [], [], []
            for n in cfg["steps"]:
                c = f"{s}_{n}"
                if c in summ.index and c in tm.index:
                    xs.append(tm.loc[c, "median"]); ys.append(summ.loc[c, "mean"])
                    lo.append(summ.loc[c, "mean"] - summ.loc[c, "ci_lo"]); hi.append(summ.loc[c, "ci_hi"] - summ.loc[c, "mean"]); ns.append(n)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colors[s], marker=marks[s], capsize=3, label=PRETTY.get(s, s))
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color=colors[s])
        if refline is not None:
            ax.axhline(refline, color="gray", ls="--", lw=1, label="100-step reference")
        ax.set_xlabel("median runtime per image (s)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    panel(axs[0, 0], lref, "LPIPS to reference (lower = closer)", "Convergence to 100-step output (LPIPS)")
    panel(axs[0, 1], iref, f"identity distance to ref ({args.face_set})", "Convergence (face embedding)")
    panel(axs[1, 0], clip, "CLIP cosine", "Prompt alignment", refline=clip.loc[ref_cfg, "mean"] if ref_cfg and ref_cfg in clip.index else None)
    panel(axs[1, 1], ldiv, "mean pairwise LPIPS", "Perceptual diversity across seeds")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Labels = number of denoising steps; bars = 95% bootstrap CI", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "summary_plots.png"), dpi=160)
    print("saved to", fig_dir)


if __name__ == "__main__":
    main()
