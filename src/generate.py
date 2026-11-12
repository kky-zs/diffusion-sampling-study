"""Resumable image generation for the scheduler comparison.

Usage (Colab):  python src/generate.py --config configs/experiment.yaml
Re-running skips every image that already exists, so a disconnect costs nothing.
Outputs: <out_dir>/images/*.png, <out_dir>/manifest.jsonl, <out_dir>/run_info.json
NOTE: per-image times logged here are indicative only. Use a separate timing script
for the reported runtimes (warm-up, synchronize, repeats, no safety checker).
"""
import argparse, json, os, platform, time
import torch, yaml, diffusers, transformers
import diffusers as D
from diffusers import StableDiffusionPipeline


def build_jobs(cfg):
    jobs = []
    for pid in cfg["prompts"]:
        for seed in cfg["seeds"]:
            for sname in cfg["schedulers"]:
                for steps in cfg["steps"]:
                    jobs.append(dict(role="main", prompt_id=pid, seed=seed,
                                     scheduler=sname, steps=steps))
    ref = cfg.get("reference", {})
    if ref.get("enabled"):
        for pid in cfg["prompts"]:
            for seed in cfg["seeds"]:
                jobs.append(dict(role="reference", prompt_id=pid, seed=seed,
                                 scheduler=ref["scheduler"], steps=ref["steps"]))
    return jobs


def fname(j):
    return f'{j["role"]}_{j["scheduler"]}_s{j["steps"]}_{j["prompt_id"]}_seed{j["seed"]}.png'


def make_scheduler(base_cfg, spec):
    return getattr(D, spec["class"]).from_config(base_cfg, **spec.get("kwargs", {}))


def initial_latents(seed, height, width, device, dtype):
    # Same CPU-generated noise for every scheduler / step count -> matched initial noise.
    g = torch.Generator("cpu").manual_seed(seed)
    z = torch.randn((1, 4, height // 8, width // 8), generator=g, dtype=torch.float32) # Generate random noise
    return z.to(device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out = cfg["paths"]["out_dir"]
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.jsonl")

    dtype = getattr(torch, cfg["model"]["dtype"])
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(cfg["model"]["id"], torch_dtype=dtype)
    if not cfg["model"]["safety_checker"]:
        pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    base_sched_cfg = pipe.scheduler.config

    run_info = dict(
        model=cfg["model"]["id"], dtype=cfg["model"]["dtype"],
        torch=torch.__version__, diffusers=diffusers.__version__,
        transformers=transformers.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        base_scheduler_config={k: str(v) for k, v in base_sched_cfg.items()},
        config=cfg,
    )
    json.dump(run_info, open(os.path.join(out, "run_info.json"), "w"), indent=2)

    jobs = build_jobs(cfg)
    f = cfg["fixed"]
    done = skipped = 0
    for j in jobs:
        path = os.path.join(img_dir, fname(j))
        if os.path.exists(path):
            skipped += 1
            continue
        spec = cfg["schedulers"][j["scheduler"]]
        pipe.scheduler = make_scheduler(base_sched_cfg, spec)
        latents = initial_latents(j["seed"], f["height"], f["width"], device, dtype)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        res = pipe(
            prompt=cfg["prompts"][j["prompt_id"]], negative_prompt=f["negative_prompt"],
            num_inference_steps=j["steps"], guidance_scale=f["guidance_scale"],
            height=f["height"], width=f["width"], latents=latents,
            **spec.get("call_kwargs", {}),
        )
        torch.cuda.synchronize(); dt = time.perf_counter() - t0

        flags = getattr(res, "nsfw_content_detected", None)
        nsfw = bool(flags[0]) if flags else False
        res.images[0].save(path)   # if flagged, the pipeline already blacked the image out
        rec = dict(file=os.path.basename(path), **j, nsfw_flagged=nsfw,
                   indicative_seconds=round(dt, 3))
        with open(manifest_path, "a") as mf:
            mf.write(json.dumps(rec) + "\n")
        done += 1
        if done % 20 == 0:
            print(f"generated {done} (skipped {skipped}) / {len(jobs)}")

    print(f"finished: generated {done}, skipped existing {skipped}, total jobs {len(jobs)}")


if __name__ == "__main__":
    main()
