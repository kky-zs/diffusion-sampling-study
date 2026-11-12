"""Timing protocol for the scheduler comparison (run AFTER generate.py, as its own step).

Protocol (report this in the paper):
  - Same pipeline, fp16, 512x512, guidance 7.5, batch size 1, safety checker DISABLED.
  - Timed region: one full pipe(...) call = text encoding + denoising loop + VAE decode
    + PIL conversion. Model loading is excluded.
  - Warm-up: `--warmup` untimed calls per config before any timing starts.
  - Repeats: `--reps` rounds. Each round runs every config once, in randomized order
    (round-robin), so GPU throttling / drift on Colab hits all configs equally.
  - torch.cuda.synchronize() before and after each timed call.
  - GPU clock + temperature logged each round via nvidia-smi.
Outputs: <out_dir>/timing/timing_runs.csv (every run), timing_summary.csv, timing_info.json

Usage (Colab):  python src/timing.py --config configs/experiment.yaml
"""
import argparse, json, os, random, subprocess, time
import torch, yaml, diffusers
import diffusers as D
import pandas as pd
from diffusers import StableDiffusionPipeline


def gpu_state():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True).strip()
        sm, smax, temp, power = [x.strip() for x in out.split(",")]
        return dict(sm_clock_mhz=sm, sm_clock_max_mhz=smax, temp_c=temp, power_w=power)
    except Exception:
        return dict(sm_clock_mhz=None, sm_clock_max_mhz=None, temp_c=None, power_w=None)


def initial_latents(seed, h, w, device, dtype):
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, 4, h // 8, w // 8), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--prompt_id", default=None, help="default: first prompt in config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out = os.path.join(cfg["paths"]["out_dir"], "timing")
    os.makedirs(out, exist_ok=True)
    f = cfg["fixed"]
    dtype, device = getattr(torch, cfg["model"]["dtype"]), "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(cfg["model"]["id"], torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    base = pipe.scheduler.config

    pid = args.prompt_id or next(iter(cfg["prompts"]))
    prompt = cfg["prompts"][pid]
    seeds = cfg["seeds"][:5]
    configs = [(s, n) for s in cfg["schedulers"] for n in cfg["steps"]]

    def run_once(sname, steps, seed):
        spec = cfg["schedulers"][sname]
        pipe.scheduler = getattr(D, spec["class"]).from_config(base, **spec.get("kwargs", {}))
        lat = initial_latents(seed, f["height"], f["width"], device, dtype)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pipe(prompt=prompt, negative_prompt=f["negative_prompt"], num_inference_steps=steps,
             guidance_scale=f["guidance_scale"], height=f["height"], width=f["width"],
             latents=lat, **spec.get("call_kwargs", {}))
        torch.cuda.synchronize()
        return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**20

    print("warm-up ...")
    for sname, steps in configs:
        for _ in range(args.warmup):
            run_once(sname, steps, seeds[0])

    rng = random.Random(0)
    rows = []
    for rep in range(args.reps):
        order = configs[:]
        rng.shuffle(order)
        gs = gpu_state()
        for sname, steps in order:
            dt, peak = run_once(sname, steps, seeds[rep % len(seeds)])
            rows.append(dict(rep=rep, scheduler=sname, steps=steps, seconds=dt,
                             peak_vram_mb=peak, **gs))
        print(f"round {rep + 1}/{args.reps} done  (GPU: {gs})")

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out, "timing_runs.csv"), index=False)
    summ = (runs.groupby(["scheduler", "steps"]).seconds
            .agg(median="median", mean="mean", std="std", min="min", max="max", n="count")
            .round(3).reset_index())
    summ["peak_vram_mb"] = runs.groupby(["scheduler", "steps"]).peak_vram_mb.max().round(0).values
    summ.to_csv(os.path.join(out, "timing_summary.csv"), index=False)
    json.dump(dict(torch=torch.__version__, diffusers=diffusers.__version__,
                   gpu=torch.cuda.get_device_name(0), prompt_id=pid, reps=args.reps,
                   warmup=args.warmup, safety_checker=False, batch_size=1,
                   timed_region="full pipe() call incl. text encoding, denoising, VAE decode, PIL conversion"),
              open(os.path.join(out, "timing_info.json"), "w"), indent=2)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
