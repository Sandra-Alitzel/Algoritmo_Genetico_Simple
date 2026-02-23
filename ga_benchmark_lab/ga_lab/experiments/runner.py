import os, json, time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from ga_lab.ga.binary_ga import run_binary_ga
from ga_lab.ga.config import GAConfig
from ga_lab.benchmarks.problems import get_problems
from ga_lab.benchmarks.known_optima import KNOWN_OPTIMA
from ga_lab.viz.plots import plot_band_curves, plot_box


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def gap_to_optimum(problem_key: str, best_f: float):
    opt = KNOWN_OPTIMA.get(problem_key, {})
    f_star = opt.get("f_star", None)
    if f_star is None:
        return None, opt
    return float(best_f - float(f_star)), opt

def time_to_threshold(best_curve: np.ndarray, thr: float) -> Optional[int]:
    idx = np.where(best_curve <= thr)[0]
    return int(idx[0]) if idx.size > 0 else None

def robustness_metrics(finals: np.ndarray, ttt_list: List[Optional[int]]):
    q25 = float(np.percentile(finals, 25))
    q75 = float(np.percentile(finals, 75))
    iqr = q75 - q25
    med = float(np.median(finals))
    mean = float(np.mean(finals))
    std = float(np.std(finals))
    success_rate = float(np.mean([t is not None for t in ttt_list])) if len(ttt_list) > 0 else None
    ttt_vals = [t for t in ttt_list if t is not None]
    med_ttt = float(np.median(ttt_vals)) if len(ttt_vals) > 0 else None
    return {
        "median": med,
        "mean": mean,
        "std": std,
        "q25": q25,
        "q75": q75,
        "iqr": float(iqr),
        "success_rate": success_rate,
        "median_time_to_threshold": med_ttt,
    }

def run_experiment(
    problem_key: str,
    scenario_name: str,
    cfg_base: GAConfig,
    k_seeds: int,
    seed_stride: int,
    threshold: Optional[float],
    out_root: str = "results",
) -> str:
    problems = get_problems()
    pr = problems[problem_key]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(out_root, f"{stamp}_{scenario_name}_{problem_key}")
    runs_dir = os.path.join(exp_dir, "runs")
    plots_dir = os.path.join(exp_dir, "plots")
    ensure_dir(runs_dir)
    ensure_dir(plots_dir)

    curves = []
    finals = []
    gaps = []
    times = []
    ttt_list: List[Optional[int]] = []
    runs_summary = []

    for i in range(k_seeds):
        seed_i = int(cfg_base.seed) + i * int(seed_stride)
        cfg = GAConfig(**{**asdict(cfg_base), "seed": seed_i})

        def obj(x: np.ndarray) -> float:
            return float(pr.func(x))

        t0 = time.perf_counter()
        res = run_binary_ga(pr.bounds, obj, cfg)
        dt = time.perf_counter() - t0

        best_curve = res["history_best_cost"]
        best_f = float(res["best_cost"])
        gap, opt = gap_to_optimum(problem_key, best_f)
        ttt = time_to_threshold(best_curve, threshold) if threshold is not None else None

        curves.append(best_curve)
        finals.append(best_f)
        gaps.append(gap)
        times.append(dt)
        ttt_list.append(ttt)

        run_payload = {
            "problem_key": problem_key,
            "problem_name": pr.name,
            "bounds": pr.bounds,
            "dims": pr.dims,
            "scenario": scenario_name,
            "config": asdict(cfg),
            "genome_len": int(res["genome_len"]),
            "bits_per_dim": res["bits_per_dim"],
            "pm_used": float(res["pm_used"]),
            "best_x": res["best_x"].tolist(),
            "best_f": best_f,
            "known_optimum": opt,
            "gap_to_f_star": gap,
            "time_sec": dt,
            "evaluations": int(cfg.pop_size) * int(cfg.generations),
            "evals_per_sec": (int(cfg.pop_size) * int(cfg.generations)) / dt if dt > 0 else None,
            "threshold": threshold,
            "time_to_threshold_gen": ttt,
            # histories for later analysis
            "history_best_cost": best_curve.tolist(),
            "history_mean_cost": res["history_mean_cost"].tolist(),
            "history_std_cost": res["history_std_cost"].tolist(),
            "history_diversity": res["history_diversity"].tolist(),
        }
        save_json(os.path.join(runs_dir, f"seed_{seed_i}.json"), run_payload)

        runs_summary.append({
            "seed": seed_i,
            "best_f": best_f,
            "gap": gap,
            "time_sec": dt,
            "time_to_threshold_gen": ttt,
        })

    curves = np.stack(curves, axis=0)
    finals = np.array(finals, dtype=float)
    times = np.array(times, dtype=float)

    rob = robustness_metrics(finals, ttt_list)

    summary = {
        "scenario": scenario_name,
        "problem_key": problem_key,
        "problem_name": pr.name,
        "k_seeds": k_seeds,
        "seed_base": int(cfg_base.seed),
        "seed_stride": seed_stride,
        "threshold": threshold,
        "robustness": rob,
        "median_best_f": float(np.median(finals)),
        "median_time_sec": float(np.median(times)),
        "runs": runs_summary,
        "config_base": asdict(cfg_base),
    }
    save_json(os.path.join(exp_dir, "summary.json"), summary)

    fig1 = plot_band_curves(curves, f"{pr.name} — {scenario_name} (k={k_seeds})")
    fig1.savefig(os.path.join(plots_dir, "band_convergence.png"), dpi=160)
    plt.close(fig1)

    fig2 = plot_box(finals, f"Final best_f — {pr.name} ({scenario_name})", "best_f")
    fig2.savefig(os.path.join(plots_dir, "box_final.png"), dpi=160)
    plt.close(fig2)

    return exp_dir