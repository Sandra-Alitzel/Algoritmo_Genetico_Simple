import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from ga_lab.ga.config import GAConfig
from ga_lab.ga.binary_ga import run_binary_ga
from ga_lab.benchmarks.problems import get_problems
from ga_lab.benchmarks.known_optima import KNOWN_OPTIMA
from ga_lab.io.save import ensure_dir, save_json
from ga_lab.viz.plots import plot_convergence_curve

def main():
    problems = get_problems()
    cfg = GAConfig(pop_size=80, generations=300, pc=0.9, elitism=1, seed=7, pm_mode="1/L", decimals=3)

    base = "results"
    ensure_dir(base)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    outdir = os.path.join(base, run_id)
    ensure_dir(outdir)

    summary = {}

    for key, pr in problems.items():
        def obj(x: np.ndarray) -> float:
            return float(pr.func(x))

        res = run_binary_ga(pr.bounds, obj, cfg)
        best_x = res["best_x"].tolist()
        best_f = float(res["best_cost"])

        opt = KNOWN_OPTIMA.get(key, {})
        f_star = opt.get("f_star", None)
        gap = (best_f - float(f_star)) if f_star is not None else None

        payload = {
            "problem_key": key,
            "problem_name": pr.name,
            "dims": pr.dims,
            "bounds": pr.bounds,
            "config": res["config"].__dict__,
            "genome_len": int(res["genome_len"]),
            "bits_per_dim": res["bits_per_dim"],
            "pm_used": float(res["pm_used"]),
            "best_x": best_x,
            "best_f": best_f,
            "known_optimum": opt,
            "gap_to_f_star": gap,
            "history_best_cost": res["history_best_cost"].tolist(),
        }

        save_json(os.path.join(outdir, f"{key}.json"), payload)

        fig = plot_convergence_curve(res["history_best_cost"], f"Convergence — {pr.name}")
        fig.savefig(os.path.join(outdir, f"{key}_convergence.png"), dpi=160)
        plt.close(fig)

        summary[key] = {"best_f": best_f, "gap": gap, "best_x": best_x}

    save_json(os.path.join(outdir, "summary.json"), summary)
    print("Saved to:", outdir)

if __name__ == "__main__":
    main()