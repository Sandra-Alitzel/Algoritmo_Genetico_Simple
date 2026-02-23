import time
from dataclasses import asdict

import numpy as np
import streamlit as st

from ga_lab.ga.config import GAConfig
from ga_lab.ga.binary_ga import run_binary_ga
from ga_lab.benchmarks.problems import get_problems
from ga_lab.benchmarks.known_optima import KNOWN_OPTIMA
from ga_lab.viz.plots import (
    plot_convergence_curve,
    plot_band_curves,
    plot_box,
    plot_series,
)
from ga_lab.experiments.runner import *


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="GA Benchmark Lab", layout="wide")
st.title("🧬 GA Benchmark Lab")
st.caption(
    "GA binario (ruleta + 1-point crossover + bit-flip mutation) para funciones benchmark. "
    "Incluye multi-seed, time-to-threshold, robustez, y guardado en results/."
)

problems = get_problems()
problem_keys = list(problems.keys())


# -----------------------------
# Helpers
# -----------------------------
def gap_to_optimum(problem_key: str, best_f: float):
    opt = KNOWN_OPTIMA.get(problem_key, {})
    f_star = opt.get("f_star", None)
    if f_star is None:
        return None, opt
    return float(best_f - float(f_star)), opt


def time_to_threshold(best_curve: np.ndarray, thr: float):
    idx = np.where(best_curve <= thr)[0]
    return int(idx[0]) if idx.size > 0 else None


def robustness_metrics(finals: np.ndarray, ttt_list):
    q25 = float(np.percentile(finals, 25))
    q75 = float(np.percentile(finals, 75))
    iqr = q75 - q25
    std = float(np.std(finals))
    med = float(np.median(finals))
    mean = float(np.mean(finals))
    success_rate = None
    med_ttt = None
    if ttt_list is not None:
        success_rate = float(np.mean([t is not None for t in ttt_list]))
        t_ok = [t for t in ttt_list if t is not None]
        med_ttt = float(np.median(t_ok)) if len(t_ok) > 0 else None

    return {
        "median_best_f": med,
        "mean_best_f": mean,
        "std_best_f": std,
        "q25_best_f": q25,
        "q75_best_f": q75,
        "iqr_best_f": float(iqr),
        "success_rate": success_rate,
        "median_time_to_threshold_gen": med_ttt,
    }


def scenario_defaults(scenario: str):
    # Presets defendibles (A–E)
    if scenario == "A_baseline":
        return dict(pop_size=80, generations=200, pc=0.90, elitism=1, pm_mode="1/L", pm_value=0.01)
    if scenario == "B_high_exploitation":
        return dict(pop_size=80, generations=200, pc=0.95, elitism=3, pm_mode="1/L", pm_value=0.01)
    if scenario == "C_no_elitism":
        return dict(pop_size=80, generations=200, pc=0.90, elitism=0, pm_mode="1/L", pm_value=0.01)
    if scenario == "D_high_mutation":
        return dict(pop_size=80, generations=200, pc=0.90, elitism=1, pm_mode="manual", pm_value=0.05)
    if scenario == "E_more_budget":
        return dict(pop_size=200, generations=400, pc=0.90, elitism=1, pm_mode="1/L", pm_value=0.01)

    # fallback (shouldn't happen)
    return dict(pop_size=80, generations=200, pc=0.90, elitism=1, pm_mode="1/L", pm_value=0.01)


def config_from_scenario(
    scenario: str,
    seed: int,
    decimals: int,
):
    base = scenario_defaults(scenario)
    return GAConfig(
        pop_size=int(base["pop_size"]),
        generations=int(base["generations"]),
        pc=float(base["pc"]),
        elitism=int(base["elitism"]),
        seed=int(seed),
        pm_mode=str(base["pm_mode"]),
        pm_value=float(base["pm_value"]),
        decimals=int(decimals),
    )


def run_once(problem_key: str, cfg: GAConfig):
    pr = problems[problem_key]

    def obj(x: np.ndarray) -> float:
        return float(pr.func(x))

    t0 = time.perf_counter()
    res = run_binary_ga(pr.bounds, obj, cfg)
    dt = time.perf_counter() - t0
    return pr, res, dt


def default_threshold_for(problem_key: str):
    # Thresholds "bonitos" para time-to-threshold (puedes ajustarlos)
    if problem_key == "himmelblau":
        return 1e-2
    if problem_key == "bukin_n6":
        return 1.0
    if problem_key == "rastrigin_2":
        return 1.0
    if problem_key == "rastrigin_5":
        return 10.0
    if problem_key == "eggholder":
        return -900.0
    return 1.0


# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.selectbox("Choose", ["Single Run", "Multi-seed (Scenario)", "Compare Scenarios", "Compare Functions"])

    st.header("Problem")
    pkey = st.selectbox("Benchmark problem", problem_keys, format_func=lambda k: problems[k].name)

    st.header("Precision")
    decimals = st.selectbox("Decimals (>= 3 for 0.001)", [3, 4, 5], index=0)

    st.header("Scenario presets")
    scenario = st.selectbox(
        "Scenario",
        ["A_baseline", "B_high_exploitation", "C_no_elitism", "D_high_mutation", "E_more_budget"],
        index=0,
    )

    st.header("Experiment (seeds)")
    k_seeds = st.slider("Num seeds (k)", 2, 30, 10, 1)
    seed_base = st.number_input("Base seed", 0, 1_000_000, 7, 1)
    seed_stride = st.number_input("Seed stride", 1, 100_000, 1, 1)

    st.header("Time-to-threshold")
    use_thr = st.checkbox("Enable threshold", value=True)
    thr_default = float(default_threshold_for(pkey))
    thr = st.number_input("Threshold value", value=thr_default) if use_thr else None

    st.header("Results")
    save_results = st.checkbox("Save results to results/", value=True)

    st.divider()
    run_btn = st.button("▶ Run", type="primary")

if not run_btn:
    st.info("Configura y presiona **Run**.")
    st.stop()


# -----------------------------
# SINGLE RUN
# -----------------------------
if mode == "Single Run":
    cfg = config_from_scenario(scenario, int(seed_base), int(decimals))
    pr, res, dt = run_once(pkey, cfg)

    best_f = float(res["best_cost"])
    gap, opt = gap_to_optimum(pkey, best_f)

    # threshold info
    ttt = time_to_threshold(res["history_best_cost"], float(thr)) if thr is not None else None

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📈 Convergence (best-so-far)")
        st.pyplot(plot_convergence_curve(res["history_best_cost"], f"{pr.name} — {scenario}"), clear_figure=True)

        st.subheader("🧪 Population diagnostics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_series(res["history_diversity"], "Diversity vs Gen", "diversity"), clear_figure=True)
        with c2:
            st.pyplot(plot_series(res["history_mean_cost"], "Mean cost vs Gen", "mean cost"), clear_figure=True)
        with c3:
            st.pyplot(plot_series(res["history_std_cost"], "Std cost vs Gen", "std cost"), clear_figure=True)

    with col2:
        st.subheader("✅ Summary")
        st.json(
            {
                "problem": pr.name,
                "scenario": scenario,
                "dims": pr.dims,
                "bounds": pr.bounds,
                "best_x": res["best_x"].tolist(),
                "best_f": best_f,
                "known_optimum": opt,
                "gap_to_f_star": gap,
                "threshold": thr,
                "time_to_threshold_gen": ttt,
                "genome_len": int(res["genome_len"]),
                "bits_per_dim": res["bits_per_dim"],
                "pm_used": float(res["pm_used"]),
                "time_sec": dt,
                "evaluations": int(cfg.pop_size) * int(cfg.generations),
                "evals_per_sec": (int(cfg.pop_size) * int(cfg.generations)) / dt if dt > 0 else None,
                "config": asdict(cfg),
            }
        )

    if save_results:
        # guardamos como experimento con k=1
        exp_dir = run_experiment(
            problem_key=pkey,
            scenario_name=f"{scenario}_single",
            cfg_base=cfg,
            k_seeds=1,
            seed_stride=int(seed_stride),
            threshold=thr,
            out_root="results",
        )
        st.success(f"Saved to: {exp_dir}")


# -----------------------------
# MULTI-SEED (ONE SCENARIO)
# -----------------------------
elif mode == "Multi-seed (Scenario)":
    cfg_base = config_from_scenario(scenario, int(seed_base), int(decimals))

    if save_results:
        exp_dir = run_experiment(
            problem_key=pkey,
            scenario_name=scenario,
            cfg_base=cfg_base,
            k_seeds=int(k_seeds),
            seed_stride=int(seed_stride),
            threshold=thr,
            out_root="results",
        )
        st.success(f"Saved experiment to: {exp_dir}")

    # Also run in-memory to display rich plots without re-reading files
    curves = []
    finals = []
    gaps = []
    ttt_list = [] if thr is not None else None

    div_curves = []
    mean_curves = []
    std_curves = []

    rows = []

    base_seed = int(cfg_base.seed)
    for i in range(int(k_seeds)):
        s = base_seed + i * int(seed_stride)
        cfg_i = GAConfig(**{**asdict(cfg_base), "seed": s})
        pr, res, dt = run_once(pkey, cfg_i)

        best_f = float(res["best_cost"])
        gap, _ = gap_to_optimum(pkey, best_f)

        curves.append(res["history_best_cost"])
        finals.append(best_f)
        gaps.append(gap)

        if thr is not None:
            ttt_list.append(time_to_threshold(res["history_best_cost"], float(thr)))

        div_curves.append(res["history_diversity"])
        mean_curves.append(res["history_mean_cost"])
        std_curves.append(res["history_std_cost"])

        rows.append(
            {
                "seed": s,
                "best_f": best_f,
                "gap": gap,
                "time_sec": dt,
                "time_to_threshold_gen": (ttt_list[-1] if thr is not None else None),
            }
        )

    curves = np.stack(curves, axis=0)
    finals = np.array(finals, dtype=float)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📉 Convergence bands (best-so-far across seeds)")
        st.pyplot(plot_band_curves(curves, f"{problems[pkey].name} — {scenario} (k={k_seeds})"), clear_figure=True)

        st.subheader("🧪 Population diagnostics (median across seeds)")
        div_median = np.median(np.stack(div_curves, axis=0), axis=0)
        mean_median = np.median(np.stack(mean_curves, axis=0), axis=0)
        std_median = np.median(np.stack(std_curves, axis=0), axis=0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_series(div_median, "Diversity (median)", "diversity"), clear_figure=True)
        with c2:
            st.pyplot(plot_series(mean_median, "Mean cost (median)", "mean cost"), clear_figure=True)
        with c3:
            st.pyplot(plot_series(std_median, "Std cost (median)", "std cost"), clear_figure=True)

    with col2:
        st.subheader("📦 Final best_f distribution")
        st.pyplot(plot_box(finals, f"Final best_f — {scenario}", "best_f"), clear_figure=True)

        st.subheader("🛡️ Robustness metrics")
        st.json(
            {
                "problem": problems[pkey].name,
                "scenario": scenario,
                "threshold": thr,
                **robustness_metrics(finals, ttt_list),
            }
        )

    st.subheader("📋 Runs table")
    st.json(rows)


# -----------------------------
# COMPARE SCENARIOS (SAME PROBLEM)
# -----------------------------
elif mode == "Compare Scenarios":
    scenarios = ["A_baseline", "B_high_exploitation", "C_no_elitism", "D_high_mutation", "E_more_budget"]

    st.subheader("🧪 Compare scenarios (same problem, same k seeds)")
    st.caption(
        "Se corre cada escenario con k semillas. Se compara distribución final, robustez, y (si activas threshold) éxito y tiempo."
    )

    all_results = {}
    stats = []

    for sc in scenarios:
        cfg_base = config_from_scenario(sc, int(seed_base), int(decimals))

        if save_results:
            exp_dir = run_experiment(
                problem_key=pkey,
                scenario_name=sc,
                cfg_base=cfg_base,
                k_seeds=int(k_seeds),
                seed_stride=int(seed_stride),
                threshold=thr,
                out_root="results",
            )
            # no spameamos la UI con 5 paths; guardamos uno por stats
        finals = []
        ttt_list = [] if thr is not None else None

        for i in range(int(k_seeds)):
            s = int(cfg_base.seed) + i * int(seed_stride)
            cfg_i = GAConfig(**{**asdict(cfg_base), "seed": s})
            _, res, _dt = run_once(pkey, cfg_i)
            finals.append(float(res["best_cost"]))
            if thr is not None:
                ttt_list.append(time_to_threshold(res["history_best_cost"], float(thr)))

        finals = np.array(finals, dtype=float)
        all_results[sc] = finals

        rob = robustness_metrics(finals, ttt_list)
        stats.append({"scenario": sc, **rob})

    # ranking por mediana (más chico es mejor en minimización)
    stats_sorted = sorted(stats, key=lambda r: r["median_best_f"])

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("🏁 Scenario ranking (by median best_f)")
        st.json(stats_sorted)

    with c2:
        st.subheader("📦 Boxplots (final best_f) — one per scenario")
        for sc in scenarios:
            st.pyplot(plot_box(all_results[sc], sc, "best_f"), clear_figure=True)

    if save_results:
        st.success("Saved all scenarios to results/ (one folder per scenario).")


# -----------------------------
# COMPARE FUNCTIONS (SAME SCENARIO)
# -----------------------------
else:
    st.subheader("🏁 Compare functions (same scenario, same k seeds)")
    st.caption(
        "Se fija un escenario y se compara desempeño final por función. "
        "Útil para responder '¿cuál función es más difícil?' y el efecto de dimensionalidad."
    )

    all_finals = {}
    stats = []

    for key in problem_keys:
        cfg_base = config_from_scenario(scenario, int(seed_base), int(decimals))

        # (opcional) guardar cada función como experimento
        if save_results:
            run_experiment(
                problem_key=key,
                scenario_name=scenario,
                cfg_base=cfg_base,
                k_seeds=int(k_seeds),
                seed_stride=int(seed_stride),
                threshold=thr if (thr is not None) else None,
                out_root="results",
            )

        finals = []
        gaps = []

        ttt_list = [] if thr is not None else None

        for i in range(int(k_seeds)):
            s = int(cfg_base.seed) + i * int(seed_stride)
            cfg_i = GAConfig(**{**asdict(cfg_base), "seed": s})
            pr, res, _dt = run_once(key, cfg_i)

            bf = float(res["best_cost"])
            finals.append(bf)

            gap, opt = gap_to_optimum(key, bf)
            gaps.append(gap)

            if thr is not None:
                ttt_list.append(time_to_threshold(res["history_best_cost"], float(thr)))

        finals = np.array(finals, dtype=float)
        all_finals[key] = finals

        # si existe f_star, es mejor rankear por gap; si no, por best_f
        opt = KNOWN_OPTIMA.get(key, {})
        f_star = opt.get("f_star", None)
        med_final = float(np.median(finals))
        med_gap = float(np.median([g for g in gaps if g is not None])) if f_star is not None else None

        rob = robustness_metrics(finals, ttt_list)

        stats.append(
            {
                "problem": problems[key].name,
                "key": key,
                "f_star": f_star,
                "median_best_f": med_final,
                "median_gap_to_f_star": med_gap,
                **rob,
            }
        )

    def rank_key(row):
        # Prefer gap if available, else best_f
        if row["median_gap_to_f_star"] is not None:
            return (0, row["median_gap_to_f_star"])
        return (1, row["median_best_f"])

    stats_sorted = sorted(stats, key=rank_key)

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("📌 Ranking")
        st.json(stats_sorted)

    with c2:
        st.subheader("📦 Boxplots (final best_f) — per function")
        for row in stats_sorted:
            k = row["key"]
            st.pyplot(plot_box(all_finals[k], problems[k].name, "best_f"), clear_figure=True)

    if save_results:
        st.success("Saved all functions to results/ (one folder per function).")