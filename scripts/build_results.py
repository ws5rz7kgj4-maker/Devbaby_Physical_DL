#!/usr/bin/env python3
"""Rebuild descriptive tables/figures from the bundled, immutable result JSON/CSV.

This script trains no model and starts no remote job. CSV generation uses only
the Python standard library. Matplotlib is optional unless --require-figures
is supplied. Run from any directory: python3 scripts/build_results.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def read_json(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def early_tables():
    """Early physical-v1 system; never merge these panels with the later WM."""
    with (RAW / "early/formal_success_curves.csv").open(newline="", encoding="utf-8") as f:
        curves = list(csv.DictReader(f))
    seed_rows = []
    for variant in sorted({r["variant"] for r in curves}):
        for age in sorted({int(r["interactions"]) for r in curves}):
            for scope in ("iid", "ood"):
                for seed in sorted({int(r["seed"]) for r in curves}):
                    rr = [r for r in curves if r["variant"] == variant and int(r["interactions"]) == age
                          and int(r["seed"]) == seed and (r["domain"] == "iid") == (scope == "iid")]
                    seed_rows.append(dict(
                        system="early_physical_v1", panel="development", variant=variant,
                        interactions=age, scope=scope, seed=seed, domain_task_cells=len(rr),
                        macro_success=mean(float(r["success_rate"]) for r in rr),
                        successes=sum(round(float(r["success_rate"]) * int(r["n"])) for r in rr),
                        scored_trials=sum(int(r["n"]) for r in rr),
                    ))
    aggregates = []
    for variant in sorted({r["variant"] for r in seed_rows}):
        for age in sorted({r["interactions"] for r in seed_rows}):
            for scope in ("iid", "ood"):
                rr = [r for r in seed_rows if (r["variant"], r["interactions"], r["scope"]) == (variant, age, scope)]
                aggregates.append(dict(
                    system="early_physical_v1", panel="development", variant=variant,
                    interactions=age, scope=scope, model_seeds=len(rr),
                    macro_success=mean(r["macro_success"] for r in rr),
                    success_pct=100 * mean(r["macro_success"] for r in rr),
                    successes=sum(r["successes"] for r in rr), scored_trials=sum(r["scored_trials"] for r in rr),
                ))
    with (RAW / "early/formal_final_success.csv").open(newline="", encoding="utf-8") as f:
        finals = list(csv.DictReader(f))
    final_comparison = []
    for condition in ("full", "initial", "no_action"):
        rr = [r for r in finals if r["condition"] == condition]
        final_comparison.append(dict(
            system="early_physical_v1", panel="sealed_final", condition=condition,
            scope="ood", model_seeds=len(rr), domain_task_cells=12,
            trials_per_seed_cell=24, macro_success=mean(float(r["ood_success"]) for r in rr),
            success_pct=100 * mean(float(r["ood_success"]) for r in rr),
            successes=sum(round(float(r["ood_success"]) * 12 * 24) for r in rr),
            scored_trials=len(rr) * 12 * 24,
        ))
    baselines = read_json("early/formal_summary_baselines.json")
    for condition in ("random", "oracle_reference"):
        rr = [v for domain, tasks in baselines[condition]["cells"].items() if domain != "iid"
              for v in tasks.values()]
        final_comparison.append(dict(
            system="early_physical_v1", panel="sealed_final_reference", condition=condition,
            scope="ood", model_seeds=None, domain_task_cells=len(rr),
            trials_per_seed_cell=None, macro_success=mean(r["success_rate"] for r in rr),
            success_pct=100 * mean(r["success_rate"] for r in rr),
            successes=sum(round(r["success_rate"] * r["n"]) for r in rr),
            scored_trials=sum(r["n"] for r in rr),
        ))
    return {"early_development_by_seed.csv": seed_rows,
            "early_development_macro.csv": aggregates,
            "early_sealed_final_comparison.csv": final_comparison}


def tables():
    """Return all derived tables without writing files or rounding values."""
    protocol = read_json("pilot_protocol.json")
    episodes = read_json("pilot_episodes_enriched.json")
    pilot = []
    for cell, hold, sigma in protocol["cells"]:
        rows = [r for r in episodes if r["cell"] == cell]
        pilot.append(dict(
            condition=cell, hold_steps=hold, noise_sd_before_clipping=sigma,
            paired_scenes=len(rows), steps_per_episode=protocol["steps"],
            contact_fraction=mean(r["contact_fraction"] for r in rows),
            contact_pct=100 * mean(r["contact_fraction"] for r in rows),
            action_effect_2cm_fraction=mean(r["action_effect_2cm_fraction"] for r in rows),
            action_effect_2cm_pct=100 * mean(r["action_effect_2cm_fraction"] for r in rows),
            action_saturation_fraction=mean(r["action_saturation_fraction"] for r in rows),
            action_change_rms=mean(r["action_change_rms"] for r in rows),
        ))
    reference = read_json("pilot_reference.json")
    reference_rows = []
    for task in sorted({r["task"] for r in reference}):
        rr = [r for r in reference if r["task"] == task]
        reference_rows.append(dict(task=task, controller="privileged_reference",
                                   successes=sum(r["success"] for r in rr), trials=len(rr),
                                   success_fraction=mean(r["success"] for r in rr)))

    baseline = []
    for row in read_json("baseline_summary.json"):
        baseline.append(dict(
            treatment=row["treatment"], update=row["update"],
            split=row["split"], horizon=row["horizon"],
            actual_object_mse=row["object_mse"]["actual"],
            zero_object_mse=row["object_mse"]["zero"],
            reversed_object_mse=row["object_mse"]["reversed"],
            persistence_object_mse=row["persistence_object_mse"],
            actual_full_image_mse=row["rgb_mse"],
        ))
    baseline.sort(key=lambda r: (r["split"], r["treatment"], r["update"], r["horizon"]))

    frozen = []
    for age, horizons in read_json("frozen_control_summary.json").items():
        for horizon, row in horizons.items():
            frozen.append(dict(
                training_transitions=int(age), horizon=int(horizon),
                actual_object_mse=row["action_object_mse"]["actual"],
                zero_object_mse=row["action_object_mse"]["zero"],
                reversed_object_mse=row["action_object_mse"]["reversed"],
                persistence_object_mse=row["new_persistence_object_mse"],
                distance_advantage_over_uniform_m=row["distance_advantage_over_uniform"],
                rank_correlation=row["rank_correlation"],
                ranking_seed_scene_rows=row["ranking_n"],
                interpretation="compromised_training_diagnostic_only",
            ))
    frozen.sort(key=lambda r: (r["training_transitions"], r["horizon"]))

    gpu, cpu = read_json("parameter_gpu_result.json"), read_json("parameter_cpu_result.json")
    gpu_rows, gpu_counts, cpu_rows = [], [], []
    for check in gpu["path_checks"]:
        rr = check["parameters"]
        for param in rr:
            gpu_rows.append(dict(mode=check["mode"], seed=check["seed"],
                                 initial_age=check["initial_age"], **param))
        gpu_counts.append(dict(
            mode=check["mode"], parameter_tensors=len(rr),
            gradients_present=sum(bool(p["grad_present"]) for p in rr),
            finite_gradients=sum(p["grad_finite"] is True for p in rr),
            nonzero_updates=sum(p["step_max_abs_change"] > 0 for p in rr),
            recurrent_tensors=sum(p["name"].split(".")[0] in ("dynamics", "filter") for p in rr),
            recurrent_gradients_present=sum(bool(p["grad_present"]) for p in rr
                                            if p["name"].split(".")[0] in ("dynamics", "filter")),
            recurrent_nonzero_updates=sum(p["step_max_abs_change"] > 0 for p in rr
                                         if p["name"].split(".")[0] in ("dynamics", "filter")),
        ))
    for check in cpu["path_checks"]:
        for param in check["parameters"]:
            cpu_rows.append(dict(seed=check["seed"], initial_age=check["initial_age"],
                                 save_reload_exact=check["save_reload_exact"], **param))

    checkpoint_params, checkpoint_groups = [], []
    for check in cpu["checkpoints"]:
        groups = defaultdict(list)
        for param in check["parameters"]:
            checkpoint_params.append(dict(seed=check["seed"], **param))
            groups[param["name"].split(".")[0]].append(param)
        for group, rr in groups.items():
            checkpoint_groups.append(dict(
                seed=check["seed"], group=group, parameter_tensors=len(rr),
                exactly_unchanged=sum(bool(p["exact_equal"]) for p in rr),
                changed=sum(not p["exact_equal"] for p in rr),
                with_optimizer_state=sum(bool(p["old_optimizer_has_state"]) for p in rr),
                max_abs_change=max(p["max_abs_change"] for p in rr),
            ))

    result = {
        "pilot_conditions.csv": pilot,
        "pilot_reference.csv": reference_rows,
        "baseline_all.csv": baseline,
        "baseline_200_validation.csv": [r for r in baseline if r["update"] == 200
                                        and r["split"] == "development_validation"],
        "frozen_control_all.csv": frozen,
        "frozen_control_final.csv": [r for r in frozen if r["training_transitions"] == 65536],
        "gpu_parameter_updates.csv": gpu_rows,
        "gpu_mode_summary.csv": gpu_counts,
        "cpu_parameter_updates.csv": cpu_rows,
        "checkpoint_parameter_changes.csv": checkpoint_params,
        "checkpoint_group_changes.csv": checkpoint_groups,
    }
    result.update(early_tables())
    return result


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def make_figures(data, required=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if required:
            raise SystemExit("Matplotlib is required for figures; install requirements-analysis.txt.")
        print("Matplotlib unavailable: CSV files rebuilt; figures skipped.")
        return

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "svg.hashsalt": "devbaby-20261111",
        "savefig.facecolor": "white",
    })
    target = ROOT / "figures"
    target.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(target / (name + ".png"), dpi=180)
        fig.savefig(target / (name + ".svg"), metadata={"Date": None})
        plt.close(fig)

    rows = data["pilot_conditions.csv"]
    labels = [r["condition"] for r in rows]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10.8, 7.6))
    ax.barh([i - .19 for i in y], [r["contact_pct"] for r in rows], height=.34,
            color="#166b8a", label="Contact time")
    ax.barh([i + .19 for i in y], [r["action_effect_2cm_pct"] for r in rows], height=.34,
            color="#d28d31", label="Action-effect time (> 2 cm vs matched idle)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 28)
    for i, row in enumerate(rows):
        ax.text(row["contact_pct"] + .18, i - .19, f'{row["contact_pct"]:.2f}', va="center", fontsize=8)
        ax.text(row["action_effect_2cm_pct"] + .18, i + .19,
                f'{row["action_effect_2cm_pct"]:.2f}', va="center", fontsize=8)
    ax.set_xlabel("Mean percentage of episode time (%)")
    ax.set_title("Exploration coverage: descriptive pilot", loc="left", pad=14)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(.02, .065, "8 paired scenes per condition; 128 steps per episode. No neural training.", fontsize=10)
    fig.text(.02, .035, "Action-effect time includes persistent displacement; it is not task success or information gain.", fontsize=10)
    fig.subplots_adjust(left=.16, right=.98, top=.91, bottom=.16)
    save(fig, "pilot_coverage")

    rr = [r for r in data["baseline_all.csv"] if r["horizon"] == 64
          and r["split"] == "development_validation"]
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    for treatment, color, label in [("original", "#166b8a", "Original loss"),
                                     ("change_balanced", "#d28d31", "Change-balanced loss")]:
        tr = sorted((r for r in rr if r["treatment"] == treatment), key=lambda r: r["update"])
        ax.plot([r["update"] for r in tr], [r["actual_object_mse"] for r in tr],
                marker="o", color=color, lw=2, label=label)
    persist = rr[0]["persistence_object_mse"]
    ax.axhline(persist, color="#595959", linestyle="--", lw=1.8,
               label=f"Persistence: {persist:.4f}")
    ax.set_ylim(0, .13)
    ax.set_xticks([0, 50, 200])
    ax.set_xlabel("Optimizer updates (FP32)")
    ax.set_ylabel("Object-region RGB MSE (lower is better)")
    ax.set_title("Development prediction error at horizon 64", loc="left", pad=14)
    ax.grid(alpha=.18)
    ax.legend(frameon=False, loc="center right")
    fig.text(.02, .08, "1 model seed; 4 training + 4 previously inspected development-validation episodes.", fontsize=10)
    fig.text(.02, .04, "Prediction error, not physical success. Both losses remain worse than persistence.", fontsize=10)
    fig.subplots_adjust(left=.12, right=.98, top=.90, bottom=.23)
    save(fig, "fp32_prediction_learning")
    # This figure belongs to the earlier physical-v1 system, not the later WM.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.6, 6.7), gridspec_kw={"width_ratios": [1.25, 1]})
    for seed in range(8):
        rr = [r for r in data["early_development_by_seed.csv"]
              if r["variant"] == "full" and r["scope"] == "ood" and r["seed"] == seed]
        rr.sort(key=lambda r: r["interactions"])
        ax.plot([r["interactions"] for r in rr], [r["macro_success"] * 100 for r in rr],
                color="#166b8a", alpha=.17, lw=.8)
    for variant, color, label in [("full", "#166b8a", "Full model (8-seed mean)"),
                                  ("no_action", "#d28d31", "No-action ablation (8-seed mean)")]:
        rr = [r for r in data["early_development_macro.csv"] if r["variant"] == variant and r["scope"] == "ood"]
        rr.sort(key=lambda r: r["interactions"])
        ax.plot([r["interactions"] for r in rr], [r["success_pct"] for r in rr],
                color=color, marker="o", lw=2.2, label=label)
        if variant == "full":
            for r in rr[1:]:
                ax.annotate(f'{r["success_pct"]:.1f}%', (r["interactions"], r["success_pct"]),
                            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    ax.set_title("A. OOD development success", loc="left", fontsize=13)
    ax.set_xlabel("Collected environment interactions")
    ax.set_ylabel("Macro task success (%)")
    ax.set_xticks([0, 8192, 16384, 24576, 32768], ["0", "8,192", "16,384", "24,576", "32,768"])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(-3, 100)
    ax.grid(alpha=.16)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    rr = data["early_sealed_final_comparison.csv"]
    labels = ["Full model", "Initial model", "No-action ablation", "Random actions", "Privileged reference"]
    colors = ["#166b8a", "#a3b9c3", "#d28d31", "#91979c", "#347864"]
    bx.barh(range(5), [r["success_pct"] for r in rr], color=colors, height=.57)
    bx.set_yticks(range(5), labels)
    bx.invert_yaxis()
    bx.set_xlim(0, 105)
    bx.set_xlabel("Macro task success (%)")
    bx.set_title("B. Separate sealed-final OOD panel", loc="left", fontsize=13)
    for i, r in enumerate(rr):
        bx.text(r["success_pct"] + 1, i, f'{r["success_pct"]:.2f}%', va="center", fontsize=10)
    bx.grid(axis="x", alpha=.16)
    bx.set_axisbelow(True)
    fig.suptitle("Early physical-control results", x=.08, ha="left", fontsize=17, fontweight="bold", y=.97)
    fig.text(.08, .91, "Body prediction + explicit coordinate/contact goals + programmed CEM; separate from the later world model.", fontsize=11)
    fig.text(.03, .12, "A: equal-weight average over 8 model seeds, 4 OOD domains and 3 tasks; 6 scenes per cell. Thin lines: individual full-model seeds.", fontsize=10)
    fig.text(.03, .08, "B: 24 scenes per domain/task; 2,304 scored trials per learned condition, 288 per reference. Shared scenes are not independent trials.", fontsize=10)
    fig.text(.03, .04, "Development endpoint 74.13% and sealed-final 72.92% use different evaluation panels; no confidence intervals shown.", fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.25, wspace=.50)
    save(fig, "early_physical_learning")
    print("Rebuilt 3 figures, each in PNG and SVG.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--require-figures", action="store_true")
    args = parser.parse_args()
    if args.no_figures and args.require_figures:
        parser.error("Choose --no-figures or --require-figures, not both.")
    data = tables()
    target = ROOT / "results" / "tables"
    target.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        (target / name).write_text(csv_text(rows), encoding="utf-8")
    print(f"Rebuilt {len(data)} CSV tables from bundled result JSON/CSV; no training performed.")
    if not args.no_figures:
        make_figures(data, required=args.require_figures)


if __name__ == "__main__":
    main()
