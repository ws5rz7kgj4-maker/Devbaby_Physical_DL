#!/usr/bin/env python3
"""Recompute archived early-study means and exact paired-seed tests.

Usage from the repository root:
    python3 scripts/validate_early_stats.py

Only the Python standard library is required. This script reads saved results;
it does not train models, launch jobs, or change any evidence file. Bootstrap
confidence intervals are ARCHIVED ONLY and are NOT recomputed here. In particular,
the complete scene-level records needed for the supplemental crossed seed/scene
bootstrap are not included in this compact release. Paired sign-flip tests remain
conditional on the saved scene panel and require the assumptions in the protocol.
"""

import csv
import itertools
import json
import math
import statistics
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "results" / "raw" / "early"
SEEDS = set(range(8))
TASKS = {"reach", "touch", "grasp_lift"}
ADAPT_DOMAINS = {"new_dynamics", "delay_3", "delay_6"}
ARMS = {"frozen", "full_update", "low_lr", "frozen_encoder", "replay_mix", "history"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        math.isfinite(float(actual))
        and math.isfinite(float(expected))
        and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10),
        f"{label}: computed {actual!r}, archived {expected!r}",
    )


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sign_flip_p(differences):
    """Exact two-sided enumeration; eight seeds produce 256 sign assignments."""
    require(len(differences) == 8, "Expected eight independent training seeds")
    observed = abs(statistics.mean(differences))
    extreme = sum(
        abs(statistics.mean(d * s for d, s in zip(differences, signs)))
        >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(differences))
    )
    return extreme / (2 ** len(differences))


def holm_adjust(p_values):
    adjusted = [None] * len(p_values)
    running_max = 0.0
    for rank, (p_value, index) in enumerate(sorted((p, i) for i, p in enumerate(p_values))):
        running_max = max(running_max, min(1.0, p_value * (len(p_values) - rank)))
        adjusted[index] = running_max
    return adjusted


def seed_cell_means(rows, domains, tasks, expected_scenes):
    """Validate the complete paired grid; equal task/domain weights per seed."""
    actual = {}
    for row in rows:
        key = (int(row["seed"]), row["domain"], row["task"])
        require(key not in actual, f"Duplicate seed/domain/task cell: {key}")
        require(int(row["n_scenes"]) == expected_scenes, f"Wrong scene count: {key}")
        actual[key] = float(row["success_rate"])
    expected = set(itertools.product(SEEDS, domains, tasks))
    require(set(actual) == expected, "Missing or unexpected seed/domain/task cells")
    return {
        seed: statistics.mean(actual[(seed, domain, task)] for domain in domains for task in tasks)
        for seed in sorted(SEEDS)
    }


def verify_supplement(curves, contrasts, derived):
    require(len(contrasts) == 6, "Supplement must retain all six primary comparisons")
    require({r["arm"] for r in derived["supplement_final_adaptation"]} == ARMS,
            "Derived adaptation summary has an incomplete arm set")
    endpoint = {}
    for arm in sorted(ARMS):
        rows = [r for r in curves if r["phase"] == "adaptation" and r["split"] == "final"
                and r["source_age"] == "32768" and r["budget"] == "2048" and r["arm"] == arm]
        endpoint[arm] = seed_cell_means(rows, ADAPT_DOMAINS, TASKS, 32)
        saved = next(r for r in derived["supplement_final_adaptation"] if r["arm"] == arm)
        close(statistics.mean(endpoint[arm].values()), saved["success_rate"], arm + " final success")

    p_values = []
    for row in contrasts:
        first, second = row["contrast"].split(" - ")
        require(first in endpoint and second in endpoint, "Unknown contrast arm")
        differences = [100 * (endpoint[first][seed] - endpoint[second][seed]) for seed in sorted(SEEDS)]
        saved_differences = json.loads(row["per_seed_differences_pp"])
        require(len(saved_differences) == 8, "Archived contrast must have eight seed differences")
        for seed, (actual, saved) in enumerate(zip(differences, saved_differences)):
            close(actual, saved, row["contrast"] + f" seed {seed}")
        close(statistics.mean(differences), row["mean_difference_pp"], row["contrast"] + " mean")
        p_value = sign_flip_p(differences)
        close(p_value, row["exact_paired_sign_flip_p"], row["contrast"] + " exact p")
        require(int(row["n_model_seeds"]) == 8 and int(row["n_domain_task_cells"]) == 9,
                "Wrong primary-comparison statistical units")
        p_values.append(p_value)
    for row, adjusted in zip(contrasts, holm_adjust(p_values)):
        close(adjusted, row["holm_p"], row["contrast"] + " Holm p")

    for saved in derived["supplement_delay_sweep"]:
        domain = "delay_" + str(saved["delay_steps"])
        rows = [r for r in curves if r["phase"] == "delay_sweep" and r["split"] == "final"
                and r["source_age"] == "32768" and r["budget"] == "0"
                and r["domain"] == domain and r["model"] == saved["model"]]
        means = seed_cell_means(rows, {domain}, TASKS, 32)
        close(statistics.mean(means.values()), saved["success_rate"], domain + " " + saved["model"])
        close(saved["delay_steps"] / 8, saved["delay_seconds"], domain + " seconds")
    require(len(derived["supplement_delay_sweep"]) == 12, "Expected two models at six delays")

    for saved in derived["supplement_push"]:
        rows = [r for r in curves if r["phase"] == "push" and r["split"] == "final"
                and r["source_age"] == "32768" and r["budget"] == "0"
                and r["domain"] == saved["domain"] and r["model"] == saved["model"]]
        means = seed_cell_means(rows, {saved["domain"]}, {"push"}, 32)
        close(statistics.mean(means.values()), saved["success_rate"], "push " + saved["model"] + " " + saved["domain"])
    require(len(derived["supplement_push"]) == 8, "Expected two models at four push domains")

    for saved in derived["supplement_retention"]:
        rows = [r for r in curves if r["phase"] == "retention" and r["split"] == "final"
                and r["source_age"] == "32768" and r["budget"] == "2048"
                and r["training_domain"] == saved["adaptation_domain"] and r["arm"] == saved["arm"]]
        means = seed_cell_means(rows, {"iid"}, TASKS, 12)
        close(statistics.mean(means.values()), saved["iid_success_rate"], "retention " + saved["arm"] + " " + saved["adaptation_domain"])
    require(len(derived["supplement_retention"]) == 18, "Expected six arms at three retention conditions")

    counts = derived["supplement_record_counts"]
    require(len(curves) == counts["evaluation_cells"] == 4918, "Supplemental cell count mismatch")
    require(sum(int(r["n_scenes"]) for r in curves) == counts["logical_scored_trial_rows"] == 76736,
            "Supplemental logical trial-row count mismatch")
    require(counts["unique_physical_rollouts_recounted"] is False,
            "Logical trial rows must not be relabeled as deduplicated physical rollouts")


def normalized_gain_auc(budgets, values):
    gains = [value - values[0] for value in values]
    area = sum((budgets[i] - budgets[i - 1]) * (gains[i] + gains[i - 1]) / 2
               for i in range(1, len(budgets)))
    return area / (budgets[-1] - budgets[0])


def verify_formal(derived):
    development = read_csv("formal_success_curves.csv")
    for saved in derived["formal_development"]:
        rows = [r for r in development if r["variant"] == "full" and r["domain"] != "iid"
                and int(r["interactions"]) == saved["interactions"]]
        require(len(rows) == 8 * 4 * 3, "Missing development model/domain/task cells")
        successes = sum(round(float(r["success_rate"]) * int(r["n"])) for r in rows)
        trials = sum(int(r["n"]) for r in rows)
        require(successes == saved["successes"] and trials == saved["scored_seed_scene_trials"] == 576,
                "Formal development count mismatch")
        close(successes / trials, saved["success_rate"], "formal development success")

    final = read_csv("formal_final_success.csv")
    scores = {}
    for row in final:
        key = (row["condition"], int(row["seed"]))
        require(key not in scores, "Duplicate formal final row")
        scores[key] = float(row["ood_success"])
    require(set(scores) == set(itertools.product({"full", "initial", "no_action"}, SEEDS)),
            "Incomplete formal endpoint matrix")
    summary = read_json("formal_summary.json")
    for condition, key in [("full", "final_full_mean"), ("initial", "final_initial_mean"),
                           ("no_action", "final_no_action_mean")]:
        close(statistics.mean(scores[(condition, s)] for s in sorted(SEEDS)), summary[key], key)
    for seed in sorted(SEEDS):
        record = read_json(f"formal_final_full_s{seed}.json")
        close(record["physical"]["ood_macro_success"], scores[("full", seed)], f"final JSON seed {seed}")

    adaptation = read_csv("formal_adaptation_curves.csv")
    auc = {}
    budgets = [0, 128, 512, 2048]
    for age in [0, 32768]:
        for seed in sorted(SEEDS):
            values = []
            for budget in budgets:
                rows = [r for r in adaptation if int(r["seed"]) == seed
                        and int(r["age_interactions"]) == age and int(r["extra_interactions"]) == budget]
                require(len(rows) == 2 and {r["domain"] for r in rows} == {"new_dynamics", "new_delay"},
                        "Incomplete formal adaptation cell")
                values.append(statistics.mean(float(r["success_rate"]) for r in rows))
            auc[(age, seed)] = normalized_gain_auc(budgets, values)
    effects = {
        "final_full_minus_initial_ood_success": [scores[("full", s)] - scores[("initial", s)] for s in sorted(SEEDS)],
        "final_full_minus_no_action_ood_success": [scores[("full", s)] - scores[("no_action", s)] for s in sorted(SEEDS)],
        "adaptation_final_minus_initial_gain_auc": [auc[(32768, s)] - auc[(0, s)] for s in sorted(SEEDS)],
    }
    tests = summary["primary_tests"]
    require(len(tests) == 3 and {r["contrast"] for r in tests} == set(effects), "Wrong formal primary family")
    p_values = []
    for row in tests:
        differences = effects[row["contrast"]]
        for actual, archived in zip(differences, row["per_seed_differences"]):
            close(actual, archived, row["contrast"] + " seed difference")
        close(statistics.mean(differences), row["mean_difference"], row["contrast"] + " mean")
        p_value = sign_flip_p(differences)
        close(p_value, row["exact_sign_flip_p"], row["contrast"] + " exact p")
        p_values.append(p_value)
    for row, adjusted in zip(tests, holm_adjust(p_values)):
        close(adjusted, row["holm_p"], row["contrast"] + " Holm p")


def main():
    curves = read_csv("supplement_curves.csv")
    key_fields = ["phase", "split", "seed", "model", "arm", "source_age", "budget", "domain", "task", "training_domain"]
    keys = [tuple(row[field] for field in key_fields) for row in curves]
    require(len(set(keys)) == len(keys), "Duplicate supplemental evaluation cells")
    for row in curves:
        rate, n = float(row["success_rate"]), int(row["n_scenes"])
        require(0 <= rate <= 1 and n > 0, "Invalid success rate or scene count")
        close(rate * n, round(rate * n), "Integer success-count compatibility")
    derived = read_json("derived_summary.json")
    verify_supplement(curves, read_csv("supplement_contrasts.csv"), derived)
    verify_formal(derived)
    print(json.dumps({
        "status": "passed",
        "supplement_primary_contrasts_recomputed": 6,
        "formal_primary_contrasts_recomputed": 3,
        "training_seeds_per_contrast": 8,
        "sign_assignments_per_exact_test": 256,
        "holm_families": [6, 3],
        "supplement_evaluation_cells": len(curves),
        "supplement_logical_trial_rows": sum(int(r["n_scenes"]) for r in curves),
        "derived_tables_checked": list(derived),
        "confidence_intervals": "archived only; NOT recomputed",
        "physical_rollouts": "not re-executed or deduplicated",
    }, indent=2))


if __name__ == "__main__":
    main()
