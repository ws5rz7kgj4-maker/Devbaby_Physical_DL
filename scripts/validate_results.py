#!/usr/bin/env python3
"""Validate this result snapshot, not the neural-training implementation.

Checks archived input hashes, paired scene identities, measurement aggregation,
parameter-audit claims, and every generated CSV value. No GPU or network needed.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from build_results import ROOT, csv_text, read_json, tables


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), message)


def main():
    import json
    manifest = json.loads((ROOT / "results/source_sha256.json").read_text())
    require(len(manifest) == 10, "Expected the 10 audited input JSON files.")
    for path, expected in manifest.items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                f"Source bytes changed: {path}")
    early_manifest = json.loads((ROOT / "results/early_source_sha256.json").read_text())
    require(len(early_manifest) == 5, "Expected five early physical-v1 input files.")
    for path, expected in early_manifest.items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                f"Early source bytes changed: {path}")

    protocol = read_json("pilot_protocol.json")
    require(protocol["development_only"] is True, "Pilot must remain development only.")
    require(protocol["steps"] == 128 and protocol["seeds"] == 8, "Pilot design mismatch.")
    episodes = read_json("pilot_episodes_enriched.json")
    require(len(episodes) == 96 and len(protocol["cells"]) == 12, "Pilot row count mismatch.")
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["cell"]].append(row)
        for key in ("contact_fraction", "action_effect_2cm_fraction", "action_saturation_fraction"):
            require(math.isfinite(row[key]) and 0 <= row[key] <= 1, f"Invalid fraction: {key}")
    expected_pairs = None
    for cell, _, _ in protocol["cells"]:
        pairs = {(r["seed_index"], r["scene_seed"]) for r in grouped[cell]}
        require(len(grouped[cell]) == len(pairs) == 8, f"Missing/duplicate pilot scene: {cell}")
        if expected_pairs is None:
            expected_pairs = pairs
        require(pairs == expected_pairs, f"Unpaired pilot scenes: {cell}")

    data = tables()
    early_seed = data["early_development_by_seed.csv"]
    require(len(early_seed) == 160, "Early development seed panel count mismatch.")
    for row in early_seed:
        expected_cells = 3 if row["scope"] == "iid" else 12
        require(row["domain_task_cells"] == expected_cells and row["scored_trials"] == expected_cells * 6,
                "Early development panel denominator mismatch.")
        close(row["macro_success"], row["successes"] / row["scored_trials"], "Early per-seed macro mismatch.")
    early_ood = [r for r in data["early_development_macro.csv"] if r["scope"] == "ood" and r["variant"] == "full"]
    require([r["successes"] for r in early_ood] == [2, 360, 380, 401, 427], "Early OOD development counts mismatch.")
    require(all(r["scored_trials"] == 576 and r["model_seeds"] == 8 for r in early_ood), "Early OOD denominator mismatch.")
    early_final = {r["condition"]: r for r in data["early_sealed_final_comparison.csv"]}
    es = read_json("early/formal_summary.json")
    for name, key in [("full", "final_full_mean"), ("initial", "final_initial_mean"), ("no_action", "final_no_action_mean")]:
        close(early_final[name]["macro_success"], es[key], f"Sealed-final early summary mismatch: {name}")
    require(early_final["full"]["successes"] == 1680 and early_final["full"]["scored_trials"] == 2304,
            "Sealed-final early full count mismatch.")
    for row in early_final.values():
        close(row["macro_success"], row["successes"] / row["scored_trials"], "Early final count/macro mismatch.")
    pilot_means = read_json("pilot_summary.json")["means"]
    analysis = read_json("pilot_analysis.json")["table"]
    for row in data["pilot_conditions.csv"]:
        cell = row["condition"]
        close(row["contact_fraction"], pilot_means[cell]["contact_fraction"], "Contact aggregate mismatch.")
        close(row["contact_fraction"], analysis[cell]["contact_fraction"]["mean"], "Analysis contact mismatch.")
        close(row["action_effect_2cm_fraction"], analysis[cell]["action_effect_2cm_fraction"]["mean"],
              "Action-effect aggregate mismatch.")
    ref = {r["task"]: r for r in data["pilot_reference.csv"]}
    require((ref["relocate"]["successes"], ref["relocate"]["trials"]) == (7, 8), "Reference relocate mismatch.")
    require((ref["lift"]["successes"], ref["lift"]["trials"]) == (8, 8), "Reference lift mismatch.")

    bp = read_json("baseline_protocol.json")
    require(bp["precision"] == "fp32" and bp["updates"] == 200, "Baseline protocol mismatch.")
    require(set(bp["training_episodes"]).isdisjoint(bp["development_validation_episodes"]),
            "Baseline train/validation episodes overlap.")
    baseline = data["baseline_all.csv"]
    require(len(baseline) == 48, "Baseline panel count mismatch.")
    keys = {(r["treatment"], r["split"], r["update"], r["horizon"]) for r in baseline}
    expected = {(t, s, u, h) for t in ("original", "change_balanced")
                for s in ("train", "development_validation") for u in (0, 50, 200)
                for h in (1, 8, 32, 64)}
    require(keys == expected, "Baseline missing/duplicate measurement cells.")
    for row in data["baseline_200_validation.csv"]:
        require(row["actual_object_mse"] > row["persistence_object_mse"],
                "Recorded 200-update prediction comparison changed.")

    for check in read_json("parameter_cpu_result.json")["checkpoints"]:
        require(check["seed"] in (0, 1), "Unexpected audited checkpoint seed.")
        rr = [p for p in check["parameters"] if p["name"].split(".")[0] in ("dynamics", "filter")]
        require(len(rr) == 8, "Expected eight recurrent parameter tensors.")
        require(all(p["exact_equal"] and p["max_abs_change"] == 0 and not p["old_optimizer_has_state"]
                    for p in rr), "Recurrent checkpoint freeze evidence changed.")
    gpu = read_json("parameter_gpu_result.json")
    checks = {r["mode"]: r["parameters"] for r in gpu["path_checks"]}
    require(set(checks) == {"fp32", "amp_cache_on", "amp_cache_off"}, "GPU mode mismatch.")
    for mode, rr in checks.items():
        require(len(rr) == 44 and len({r["name"] for r in rr}) == 44, "GPU parameter count mismatch.")
        recurrent = [p for p in rr if p["name"].split(".")[0] in ("dynamics", "filter")]
        if mode == "amp_cache_on":
            require(all(not p["grad_present"] and p["step_max_abs_change"] == 0 for p in recurrent),
                    "Cached recurrent freeze evidence changed.")
            require(sum(not p["grad_present"] for p in rr) == 26, "Cached absent-gradient count mismatch.")
        else:
            require(all(p["grad_present"] and p["grad_finite"] for p in rr), "Missing/nonfinite gradient.")
            require(all(p["step_max_abs_change"] > 0 for p in recurrent), "Recurrent update missing.")
        if mode == "amp_cache_off":
            zero = {p["name"] for p in rr if p["step_max_abs_change"] == 0}
            require(zero == {"image_start.weight", "image_start.bias"}, "FP16 single-step caveat changed.")

    frozen = data["frozen_control_all.csv"]
    require(len(frozen) == 12, "Frozen audit panel count mismatch.")
    require(all(r["ranking_seed_scene_rows"] == 32 for r in frozen), "Ranking row count mismatch.")
    for name, rows in data.items():
        path = ROOT / "results/tables" / name
        require(path.read_text(encoding="utf-8") == csv_text(rows), f"Derived CSV differs: {name}")
    print(f"PASS: {len(manifest) + len(early_manifest)} source hashes; early physical-v1 panels; paired pilot; FP32 panels; parameter audits; {len(data)} CSV tables.")
    print("This verifies documentation/data consistency only; it does not test the repaired optimizer or retrain the model.")


if __name__ == "__main__":
    main()
