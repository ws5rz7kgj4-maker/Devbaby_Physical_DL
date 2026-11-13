#!/usr/bin/env python3
"""Check documentation release integrity; never imports/runs training code."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ROOT / "provenance/content_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["snapshot_date"] == "2026-11-11", "Unexpected snapshot date")
    for name, expected in manifest["files"].items():
        p = ROOT / name
        require(p.is_file(), f"Missing release file: {name}")
        require(digest(p) == expected, f"Release file changed: {name}")

    en = {p.name for p in (ROOT / "docs/en").glob("*.md")}
    zh = {p.name for p in (ROOT / "docs/zh").glob("*.md")}
    require(en == zh and len(en) == 6, "Bilingual document set mismatch")
    docs = [ROOT / n for n in ("README.md", "README.zh-CN.md", "CONTRIBUTING.md")]
    docs += list((ROOT / "docs").rglob("*.md"))
    docs += [ROOT / n for n in ("results/README.md", "provenance/README.md", "code/gradient_audit/README.md")]
    link_count = 0
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        require("/workspace/scratch/" not in text, f"Machine-local link in {doc}")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.split(' "', 1)[0].strip("<>")
            if not target or target.startswith("#") or urlsplit(target).scheme:
                continue
            path = unquote(target.split("#", 1)[0])
            require((doc.parent / path).exists(), f"Broken local link: {doc.relative_to(ROOT)} -> {target}")
            link_count += 1

    for name in ("early_physical_learning", "pilot_coverage", "fp32_prediction_learning"):
        for suffix in ("png", "svg"):
            require((ROOT / "figures" / f"{name}.{suffix}").stat().st_size > 1000,
                    f"Missing/empty figure: {name}.{suffix}")

    source_archives = json.loads((ROOT / "provenance/source_archives.json").read_text())
    included = [x for x in source_archives if x["included"]]
    require(len(included) == 6, "Expected six compact source archives")
    for row in included:
        p = ROOT / "provenance/source_archives" / row["filename"]
        require(p.stat().st_size == row["bytes"] and digest(p) == row["sha256"],
                f"Source archive mismatch: {row['filename']}")

    coverage = [json.loads(p.read_text()) for p in sorted((ROOT / "results/raw/worldmodel_coverage").glob("seed*.json"))]
    require(len(coverage) == 8, "Expected eight original training-coverage records")
    require(sum(x["transitions"] for x in coverage) == 524288, "Coverage transition total mismatch")
    require(sum(x["episodes"] for x in coverage) == 4096, "Coverage episode total mismatch")
    for row in coverage:
        for key in ("frames_with_hand_contact", "frames_with_grasp", "frames_with_moving_object"):
            require(abs(row[key] / row["transitions"] - row[key + "_fraction"]) < 1e-14,
                    "Coverage rate differs from count")

    for p in (ROOT / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    for p in (ROOT / "code/gradient_audit").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

    print(f"PASS: {len(manifest['files'])} release hashes, {len(docs)} documentation files, "
          f"{link_count} local links, bilingual pairs, figures, archive identity and coverage counts.")
    print("No training, remote calls, cloud submissions or scientific-training reproduction performed.")


if __name__ == "__main__":
    main()
