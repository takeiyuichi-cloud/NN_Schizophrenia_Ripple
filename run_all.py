#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: reproduce all figures and supplementary tables for the manuscript.

Usage (from repository root):
  python run_all.py
  python run_all.py --dry-run
  python run_all.py --only fig3 fig4
  python run_all.py --skip fig2a_nifti
  python run_all.py --continue-on-error

Notes:
- This script assumes individual analysis scripts are located under ./scripts/
- Outputs are written by each script to ./outputs/ (or project-defined output paths).
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


# -------------------------
# Pipeline definition (edit here if needed)
# -------------------------
PIPELINE = [
    # ---- Fig 1 ----
    ("fig1", "Fig1: counts/durations + supp tables", [
        "fig1_ac_with_supp_table12.py",
        "fig1_b.py",
        "fig1_d_with_supp_table3.py",
    ]),

    # ---- Fig 2 ----
    ("fig2", "Fig2: network composition + IEI + (optional NIfTI maps already in fig2a/)", [
        "fig2_a.py",
        "fig2_b_with_supp_table4.py",
    ]),

    # ---- Fig 3 ----
    ("fig3", "Fig3: high-rate epochs + S5/S6/S7", [
        "fig3_a.py",
        "fig3_bde_with_supp_table5_to7.py",
        "fig3_c.py",
        # S6 is sometimes implemented as a separate builder:
        "supp_table6.py",
    ]),

    # ---- Fig 4 ----
    ("fig4", "Fig4: transition share + IEI + supplementary tables", [
        "fig4_a_with_supp_table8.py",
        "fig4_b_with_supp_table9.py",
        "fig4_c.py",
        "fig4_d.py",
        # S10 may be tied to Fig5a, but keeping it near downstream is OK
    ]),

    # ---- Fig 5 ----
    ("fig5", "Fig5: symptom associations + supplementary tables", [
        "fig5_a.py",
        "fig5_b.py",
        "fig5_c.py",
        "fig5_d.py",
        "supp_table10.py",
        "supp_table11.py",
        "supp_table12_builder.py",
    ]),
]

# Optional extra step: Fig2a NIfTI outputs are already provided as files (no script required).
# Keep a skip-token here in case you later add a generator script.
OPTIONAL_STEPS = {
    "fig2a_nifti": "Fig2a: NIfTI group maps (already in data/fig2a/; no computation)",
}


# -------------------------
# Helpers
# -------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def run_one(script_path: Path, *, python_exe: str, cwd: Path, env: dict, log_fp, dry_run: bool) -> int:
    cmd = [python_exe, str(script_path)]
    print(f"\n>>> RUN: {' '.join(cmd)}", flush=True)
    log_fp.write(f"\n>>> RUN: {' '.join(cmd)}\n")
    log_fp.flush()

    if dry_run:
        print(">>> DRY-RUN: skipped execution.", flush=True)
        log_fp.write(">>> DRY-RUN: skipped execution.\n")
        log_fp.flush()
        return 0

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # stream output to console and to log file
    if p.stdout:
        print(p.stdout, flush=True)
        log_fp.write(p.stdout)
        log_fp.flush()
    return int(p.returncode)


def normalize_tags(tag_list: list[str] | None, valid: set[str]) -> set[str]:
    if not tag_list:
        return set(valid)
    out = set()
    for t in tag_list:
        t2 = str(t).strip()
        if t2 in valid:
            out.add(t2)
        else:
            raise ValueError(f"Unknown tag: '{t2}'. Valid tags: {sorted(valid)}")
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", type=str, default="scripts",
                    help="Directory containing analysis scripts (default: scripts)")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python executable to use (default: current interpreter)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only selected tags (e.g., --only fig3 fig4)")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="Skip selected tags or optional steps (e.g., --skip fig2 fig2a_nifti)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned commands without executing")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Continue execution even if a script fails (NOT recommended for formal reproduction)")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    scripts_dir = (repo_root / args.scripts_dir).resolve()

    if not scripts_dir.exists():
        raise FileNotFoundError(f"scripts directory not found: {scripts_dir}")

    valid_tags = {t for (t, _desc, _lst) in PIPELINE}
    valid_optional = set(OPTIONAL_STEPS.keys())
    valid_all = valid_tags | valid_optional

    only_set = normalize_tags(args.only, valid_tags)
    skip_set = set()
    if args.skip:
        for t in args.skip:
            t2 = str(t).strip()
            if t2 not in valid_all:
                raise ValueError(f"Unknown skip target: '{t2}'. Valid: {sorted(valid_all)}")
            skip_set.add(t2)

    # -------------------------
    # Ensure output directories
    # -------------------------
    outputs_root = repo_root / "outputs"
    (outputs_root / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_root / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_root / "logs").mkdir(parents=True, exist_ok=True)

    # logging
    log_path = outputs_root / "logs" / f"run_all_log_{now_str()}.txt"

    # environment (ensure consistent working directory)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print("\n=== Reproduction pipeline (run_all.py) ===")
    print("Repository root:", repo_root)
    print("Scripts dir:", scripts_dir)
    print("Python:", args.python)
    print("Only tags:", sorted(only_set))
    print("Skip:", sorted(skip_set))
    print("Dry-run:", args.dry_run)
    print("Log file:", log_path)
    print("=========================================\n")

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write("=== Reproduction pipeline (run_all.py) ===\n")
        log_fp.write(f"Repository root: {repo_root}\n")
        log_fp.write(f"Scripts dir: {scripts_dir}\n")
        log_fp.write(f"Python: {args.python}\n")
        log_fp.write(f"Only tags: {sorted(only_set)}\n")
        log_fp.write(f"Skip: {sorted(skip_set)}\n")
        log_fp.write(f"Dry-run: {args.dry_run}\n")
        log_fp.write(f"Started: {datetime.now().isoformat()}\n")
        log_fp.write("=========================================\n\n")
        log_fp.flush()

        failures = []

        # optional steps (currently informational)
        for opt, desc in OPTIONAL_STEPS.items():
            if opt in skip_set:
                continue
            # Only run optional steps if all tags are run OR user explicitly includes everything.
            # Here: informational only.
            log_fp.write(f"[INFO] Optional step: {opt} — {desc}\n")
            log_fp.flush()

        # main pipeline
        for tag, desc, script_list in PIPELINE:
            if tag not in only_set:
                continue
            if tag in skip_set:
                print(f"--- SKIP TAG: {tag} ---")
                log_fp.write(f"--- SKIP TAG: {tag} ---\n")
                continue

            print(f"\n=== [{tag}] {desc} ===", flush=True)
            log_fp.write(f"\n=== [{tag}] {desc} ===\n")
            log_fp.flush()

            for script_name in script_list:
                script_path = scripts_dir / script_name
                if not script_path.exists():
                    msg = f"[ERROR] Missing script: {script_path}"
                    print(msg, flush=True)
                    log_fp.write(msg + "\n")
                    log_fp.flush()
                    failures.append((tag, script_name, "missing"))
                    if not args.continue_on_error:
                        print(f"\nStopped due to missing script. See log: {log_path}", flush=True)
                        sys.exit(1)
                    continue

                rc = run_one(
                    script_path,
                    python_exe=args.python,
                    cwd=repo_root,
                    env=env,
                    log_fp=log_fp,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    msg = f"[FAIL] {script_name} (exit code {rc})"
                    print(msg, flush=True)
                    log_fp.write(msg + "\n")
                    log_fp.flush()
                    failures.append((tag, script_name, f"exit{rc}"))
                    if not args.continue_on_error:
                        print(f"\nStopped due to error. See log: {log_path}", flush=True)
                        sys.exit(rc)

        # summary
        if failures:
            print("\n=== PIPELINE FINISHED WITH FAILURES ===")
            for f in failures:
                print(" -", f)
            print("Log:", log_path)
            log_fp.write("\n=== PIPELINE FINISHED WITH FAILURES ===\n")
            for f in failures:
                log_fp.write(f" - {f}\n")
            log_fp.write(f"Finished: {datetime.now().isoformat()}\n")
            log_fp.flush()
            sys.exit(2)

        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Log:", log_path)
        log_fp.write("\n=== PIPELINE COMPLETED SUCCESSFULLY ===\n")
        log_fp.write(f"Finished: {datetime.now().isoformat()}\n")
        log_fp.flush()


if __name__ == "__main__":
    main()
