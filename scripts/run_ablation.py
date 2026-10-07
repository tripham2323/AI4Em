"""Phase 15 ablation orchestration: freeze the A-E matrix, validate runs, collect results.csv.

Subcommands
  freeze   write an immutable runs/ablation_<id>/matrix.json from configs/training.yaml
  collect  validate evaluation runs against the frozen matrix; write results.csv, paired.json, not_run.json
  train    delegate each variant/seed/fold to scripts/train_lstm.py / train_baseline.py (modeling owner);
           BLOCKED (exit 3) while those scripts do not exist; --dry-run writes train_plan.json only
Nothing here fabricates results: experiments without runs are listed as not run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.evaluation import evaluator as ev

SCHEMA_VERSION = "ablation_matrix_v1"
DEFAULT_OUTER_FOLDS = (1, 2, 3, 4, 5)
EXIT_BLOCKED = 3

# Signal groups from docs/18_Experiment_Plan.md. Validity masks stay with their signal group;
# perclos_ready only with PERCLOS; calibration_valid is present in every variant.
GROUPS: dict[str, tuple[str, ...]] = {
    "eye": ("ear_left_norm", "ear_right_norm", "closure_elapsed_s", "left_eye_valid", "right_eye_valid"),
    "perclos": ("perclos_60", "perclos_ready"),
    "mouth": ("mar_delta", "yawn_elapsed_s", "mouth_valid"),
    "pose": ("pitch_delta", "yaw_delta", "roll_delta", "pitch_velocity_dps", "pose_valid"),
    "always": ("calibration_valid",),
}
VARIANT_GROUPS = {
    "A": ("eye", "always"), "B": ("eye", "perclos", "always"),
    "C": ("eye", "perclos", "mouth", "always"), "D": ("eye", "perclos", "mouth", "pose", "always"),
    "E": ("eye", "perclos", "mouth", "pose", "always"),
}
EXTENSIONS = (
    {"experiment_id": "E03", "name": "window length 5/10/20 s", "status": "chưa thực hiện",
     "reason": "mở rộng theo docs/18; không phải acceptance của Phase 15"},
    {"experiment_id": "E04", "name": "landmark sampling FPS", "status": "chưa thực hiện",
     "reason": "mở rộng theo docs/18; không phải acceptance của Phase 15"},
    {"experiment_id": "E06", "name": "temporal order (reversed timesteps)", "status": "chưa thực hiện",
     "reason": "mở rộng theo docs/18; không phải acceptance của Phase 15"},
)
RESULT_COLUMNS = ["experiment_id", "model", "feature_set", "features", "input_size", "calibration_mode",
                  "outer_fold", "seed", "train_subjects", "accepted_windows", "coverage", "macro_f1",
                  "weighted_f1", "low_precision", "low_recall", "drowsy_recall", "video_accuracy", "p95_forward_ms"]
PAIRS = (("lstm_B_vs_A", "A", "B"), ("lstm_C_vs_B", "B", "C"), ("lstm_D_vs_C", "C", "D"),
         ("lstm_E_vs_D", "D", "E"), ("rf_control_vs_lstm_D", "RF_D", "D"))


class AblationError(ValueError):
    """Config/schema/consistency problem; CLI prints it and exits 2."""


class Blocked(RuntimeError):
    """A prerequisite is missing; CLI exits 3 without producing results."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ordered(feature_names: Sequence[str], groups: Sequence[str]) -> list[str]:
    wanted = {name for g in groups for name in GROUPS[g]}
    return [n for n in feature_names if n in wanted]


def build_matrix(config: Mapping[str, Any], *, seeds: Sequence[int], outer_folds: Sequence[int] = DEFAULT_OUTER_FOLDS,
                 config_sha256: str | None = None) -> dict[str, Any]:
    """Frozen A-E matrix plus the mandatory RF-full control. Feature order follows the config."""
    names = list(config.get("feature_names") or [])
    if not names or len(set(names)) != len(names):
        raise AblationError("config feature_names must be a non-empty list without duplicates")
    known = {n for g in GROUPS.values() for n in g}
    if set(names) != known:
        raise AblationError(f"config feature_names differ from ablation groups: "
                            f"missing {sorted(known - set(names))}, unknown {sorted(set(names) - known)}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise AblationError("seeds must be a non-empty list without duplicates")
    variants = []
    for exp, groups in VARIANT_GROUPS.items():
        feats = _ordered(names, groups)
        variants.append({"experiment_id": exp, "model": "lstm", "feature_set": exp, "feature_names": feats,
                         "input_size": len(feats), "calibration_mode": "P1" if exp == "E" else "P0"})
    full = _ordered(names, VARIANT_GROUPS["D"])
    variants.append({"experiment_id": "RF_D", "model": "rf", "feature_set": "D", "feature_names": full,
                     "input_size": len(full), "calibration_mode": "P0"})
    if variants[3]["feature_names"] != names:
        raise AblationError("variant D must contain every configured feature")
    body = {
        "schema_version": SCHEMA_VERSION, "config_sha256": config_sha256, "seeds": list(seeds),
        "outer_folds": list(outer_folds), "variants": variants, "extensions": list(EXTENSIONS),
        "fairness_rules": [
            "A-D and RF_D use the same P0 cohort, outer splits, seeds and quality policy",
            "E is compared with D only on the intersection of valid P1 subjects; D is re-evaluated with the "
            "matching calibration prefix excluded; full-cohort coverage is reported separately",
            "each variant has its own train-only scaler and checkpoint; no zeroing of channels of the full model",
        ],
    }
    body["matrix_sha256"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
    return body


def freeze(run_id: str, config_path: Path, seeds: Sequence[int] | None, out_root: Path,
           outer_folds: Sequence[int] = DEFAULT_OUTER_FOLDS) -> Path:
    if not config_path.is_file():
        raise AblationError(f"config not found: {config_path}")
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw) or {}
    seeds = list(seeds) if seeds else [config.get("seed", 42)]
    matrix = build_matrix(config, seeds=seeds, outer_folds=outer_folds,
                          config_sha256=hashlib.sha256(raw).hexdigest())
    target = out_root / f"ablation_{run_id}" / "matrix.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("matrix_sha256") != matrix["matrix_sha256"]:
            raise AblationError(f"{target} is already frozen with a different matrix; use a new --run-id")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    return target


TRAINER_SCRIPTS = {"lstm": Path("scripts/train_lstm.py"), "rf": Path("scripts/train_baseline.py")}


def train_command(spec: Mapping[str, Any], *, seed: int, fold: int, run_dir: Path, python: str = sys.executable) -> list[str]:
    """Proposed trainer contract (see docs/Team_01_Interface_Conventions.md section 6).

    The trainer is owned by the modeling owner; each variant gets its own feature order, train-only scaler and
    checkpoint, and must write predictions + a provenance JSON into ``run_dir`` for `scripts.evaluate`.
    """
    module = TRAINER_SCRIPTS[spec["model"]].with_suffix("").as_posix().replace("/", ".")
    return [python, "-m", module, "--experiment-id", spec["experiment_id"], "--feature-names",
            ",".join(spec["feature_names"]), "--calibration-mode", spec["calibration_mode"],
            "--seed", str(seed), "--outer-fold", str(fold), "--run-dir", str(run_dir)]


def plan_runs(matrix: Mapping[str, Any], out_dir: Path, experiments: Sequence[str] | None = None,
              folds: Sequence[int] | None = None) -> list[dict[str, Any]]:
    wanted = set(experiments) if experiments else None
    known = {v["experiment_id"] for v in matrix["variants"]}
    if wanted and wanted - known:
        raise AblationError(f"unknown experiments {sorted(wanted - known)}; frozen: {sorted(known)}")
    use_folds = list(folds) if folds else matrix["outer_folds"]
    if set(use_folds) - set(matrix["outer_folds"]):
        raise AblationError(f"folds {sorted(set(use_folds) - set(matrix['outer_folds']))} are not frozen outer folds")
    plan = []
    for spec in matrix["variants"]:
        if wanted and spec["experiment_id"] not in wanted:
            continue
        for seed in matrix["seeds"]:
            for fold in use_folds:
                run_dir = out_dir / "runs" / f"{spec['experiment_id']}_s{seed}_f{fold}"
                plan.append({"experiment_id": spec["experiment_id"], "seed": seed, "outer_fold": fold,
                             "run_dir": str(run_dir), "command": train_command(spec, seed=seed, fold=fold, run_dir=run_dir)})
    return plan


def run_experiment(spec: Mapping[str, Any], *, seed: int, fold: int, run_dir: Path) -> int:
    """Delegate one variant/seed/fold to the trainer script; BLOCKED if the owner has not delivered it."""
    import subprocess

    script = TRAINER_SCRIPTS[spec["model"]]
    if not script.is_file():
        raise Blocked(f"{script} does not exist yet (modeling owner, Phase 11/13); cannot train {spec['experiment_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(train_command(spec, seed=seed, fold=fold, run_dir=run_dir)).returncode


# ------------------------------------------------------------------ collect
def _load_matrix(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AblationError(f"frozen matrix not found: {path}; run `freeze` first")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in matrix.items() if k != "matrix_sha256"}
    if hashlib.sha256(_canonical(body).encode()).hexdigest() != matrix.get("matrix_sha256"):
        raise AblationError("matrix.json was modified after freezing")
    return matrix


def _train_subjects(prov: Mapping[str, Any], fold: str) -> Any:
    ts = prov.get("train_subjects")
    if isinstance(ts, Mapping):
        return ts.get(fold, ts.get(int(fold) if fold.isdigit() else fold, ""))
    return "" if ts is None else ts


def collect_rows(matrix: Mapping[str, Any], eval_runs: Sequence[Path]) -> tuple[list[dict], dict]:
    """Rows for results.csv plus per-experiment subject data; validates every run against the matrix."""
    variants = {v["experiment_id"]: v for v in matrix["variants"]}
    rows: list[dict] = []
    cohorts: dict[tuple, dict[str, set]] = {}
    per_subject: dict[str, dict[str, list[float]]] = {}
    seen: set[tuple] = set()
    for run in eval_runs:
        mpath = run / "metrics.json"
        if not mpath.is_file():
            raise AblationError(f"{run}: metrics.json missing")
        report = json.loads(mpath.read_text(encoding="utf-8"))
        prov = (report.get("provenance") or {}).get("user_provenance") or {}
        exp = prov.get("experiment_id")
        if exp not in variants:
            raise AblationError(f"{run}: experiment_id {exp!r} is not in the frozen matrix")
        spec, seed = variants[exp], prov.get("seed")
        if seed not in matrix["seeds"]:
            raise AblationError(f"{run}: seed {seed!r} is not a frozen seed {matrix['seeds']}")
        if prov.get("feature_names") != spec["feature_names"]:
            raise AblationError(f"{run}: feature_names/order do not match variant {exp}")
        if prov.get("calibration_mode") != spec["calibration_mode"]:
            raise AblationError(f"{run}: calibration_mode must be {spec['calibration_mode']} for {exp}")
        for fold, rep in report["folds"].items():
            if int(fold) not in matrix["outer_folds"]:
                raise AblationError(f"{run}: fold {fold} is not a frozen outer fold")
            key = (exp, seed, int(fold))
            if key in seen:
                raise AblationError(f"duplicate run for experiment {exp}, seed {seed}, fold {fold}")
            seen.add(key)
            w, m = rep["window"], rep["window"]["metrics"]
            rows.append({
                "experiment_id": exp, "model": spec["model"], "feature_set": spec["feature_set"],
                "features": ";".join(spec["feature_names"]), "input_size": spec["input_size"],
                "calibration_mode": spec["calibration_mode"], "outer_fold": int(fold), "seed": seed,
                "train_subjects": _train_subjects(prov, fold), "accepted_windows": w["counts"]["accepted"],
                "coverage": w["coverage"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                "low_precision": m["low_vigilance_precision"], "low_recall": m["low_vigilance_recall"],
                "drowsy_recall": m["drowsy_recall"], "video_accuracy": rep["video"]["report"]["metrics"]["accuracy"],
                "p95_forward_ms": prov.get("p95_forward_ms"),
            })
            subs = rep["subject"]["per_subject"]
            cohorts.setdefault((seed, int(fold)), {})[exp] = set(subs)
            for s, v in subs.items():
                if v["metrics"]["macro_f1"] is not None:
                    per_subject.setdefault(exp, {}).setdefault(s, []).append(v["metrics"]["macro_f1"])
    for (seed, fold), by_exp in cohorts.items():  # A-D and RF_D must share the P0 cohort
        p0 = {e: s for e, s in by_exp.items() if variants[e]["calibration_mode"] == "P0"}
        if len({frozenset(s) for s in p0.values()}) > 1:
            raise AblationError(f"P0 variants disagree on the test cohort for seed {seed}, fold {fold}: "
                                f"{ {e: sorted(s) for e, s in p0.items()} }")
    return rows, per_subject


def compare_paired_subjects(per_subject: Mapping[str, Mapping[str, Sequence[float]]], *, n_boot: int = 1000,
                            seed: int = 0) -> dict[str, Any]:
    """Paired per-subject Macro F1 difference (b - a) with bootstrap CI for the declared pairs."""
    out: dict[str, Any] = {}
    for name, a, b in PAIRS:
        if a not in per_subject or b not in per_subject:
            out[name] = {"status": "chưa thực hiện", "reason": f"missing runs for {a if a not in per_subject else b}"}
            continue
        mean = lambda d: {s: sum(v) / len(v) for s, v in d.items()}  # average over seeds per subject
        res = ev.paired_subject_difference(mean(per_subject[a]), mean(per_subject[b]), n_boot=n_boot, seed=seed)
        lo, hi = res["ci_low"], res["ci_high"]
        if lo is None:
            hint = "insufficient_evidence"
        elif lo > 0:
            hint = "gain_ci_excludes_zero"
        elif hi < 0:
            hint = "loss_ci_excludes_zero"
        else:
            hint = "inconclusive"
        out[name] = {**res, "metric": "macro_f1", "evidence_hint": hint,
                     "note": "hint only; keep/drop decisions need human review of CI width and subject count"}
    return out


def write_results(out_dir: Path, matrix: Mapping[str, Any], rows: list[dict], paired: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in RESULT_COLUMNS})  # undefined stays empty
    ran = {r["experiment_id"] for r in rows}
    not_run = [{"experiment_id": v["experiment_id"], "status": "chưa thực hiện",
                "reason": "no evaluation run supplied"} for v in matrix["variants"] if v["experiment_id"] not in ran]
    not_run += list(matrix["extensions"])
    (out_dir / "not_run.json").write_text(json.dumps(not_run, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "paired.json").write_text(json.dumps(ev.clean_for_json(paired), indent=2, sort_keys=True,
                                                    ensure_ascii=False, allow_nan=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="write the frozen experiment matrix")
    f.add_argument("--run-id", required=True)
    f.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    f.add_argument("--seeds", type=int, nargs="+", help="default: config seed")
    f.add_argument("--out-root", type=Path, default=Path("runs"))
    c = sub.add_parser("collect", help="validate evaluation runs and write results.csv")
    c.add_argument("--run-id", required=True)
    c.add_argument("--eval-run", type=Path, nargs="+", required=True, help="run dirs produced by scripts.evaluate")
    c.add_argument("--out-root", type=Path, default=Path("runs"))
    c.add_argument("--n-boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train", help="train each frozen variant via the owner's trainer scripts")
    t.add_argument("--run-id", required=True)
    t.add_argument("--out-root", type=Path, default=Path("runs"))
    t.add_argument("--experiments", nargs="+", help="subset of frozen experiment ids (default all)")
    t.add_argument("--folds", type=int, nargs="+", help="subset of frozen outer folds (default all)")
    t.add_argument("--dry-run", action="store_true", help="only write train_plan.json and print the commands")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            print(f"frozen: {freeze(args.run_id, args.config, args.seeds, args.out_root)}")
        elif args.command == "collect":
            out = args.out_root / f"ablation_{args.run_id}"
            matrix = _load_matrix(out / "matrix.json")
            rows, per_subject = collect_rows(matrix, args.eval_run)
            write_results(out, matrix, rows, compare_paired_subjects(per_subject, n_boot=args.n_boot, seed=args.seed))
            print(f"wrote {out / 'results.csv'} ({len(rows)} rows)")
        else:
            out = args.out_root / f"ablation_{args.run_id}"
            matrix = _load_matrix(out / "matrix.json")
            plan = plan_runs(matrix, out, args.experiments, args.folds)
            (out / "train_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            if args.dry_run:
                for item in plan:
                    print(" ".join(item["command"]))
                print(f"wrote {out / 'train_plan.json'} ({len(plan)} runs, nothing trained)")
                return 0
            specs = {v["experiment_id"]: v for v in matrix["variants"]}
            for item in plan:
                code = run_experiment(specs[item["experiment_id"]], seed=item["seed"], fold=item["outer_fold"],
                                      run_dir=Path(item["run_dir"]))
                if code != 0:
                    print(f"run_ablation: trainer failed ({code}) for {item['experiment_id']} "
                          f"seed {item['seed']} fold {item['outer_fold']}", file=sys.stderr)
                    return code
            print(f"trained {len(plan)} runs; next: scripts.evaluate per run, then `collect`")
    except Blocked as exc:
        print(f"run_ablation: BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except (AblationError, json.JSONDecodeError, KeyError) as exc:
        print(f"run_ablation: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
