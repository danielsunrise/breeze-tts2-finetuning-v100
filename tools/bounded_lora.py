"""Keep one fixed LR horizon while pausing/checking a capped LoRA trial."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_plan(plan, train_rows, validation_rows):
    plan = dict(plan)
    if train_rows <= 0 or validation_rows <= 0:
        raise ValueError("Training and validation caches must both be nonempty")
    for key in ("max_steps", "gradient_accumulation", "rank", "save_every",
                "validation_examples", "early_stop_patience"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if plan.get("schedule_horizon_steps", plan["max_steps"]) != plan["max_steps"]:
        raise ValueError("Schedule horizon must equal max_steps at every stage")
    presentations = plan["max_steps"] * plan["gradient_accumulation"]
    if presentations > plan["max_example_presentations"]:
        raise ValueError("Trial exceeds the configured example-presentation budget")
    epochs = float(plan.get("max_effective_epochs", 1.0))
    if not math.isfinite(epochs) or epochs <= 0 or presentations > train_rows * epochs:
        raise ValueError("Trial exceeds the configured epoch budget; reduce max_steps")
    stages = plan["staged_pause_steps"]
    if (any(type(step) is not int or not 0 < step < plan["max_steps"] for step in stages)
            or stages != sorted(set(stages))):
        raise ValueError("Pause steps must be unique, sorted, and below max_steps")
    for key in ("alpha", "learning_rate", "early_stop_min_delta"):
        if not math.isfinite(float(plan[key])) or float(plan[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= plan["warmup_steps"] < plan["max_steps"]:
        raise ValueError("warmup_steps must be within the fixed schedule")
    plan.update(train_rows=train_rows, validation_rows=validation_rows,
                validation_examples=min(plan["validation_examples"], validation_rows))
    return plan


def separate_paths(model, cache, run):
    model, cache, run = (path.resolve() for path in (model, cache, run))
    for first, second in ((model, cache), (model, run), (cache, run)):
        if first == second or first in second.parents or second in first.parents:
            raise ValueError("Model, cache, and run directories must not overlap")


def snapshot(model):
    return {str(p.relative_to(model)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in model.rglob("*") if p.is_file() and ".cache" not in p.parts}


def run_trial(plan, model, cache, run, runner=subprocess.run):
    separate_paths(model, cache, run)
    receipt = json.loads((cache / "cache-receipt.json").read_text())
    plan = validate_plan(plan, len(receipt["splits"]["train"]), len(receipt["splits"]["validation"]))
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "trial-plan.json", plan)
    before = snapshot(model)
    write_json(run / "base-before.json", before)
    state = dict(status="running", cap_steps=plan["max_steps"],
                 cap_example_presentations=plan["max_steps"] * plan["gradient_accumulation"],
                 completed_steps=0, stages=[], started_unix=time.time(),
                 initial_validation=None, best_checkpoint=None, best_step=None,
                 best_validation=None)
    write_json(run / "trial-state.json", state)
    command = [sys.executable, "-u", "-m", "training.real_lora",
               "--model-root", str(model), "--cache-root", str(cache), "--output-root", str(run)]
    for key in ("rank", "alpha", "max_steps", "gradient_accumulation", "learning_rate",
                "warmup_steps", "save_every", "validation_examples", "seed"):
        command.extend(["--" + key.replace("_", "-"), str(plan[key])])
    resume = None
    patience_reference = None
    bad_checks = 0
    try:
        for stage in [*plan["staged_pause_steps"], plan["max_steps"]]:
            cmd = list(command)
            if resume is not None:
                cmd += ["--resume-checkpoint", str(resume)]
            if stage < plan["max_steps"]:
                cmd += ["--stop-after-step", str(stage)]
            print(json.dumps(dict(event="stage_start", target_step=stage, cap_steps=plan["max_steps"])), flush=True)
            completed = runner(cmd, cwd=SOURCE)
            if completed.returncode != (75 if stage < plan["max_steps"] else 0):
                raise RuntimeError(f"Training subprocess failed: exit={completed.returncode}")
            latest = json.loads((run / "latest.json").read_text())
            if latest["global_step"] != stage:
                raise RuntimeError("Training did not reach the requested pause step")
            resume = Path(latest["checkpoint"])
            trainer = json.loads((resume / "trainer-state.json").read_text())
            baseline = trainer["initial_validation"]["total"]
            score = trainer["history"][-1]["validation_loss"]["total"]
            if not math.isfinite(score) or not math.isfinite(baseline):
                raise RuntimeError("Non-finite validation loss")
            state.update(initial_validation=baseline, completed_steps=stage)
            if state["best_validation"] is None or score < state["best_validation"]:
                state.update(best_validation=score, best_step=stage, best_checkpoint=str(resume))
            if patience_reference is None:
                patience_reference = baseline
            if score < patience_reference - plan["early_stop_min_delta"]:
                patience_reference, bad_checks = score, 0
            else:
                bad_checks += 1
            row = dict(step=stage, validation_loss=score, no_improvement_checks=bad_checks,
                       example_presentations=trainer["micro_step"],
                       effective_epochs=trainer["micro_step"] / plan["train_rows"])
            state["stages"].append(row)
            write_json(run / "trial-state.json", state)
            print(json.dumps(dict(event="stage_validated", **row)), flush=True)
            if bad_checks >= plan["early_stop_patience"]:
                state.update(status="completed_with_early_stop", stop_reason="held_out_loss_not_improving")
                break
        else:
            state.update(status="completed_at_user_cap", stop_reason="step_budget")
        state.update(finished_unix=time.time(), base_files_unchanged=before == snapshot(model),
                     selected_improves_vs_base=state["best_validation"] < state["initial_validation"])
        if not state["base_files_unchanged"]:
            raise RuntimeError("Shared base model file sizes/timestamps changed")
        write_json(run / "trial-state.json", state)
        write_json(run / "selected-checkpoint.json", dict(
            checkpoint=state["best_checkpoint"], step=state["best_step"],
            validation_loss=state["best_validation"], initial_validation=state["initial_validation"],
            completed_steps=state["completed_steps"], improves_vs_base=state["selected_improves_vs_base"],
            selection_policy="lowest_fixed_held_out_loss_among_saved_adapters"))
    except BaseException as error:
        state.update(status="failed_or_interrupted", error=str(error))
        write_json(run / "trial-state.json", state)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    for name in ("model", "cache", "run"):
        parser.add_argument("--" + ("output" if name == "run" else name) + "-root",
                            dest=name, type=Path, default=os.environ.get("BREEZE_" + name.upper() + "_ROOT"))
    args = parser.parse_args()
    if any(getattr(args, name) is None for name in ("model", "cache", "run")):
        parser.error("Set model/cache/output roots explicitly or through BREEZE_*_ROOT")
    state = run_trial(json.loads(args.plan.read_text()), args.model.resolve(), args.cache.resolve(), args.run.resolve())
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
