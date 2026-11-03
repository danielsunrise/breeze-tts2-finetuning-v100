"""Prepare every row from separate train/validation manifests using shared weights."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from training.real_data import read_manifest


def cache_command(model, train, validation, output):
    model, output = model.resolve(), output.resolve()
    if model == output or model in output.parents or output in model.parents:
        raise ValueError("Cache output must not overlap the shared model directory")
    train_rows, validation_rows = read_manifest(train), read_manifest(validation)
    if {row.audio.resolve() for row in train_rows} & {row.audio.resolve() for row in validation_rows}:
        raise ValueError("Training and validation audio files must be disjoint")
    return [sys.executable, "-m", "training.real_data", "--model-root", str(model),
            "--train-manifest", str(train.resolve()), "--validation-manifest", str(validation.resolve()),
            "--output-root", str(output), "--train-limit", str(len(train_rows)),
            "--validation-limit", str(len(validation_rows))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, variable in (("model-root", "BREEZE_MODEL_ROOT"), ("train-manifest", "BREEZE_TRAIN_MANIFEST"),
                           ("validation-manifest", "BREEZE_VALIDATION_MANIFEST"), ("output-root", "BREEZE_CACHE_ROOT")):
        parser.add_argument("--" + name, type=Path, default=os.environ.get(variable))
    args = parser.parse_args()
    if any(value is None for value in vars(args).values()):
        parser.error("Set model, train/validation manifests, and cache output paths")
    command = cache_command(args.model_root, args.train_manifest, args.validation_manifest, args.output_root)
    raise SystemExit(subprocess.call(command, cwd=Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    main()
