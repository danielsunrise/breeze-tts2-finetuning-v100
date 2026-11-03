import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tools.bounded_lora import run_trial, separate_paths, validate_plan
from tools.prepare_cache import cache_command

ROOT = Path(__file__).resolve().parents[2]


def plan():
    return json.loads((ROOT / "configs/v100-lora.example.json").read_text())


def test_plan_rejects_extra_epochs_or_changed_schedule():
    assert validate_plan(plan(), 1000, 10)["validation_examples"] == 10
    with pytest.raises(ValueError, match="epoch budget"):
        validate_plan(plan(), 100, 10)
    changed = plan()
    changed["schedule_horizon_steps"] = 1000
    with pytest.raises(ValueError, match="horizon"):
        validate_plan(changed, 1000, 10)


def test_paths_cannot_write_into_shared_model(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        separate_paths(tmp_path / "model", tmp_path / "cache", tmp_path / "model/output")


def test_staged_resume_preserves_horizon_and_selects_best(tmp_path):
    model, cache, output = (tmp_path / name for name in ("model", "cache", "run"))
    model.mkdir()
    cache.mkdir()
    (model / "config.json").write_text("{}")
    (cache / "cache-receipt.json").write_text(json.dumps({"splits": {"train": [0] * 1000, "validation": [0] * 20}}))
    commands = []

    def run(command, cwd):
        commands.append(command)
        assert int(command[command.index("--max-steps") + 1]) == 200
        stage = int(command[command.index("--stop-after-step") + 1]) if "--stop-after-step" in command else 200
        assert ("--resume-checkpoint" in command) == (stage > 50)
        checkpoint = output / f"checkpoint-step-{stage:06d}"
        checkpoint.mkdir()
        (output / "latest.json").write_text(json.dumps({"global_step": stage, "checkpoint": str(checkpoint)}))
        (checkpoint / "trainer-state.json").write_text(json.dumps({"initial_validation": {"total": 7.0},
            "history": [{"validation_loss": {"total": 7.0 - stage / 1000}}], "micro_step": stage * 4}))
        return SimpleNamespace(returncode=75 if stage < 200 else 0)

    result = run_trial(plan(), model, cache, output, runner=run)
    assert result["completed_steps"] == 200
    assert result["best_step"] == 200
    assert result["base_files_unchanged"]
    assert len(commands) == 4


def test_converter_keeps_sources_and_transcript_groups(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    annotations = []
    hashes = {}
    for i in range(6):
        path = audio / f"clip_{i}.wav"
        t = np.arange(6000) / 24000
        sf.write(path, 0.1 * np.sin(2 * np.pi * (200 + i * 50) * t), 24000)
        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        text = "重复的测试句子。" if i < 2 else f"测试句子{i}。"
        annotations.append(f"C:\\old\\clip_{i}.wav|speaker_a|ZH|{text}\n")
    source = tmp_path / "source.list"
    source.write_text("".join(annotations), encoding="utf-8-sig")
    source_before = source.read_bytes()
    output = tmp_path / "converted"
    subprocess.run([sys.executable, str(ROOT / "tools/convert_gptsovits.py"),
                    "--list", str(source), "--audio-root", str(audio), "--output-root", str(output)],
                   check=True, capture_output=True)
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert len(train) + len(validation) == 6
    assert not {row["text"] for row in train} & {row["text"] for row in validation}
    assert all(Path(row["audio"]).is_absolute() for row in train + validation)
    assert source.read_bytes() == source_before
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    command = cache_command(tmp_path / "model", output / "train.jsonl", output / "validation.jsonl", tmp_path / "cache")
    assert command[command.index("--train-limit") + 1] == str(len(train))
    with pytest.raises(ValueError, match="disjoint"):
        cache_command(tmp_path / "model", output / "train.jsonl", output / "train.jsonl", tmp_path / "cache")


@pytest.mark.parametrize("configured,service_status,expected", [
    (False, 0, []),
    (True, 0, ["--user is-active --quiet test-tts.service", "--user stop test-tts.service", "--user start test-tts.service"]),
    (True, 3, ["--user is-active --quiet test-tts.service"]),
])
def test_gpu_wrapper_only_restores_previously_active_configured_service(tmp_path, configured, service_status, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    fake = bin_dir / "systemctl"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_EVENTS"\nif [ "$2" = is-active ]; then exit "$TEST_SERVICE_STATUS"; fi\n')
    fake.chmod(0o755)
    environment = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                       BREEZE_TTS_SERVICE="test-tts.service" if configured else "",
                       BREEZE_GPU_LOCK_FILE=str(tmp_path / "gpu.lock"),
                       TEST_EVENTS=str(events), TEST_SERVICE_STATUS=str(service_status))
    result = subprocess.run(["bash", str(ROOT / "tools/gpu_session.sh"), "bash", "-c", "exit 13"],
                            env=environment, capture_output=True)
    assert result.returncode == 13
    assert (events.read_text().splitlines() if events.exists() else []) == expected
