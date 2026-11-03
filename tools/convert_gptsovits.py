"""Convert GPT-SoVITS lists without modifying audio; keep duplicate groups together."""
import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--speaker")
    parser.add_argument("--languages", default="zh,en")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    root = args.audio_root.resolve()
    candidates = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}:
            candidates[path.name.casefold()].append(path.resolve())
    data = args.list.read_bytes()
    encoding = None
    for trial in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = data.decode(trial)
            if "\x00" not in text:
                encoding = trial
                break
        except UnicodeError:
            pass
    if encoding is None:
        raise ValueError("Unknown annotation encoding")
    parsed = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ValueError(f"Malformed annotation at line {line_number}")
        old_path, speaker, language, transcript = (value.strip() for value in fields)
        if not transcript:
            raise ValueError(f"Empty transcript at line {line_number}")
        parsed.append((line_number, old_path, speaker, language.lower(), transcript))
    speakers = sorted({row[2] for row in parsed})
    if args.speaker is None and len(speakers) != 1:
        raise ValueError(f"Choose one --speaker from {speakers}")
    speaker = args.speaker or speakers[0]
    languages = set(args.languages.lower().split(","))
    selected = [row for row in parsed if row[2] == speaker and row[3] in languages]
    rows, skipped, seen_audio = [], [], {}
    for line_number, old_path, name, language, transcript in selected:
        basename = old_path.replace("\\", "/").rsplit("/", 1)[-1]
        matches = candidates[basename.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Line {line_number}: audio not uniquely resolved: {basename}")
        path = matches[0]
        before = path.stat()
        waveform, rate = sf.read(path, dtype="float32", always_2d=True)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"Empty/non-finite audio: {path}")
        mono = waveform.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        if rms < 1e-6:
            raise ValueError(f"Silent audio: {path}")
        audio_hash = hashlib.sha256(int(rate).to_bytes(8, "little") + mono.tobytes()).hexdigest()
        text_key = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", transcript), flags=re.UNICODE).casefold()
        if audio_hash in seen_audio:
            prior = seen_audio[audio_hash]
            if prior["text_key"] != text_key:
                raise ValueError(f"Conflicting transcripts for identical audio: {prior['audio']} / {path}")
            skipped.append(dict(line=line_number, reason="duplicate_audio", audio=str(path), kept=prior["audio"]))
            continue
        if (path.stat().st_size, path.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise RuntimeError(f"Audio changed during inspection: {path}")
        row = dict(audio=str(path), text=transcript, speaker=name, language=language, line=line_number, original_path=old_path, seconds=len(mono)/rate, sample_rate=rate, channels=waveform.shape[1], audio_pcm_sha256=audio_hash, file_sha256=digest(path), text_key=text_key, rms=rms, peak=float(np.max(np.abs(mono))), clipping_fraction=float(np.mean(np.abs(mono) >= 0.99999)))
        rows.append(row)
        seen_audio[audio_hash] = row
    if len(rows) < 2:
        raise ValueError("Need at least two usable examples")
    # Same normalized transcript stays in one split, reducing phrase leakage.
    groups = defaultdict(list)
    for row in rows:
        groups[row["text_key"]].append(row)
    if len(groups) < 2:
        raise ValueError("Need different transcripts for held-out validation")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{args.seed}\0{key}".encode()).digest())
    target = max(1, round(len(rows) * args.validation_fraction))
    validation_keys, count = set(), 0
    for key in ordered[:-1]:
        if count >= target:
            break
        validation_keys.add(key)
        count += len(groups[key])
    train = [row for row in rows if row["text_key"] not in validation_keys]
    validation = [row for row in rows if row["text_key"] in validation_keys]
    if not train or not validation:
        raise ValueError("Empty training or validation split")
    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)
    for name, values in (("all", rows), ("train", train), ("validation", validation)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps({key: row[key] for key in ("audio", "text")}, ensure_ascii=False) + "\n" for row in values))
    (out / "source-map.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    durations = sorted(row["seconds"] for row in rows)
    report = dict(status="converted_and_structurally_checked", source_list=str(args.list.resolve()), source_list_sha256=hashlib.sha256(data).hexdigest(), source_audio_root=str(root), source_encoding=encoding, source_rows=len(parsed), selected_rows=len(selected), usable_rows=len(rows), selected_speaker=speaker, languages=dict(Counter(row["language"] for row in rows)), train_rows=len(train), validation_rows=len(validation), train_seconds=sum(row["seconds"] for row in train), validation_seconds=sum(row["seconds"] for row in validation), duration_seconds=dict(min=durations[0], median=statistics.median(durations), p95=durations[int((len(durations)-1)*0.95)], max=durations[-1], total=sum(durations)), sample_rates=dict(Counter(row["sample_rate"] for row in rows)), channels=dict(Counter(row["channels"] for row in rows)), duplicate_audio_removed=skipped, repeated_transcript_groups=sum(len(group)>1 for group in groups.values()), clipping_warning_files=[row["audio"] for row in rows if row["clipping_fraction"] > 0.01], validation_fraction=args.validation_fraction, seed=args.seed, split_policy="group_by_normalized_transcript_after_pcm_deduplication", source_audio_modified=False, transcript_accuracy_manually_verified=False)
    (out / "conversion-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
