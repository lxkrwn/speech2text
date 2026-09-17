#!/usr/bin/env python3
"""Local Russian transcription with GigaAM-v3 and pyannote.audio."""

from __future__ import annotations

import os
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

# Automator may not load the shell profile, so include common Homebrew paths.
for extra_path in ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"):
    if extra_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = extra_path + os.pathsep + os.environ.get("PATH", "")

# ---------- SETTINGS (see config.py) ----------

from config import (  # noqa: E402
    ARCHIVE_DIR,
    ASR_MODEL,
    COMMUNITY_MODEL,
    DIARIZATION_MODEL,
    INPUT_DIR,
    MIN_FLICKER_DURATION,
    MIN_FLICKER_WORDS,
    NUM_SPEAKERS,
    OUTPUT_DIR,
    OUTPUT_FORMAT,
    PROJECT_DIR,
    SEGMENTATION_MODEL,
    SPEAKER_NAMES,
    SUPPORTED_EXTENSIONS,
    TERMINOLOGY_FILE,
)

Replacement = tuple[str, str, str]
Flag = tuple[str, str]

ENV_FILE = PROJECT_DIR / ".env"
LOG_FILE = PROJECT_DIR / "log_tech.txt"
PROCESSING_LOG_FILE = PROJECT_DIR / "log_user.txt"
CHECKPOINT_DIR = PROJECT_DIR / ".checkpoints"


def load_env() -> None:
    """Load simple KEY=VALUE pairs from the project .env file."""
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ[key.strip()] = value


def find_input_files() -> list[Path]:
    if not INPUT_DIR.exists():
        return []
    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def convert_to_wav(source: Path) -> Path:
    wav_path = source.with_name(f"_{source.stem}_tmp16k.wav")
    command = [
        "ffmpeg", "-y", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg не найден. Установите его через: brew install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.decode(errors="replace").strip().splitlines()
        detail = details[-1] if details else "неизвестная ошибка ffmpeg"
        raise RuntimeError(f"ffmpeg не смог обработать файл: {detail}") from exc
    return wav_path


def run_asr(wav_path: str, model_name: str):
    """Run GigaAM in a worker process and return word-level timestamps."""
    import torch
    import gigaam

    with open(os.devnull, "w") as quiet_stream:
        with redirect_stdout(quiet_stream), redirect_stderr(quiet_stream):
            model = gigaam.load_model(model_name)
            if torch.backends.mps.is_available():
                model = model.to("mps")
            result = model.transcribe_longform(wav_path, word_timestamps=True)
    words = []
    for segment in result.segments:
        if not segment.words:
            words.append((float(segment.start), float(segment.end), segment.text))
            continue
        for word in segment.words:
            words.append((float(word.start), float(word.end), word.text))
    return words


def run_diarization(wav_path: str, token: str, model_name: str, num_speakers: int | None):
    """Run pyannote in a worker process and return speaker intervals."""
    import torch
    from pyannote.audio import Pipeline

    with open(os.devnull, "w") as quiet_stream:
        with redirect_stdout(quiet_stream), redirect_stderr(quiet_stream):
            pipeline = Pipeline.from_pretrained(model_name, token=token)
            if torch.backends.mps.is_available():
                try:
                    pipeline.to(torch.device("mps"))
                except Exception:
                    pass
            options = {"num_speakers": num_speakers} if num_speakers else {}
            diarization = pipeline(wav_path, **options)
    annotation = getattr(diarization, "speaker_diarization", diarization)
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def check_diarization_access(token: str) -> None:
    """Download required gated files before starting parallel worker processes."""
    from huggingface_hub import hf_hub_download

    required_files = (
        (DIARIZATION_MODEL, "config.yaml"),
        (SEGMENTATION_MODEL, "config.yaml"),
        (SEGMENTATION_MODEL, "pytorch_model.bin"),
        (COMMUNITY_MODEL, "plda/xvec_transform.npz"),
    )
    for model_name, filename in required_files:
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                token=token,
            )
        except Exception as error:
            raise RuntimeError(
                "Нет доступа к gated-модели pyannote. Откройте "
                f"https://huggingface.co/{model_name}, запросите/подтвердите "
                "доступ под тем же аккаунтом и повторите запуск."
            ) from error


def assign_speaker(start: float, end: float, turns) -> str:
    best_speaker = "Неизвестно"
    best_overlap = 0.0
    for turn_start, turn_end, speaker in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def merge_results(words, turns) -> list[dict]:
    raw_words = []
    for start, end, text in words:
        text = text.strip()
        if text:
            raw_words.append({
                "speaker": assign_speaker(start, end, turns),
                "start": start,
                "end": end,
                "text": text,
            })

    if not raw_words:
        return []

    runs = []
    for word in raw_words:
        if runs and runs[-1]["speaker"] == word["speaker"]:
            runs[-1]["end"] = word["end"]
            runs[-1]["text"] += " " + word["text"]
            runs[-1]["word_count"] += 1
        else:
            runs.append({**word, "word_count": 1})

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for index, run in enumerate(runs):
            duration = run["end"] - run["start"]
            if duration >= MIN_FLICKER_DURATION or run["word_count"] > MIN_FLICKER_WORDS:
                continue
            if index > 0:
                neighbor = runs[index - 1]
                neighbor["end"] = run["end"]
                neighbor["text"] += " " + run["text"]
                neighbor["word_count"] += run["word_count"]
            else:
                neighbor = runs[index + 1]
                neighbor["start"] = run["start"]
                neighbor["text"] = run["text"] + " " + neighbor["text"]
                neighbor["word_count"] += run["word_count"]
            runs.pop(index)
            changed = True
            break

    merged_runs = []
    for run in runs:
        if merged_runs and merged_runs[-1]["speaker"] == run["speaker"]:
            merged_runs[-1]["end"] = run["end"]
            merged_runs[-1]["text"] += " " + run["text"]
        else:
            merged_runs.append(dict(run))

    speaker_map = {}
    result = []
    for run in merged_runs:
        raw_speaker = run["speaker"]
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"Спикер {len(speaker_map) + 1}"
        result.append({
            "speaker": speaker_map[raw_speaker],
            "start": run["start"],
            "end": run["end"],
            "text": run["text"].strip(),
        })
    return result


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_audio_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def meeting_metadata(source: Path) -> tuple[str, str]:
    stem = source.stem
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", stem)
    date = (
        date_match.group(1)
        if date_match
        else datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d")
    )
    title = re.sub(r"\s*\d{4}-\d{2}-\d{2}\s*", " ", stem)
    title = re.sub(r"^[\s._-]+|[\s._-]+$", "", title).strip()
    return title or stem, date


def checkpoint_path(source: Path) -> Path:
    return CHECKPOINT_DIR / f"{source.name}.json"


def load_checkpoint(source: Path) -> dict:
    path = checkpoint_path(source)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return {}


def save_checkpoint(source: Path, checkpoint: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path(source).write_text(
        json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8"
    )


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_same_file(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and file_digest(a) == file_digest(b)


def archive_source(source: Path) -> tuple[Path, bool]:
    """Move source into the archive; returns (archive_path, was_duplicate).

    A file already in the archive with the same name, size and hash is
    treated as the same recording: the source is removed instead of being
    duplicated under a numbered name.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = ARCHIVE_DIR / source.name
    if candidate.exists():
        if is_same_file(source, candidate):
            source.unlink()
            return candidate, True
        index = 2
        while True:
            candidate = ARCHIVE_DIR / f"{source.stem} ({index}){source.suffix}"
            if not candidate.exists():
                break
            index += 1
    shutil.move(str(source), str(candidate))
    return candidate, False


def _as_pattern(raw: str) -> str:
    """Turn a dictionary entry into a regex: literal words get \\b, ~ means raw regex."""
    raw = raw.strip()
    if raw.startswith("~"):
        return raw[1:].strip()
    return r"\b" + re.escape(raw) + r"\b"


def parse_terminology(path: Path) -> tuple[list[Replacement], list[Flag]]:
    """Load manual replacement/highlight rules from a plain-text dictionary file."""
    replacements: list[Replacement] = []
    flags: list[Flag] = []
    if not path.exists():
        return replacements, flags

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            pattern_part, _, reason = line[1:].partition("|")
            flags.append((_as_pattern(pattern_part), reason.strip()))
            continue
        rule, _, description = line.partition("|")
        pattern_part, _, replacement = rule.partition("->")
        replacements.append((
            _as_pattern(pattern_part),
            replacement.strip(),
            description.strip() or replacement.strip(),
        ))
    return replacements, flags


def clean_text(text: str, replacements: list[Replacement]) -> tuple[str, list[tuple[str, int]]]:
    statistics = []
    for pattern, replacement, description in replacements:
        expression = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        count = len(expression.findall(text))
        if count:
            text = expression.sub(replacement, text)
            statistics.append((description, count))
    return text, statistics


def find_flags(text: str, flags: list[Flag]) -> list[str]:
    hits = []
    for pattern, reason in flags:
        expression = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if expression.search(line):
                snippet = line.strip()
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                hits.append(f"стр.{line_number}: [{reason}] -> {snippet}")
    return hits


def clean_file(
    source: Path, output: Path, replacements: list[Replacement], flags: list[Flag]
) -> tuple[Path, int, int]:
    text = source.read_text(encoding="utf-8")
    cleaned, statistics = clean_text(text, replacements)
    found_flags = find_flags(cleaned, flags)
    output.write_text(cleaned, encoding="utf-8")
    return output, sum(count for _, count in statistics), len(found_flags)


def write_transcript_markdown(
    segments: list[dict], output_path: Path, source: Path, duration: float
) -> None:
    title, date = meeting_metadata(source)
    speakers = list(dict.fromkeys(segment["speaker"] for segment in segments))
    parts = [
        f"# {title}",
        "",
        f"- **Дата:** {date}",
        f"- **Длительность:** {format_time(duration)}",
        "",
        "## Спикеры",
        *[
            f"- **{speaker}** — {SPEAKER_NAMES[speaker]}"
            if SPEAKER_NAMES.get(speaker)
            else f"- **{speaker}**"
            for speaker in speakers
        ],
        "",
        "## Транскрипт",
    ]
    lines = [
        f"{format_time(segment['start'])}-{format_time(segment['end'])} "
        f"{segment['speaker']}: {segment['text']}"
        for segment in segments
    ]
    content = "\n".join(parts)
    if lines:
        content += "\n\n" + "\n\n".join(lines)
    output_path.write_text(content.rstrip("\n") + "\n", encoding="utf-8")


def write_transcript_text(
    segments: list[dict], output_path: Path, source: Path, duration: float
) -> None:
    title, date = meeting_metadata(source)
    speakers = list(dict.fromkeys(segment["speaker"] for segment in segments))
    legend = [
        f"Название встречи: {title}",
        f"Дата: {date}",
        f"Длительность: {format_time(duration)}",
        "",
        "Легенда спикеров:",
        *[
            f"{speaker} - {SPEAKER_NAMES[speaker]}"
            if SPEAKER_NAMES.get(speaker)
            else f"{speaker} -"
            for speaker in speakers
        ],
    ]
    lines = [
        f"[{format_time(segment['start'])}] **{segment['speaker']}:** {segment['text']}"
        for segment in segments
    ]
    content = "\n".join(legend)
    if lines:
        content += "\n\n" + "\n\n".join(lines)
    output_path.write_text(content + "\n", encoding="utf-8")


def write_transcript(
    segments: list[dict], output_path: Path, source: Path, duration: float
) -> None:
    if output_path.suffix.lower() == ".md":
        write_transcript_markdown(segments, output_path, source, duration)
    else:
        write_transcript_text(segments, output_path, source, duration)


def log(message: str, console: bool = False) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as stream:
        stream.write(f"[{timestamp}] {message}\n")
    if console:
        print(message, flush=True)


def log_processing_record(
    source: Path, started_at: datetime, finished_at: datetime, elapsed: float
) -> None:
    PROCESSING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSING_LOG_FILE.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{started_at.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{finished_at.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{format_time(elapsed)} | "
            f"{source.name}\n"
        )


def process_file(
    source: Path,
    token: str,
    index: int,
    total: int,
    replacements: list[Replacement],
    flags: list[Flag],
) -> None:
    started = time.time()
    started_at = datetime.now()
    wav_path: Path | None = None
    print(f"[{index}/{total}] {source.name}", flush=True)
    print(started_at.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    log(f"Начало обработки: {source.name}")
    try:
        log("Конвертация в WAV 16 кГц mono...")
        wav_path = convert_to_wav(source)
        checkpoint = load_checkpoint(source)
        words = checkpoint.get("words")
        turns = checkpoint.get("turns")
        log("Запуск GigaAM и pyannote для недостающих результатов...")
        with ProcessPoolExecutor(max_workers=2) as executor:
            asr_future = (
                executor.submit(run_asr, str(wav_path), ASR_MODEL)
                if words is None else None
            )
            diarization_future = (
                executor.submit(
                    run_diarization, str(wav_path), token, DIARIZATION_MODEL, NUM_SPEAKERS
                )
                if turns is None else None
            )
            if asr_future is not None:
                words = asr_future.result()
                checkpoint["words"] = words
                save_checkpoint(source, checkpoint)
            if diarization_future is not None:
                turns = diarization_future.result()
                checkpoint["turns"] = turns
                save_checkpoint(source, checkpoint)

        segments = merge_results(words, turns)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        extension = "md" if OUTPUT_FORMAT == "md" else "txt"
        output_path = OUTPUT_DIR / f"{source.stem} {timestamp}.{extension}"
        write_transcript(segments, output_path, source, get_audio_duration(wav_path))
        cleaned_path, replaced_count, flagged_count = clean_file(
            output_path, output_path, replacements, flags
        )
        archive_path, was_duplicate = archive_source(source)
        checkpoint_path(source).unlink(missing_ok=True)
        elapsed = time.time() - started
        finished_at = datetime.now()
        log_processing_record(source, started_at, finished_at, elapsed)
        archive_note = (
            f"уже был в архиве ({archive_path}), исходник удалён"
            if was_duplicate
            else f"исходник перемещён в {archive_path}"
        )
        log(
            f"Готово: {output_path.name}; очищенный текст: {cleaned_path.name}; "
            f"{archive_note}; "
            f"время {elapsed / 60:.1f} мин"
        )
        print(
            f"{finished_at.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{format_time(elapsed)} | Автозамены {replaced_count} | "
            f"Проверки {flagged_count}"
        )
    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink()


def run_clean_command(argv: list[str]) -> int:
    """Manually clean an already-generated transcript: --clean входной.md [выходной.md]"""
    if len(argv) < 1:
        print("Использование: python3 transcribe.py --clean входной.md [выходной.md]")
        return 1
    source = Path(argv[0]).expanduser()
    if not source.exists():
        print(f"Файл не найден: {source}")
        return 1
    output = Path(argv[1]).expanduser() if len(argv) >= 2 else source
    replacements, flags = parse_terminology(TERMINOLOGY_FILE)
    _, replaced_count, flagged_count = clean_file(source, output, replacements, flags)
    print(f"Готово. Автозамен: {replaced_count}, проверок: {flagged_count}")
    print(f"-> {output}")
    return 0


def main() -> int:
    load_env()

    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        return run_clean_command(sys.argv[2:])

    token = os.environ.get("HF_TOKEN")
    if not token:
        log(f"ОШИБКА: не найден HF_TOKEN в {ENV_FILE}", console=True)
        return 1

    try:
        check_diarization_access(token)
    except RuntimeError as error:
        log(f"ОШИБКА: {error}", console=True)
        return 1

    if len(sys.argv) > 1:
        requested = Path(sys.argv[1]).expanduser().resolve()
        files = [requested] if requested.is_file() else []
    else:
        files = find_input_files()

    if not files:
        log(f"Не найдено поддерживаемых медиафайлов в {INPUT_DIR}", console=True)
        return 0

    replacements, flags = parse_terminology(TERMINOLOGY_FILE)

    total = len(files)
    for index, source in enumerate(files, start=1):
        try:
            process_file(source, token, index, total, replacements, flags)
        except Exception as error:
            log(
                f"[{index}/{total}] ОШИБКА при обработке {source.name}: {error}",
                console=True,
            )
            log(traceback.format_exc())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
