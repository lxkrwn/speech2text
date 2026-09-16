"""Настройки проекта Speech2Text.

Здесь собраны все параметры, которые раньше были зашиты в transcribe.py.
Базовые (по умолчанию) пути указывают на подпапки этого проекта — "Аудиозаписи"
и "Транскрипты". Чтобы обрабатывать файлы из других мест (например, с личного
Яндекс.Диска), не редактируйте этот файл — создайте рядом config_local.py
(см. config_local.py.example) и укажите пути там. config_local.py не попадает
в git, поэтому личные пути не светятся в репозитории. Если config_local.py
нет или в нём не задана какая-то из переменных — используются базовые папки
проекта, объявленные ниже.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ---------- Базовые папки (по умолчанию — внутри проекта) ----------

DEFAULT_INPUT_DIR = PROJECT_DIR / "Аудиозаписи"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Транскрипты"
ARCHIVE_SUBDIR_NAME = "Архив"

# ---------- Локальные переопределения (личные для этой машины, не в git) ----------
# Реальные значения задаются в config_local.py. Отсутствующий файл или
# отсутствующая в нём переменная равнозначны None — используется папка проекта.

try:
    import config_local as _local
except ImportError:
    _local = None


def _local_override(name: str) -> Path | None:
    return getattr(_local, name, None) if _local else None


OVERRIDE_INPUT_DIR: Path | None = _local_override("OVERRIDE_INPUT_DIR")
OVERRIDE_OUTPUT_DIR: Path | None = _local_override("OVERRIDE_OUTPUT_DIR")

INPUT_DIR = OVERRIDE_INPUT_DIR or DEFAULT_INPUT_DIR
OUTPUT_DIR = OVERRIDE_OUTPUT_DIR or DEFAULT_OUTPUT_DIR
ARCHIVE_DIR = INPUT_DIR / ARCHIVE_SUBDIR_NAME

# ---------- Формат выходного файла ----------
# "md" — Markdown (по умолчанию), "txt" — обычный текст.

OUTPUT_FORMAT = "md"

# ---------- Словарь автозамены терминов ----------
# Обычный текстовый файл, который правится вручную (см. terminology.txt) —
# никакого кода трогать не нужно. Формат описан в комментариях самого файла.

DEFAULT_TERMINOLOGY_FILE = PROJECT_DIR / "terminology.txt"
OVERRIDE_TERMINOLOGY_FILE: Path | None = _local_override("OVERRIDE_TERMINOLOGY_FILE")
TERMINOLOGY_FILE = OVERRIDE_TERMINOLOGY_FILE or DEFAULT_TERMINOLOGY_FILE

# ---------- Модели ASR и диаризации ----------

ASR_MODEL = "v3_e2e_rnnt"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
NUM_SPEAKERS: int | None = None
MIN_FLICKER_DURATION = 1.0
MIN_FLICKER_WORDS = 2

# Имена спикеров по номеру (если известны). Оставьте пустым, если имена неизвестны.
SPEAKER_NAMES: dict[str, str] = {
    # "Спикер 1": "Фамилия Имя",
}

# ---------- Поддерживаемые форматы файлов ----------

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
