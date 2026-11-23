"""Внутренние константы Speech2Text.

Здесь лежит то, что скрипту нужно для работы, но обычно не требует правки.
Настройки, которые меняет пользователь (папки, модели, пороги), находятся
в config.py.
"""

from __future__ import annotations

from config import PROJECT_DIR

# ---------- Служебные файлы и папки проекта ----------

ENV_FILE = PROJECT_DIR / ".env"
LOG_FILE = PROJECT_DIR / "log_tech.txt"
PROCESSING_LOG_FILE = PROJECT_DIR / "log_user.txt"
CHECKPOINT_DIR = PROJECT_DIR / ".checkpoints"
# Временные WAV после ffmpeg: _{stem}_tmp16k.wav — не брать в очередь входа.
TMP_WAV_SUFFIX = "_tmp16k.wav"

# ---------- Окружение ----------

# Automator может не загружать профиль оболочки, поэтому добавляем
# типичные папки Homebrew в PATH (там лежит ffmpeg).
EXTRA_PATH_DIRS = ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin")

# ---------- Метки спикеров ----------

# Метка слов, которым диаризация не назначила спикера.
UNKNOWN_SPEAKER = "Неизвестно"
# Префикс итоговых имён спикеров: "Спикер 1", "Спикер 2" и так далее.
SPEAKER_LABEL_PREFIX = "Спикер"
