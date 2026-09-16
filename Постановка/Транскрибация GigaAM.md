Локальная транскрибация · macOS · Apple Silicon
# GigaAM​-v3 + pyannote

Расшифровка русскоязычных встреч с разделением по спикерам. Всё считается на своём Mac — записи никуда не уходят. Запуск в один клик, готовый текст через несколько минут.

**1–8 мин** на встречу Mac **M1**, 16 ГБ Python **3.11** Установка **~40 мин**

Как выглядит результат

00:03:12

Спикер 1: давайте начнём с маршрутов согласования в ЭДО, у нас там две ветки разошлись после апреля

00:03:29

Спикер 2: одна через 1С, вторая ручная. Вторую мы как раз и хотим закрыть в этом этапе

00:03:47

Спикер 1: тогда зафиксируем это как требование и вернёмся к матрице согласования

1. [](#stack)
    
[01Стек и ограничения](#stack)
1. [02Требования](#req)
2. [03Базовое окружение](#env)
3. [04Установка моделей](#models)
4. [05Токен Hugging Face](#token)
5. [06Скрипт транскрибации](#script)
6. [07Корректировка фраз](#clean)
7. [08Запуск в один клик](#launch)
8. [09Типовые ошибки](#errors)
9. [10Порядок работы](#daily)

## 

01Стек и ограничения

Три компонента. Два тяжёлых шага — распознавание и диаризация — запускаются **параллельно, в двух процессах**, поэтому общее время равно длительности более медленного из них, а не их сумме.

Распознавание речи

**GigaAM-v3 (RNNT), Сбер**

Лучшая на сегодня точность именно для русского: заметно точнее Whisper large-v3 на деловой речи, аббревиатурах и именах.

Диаризация — кто и когда говорил

**pyannote.audio (speaker-diarization-3.1)**

Размечает реплики по голосам даже в одноканальной записи из Zoom, Telemost или диктофона.

Подготовка звука

**ffmpeg**

Приводит любой аудио- или видеоформат к 16 кГц моно. Из видео звук извлекается автоматически.

### Что стек не умеет

- **Имена не проставляет.** На выходе «Спикер 1 / Спикер 2» — имена ставятся вручную поиском-заменой после первого прослушивания, это пара минут.
- **Переговорка на один микрофон разваливает диаризацию.** Если участники сидят в одной комнате перед одним микрофоном или постоянно перебивают друг друга, разделение по голосам будет условным. Лечится только раздельной записью участников.
- **Терминологию всё равно перевирает.** Узкие аббревиатуры искажаются у любой модели — под это есть отдельный шаг [07](#clean) со словарём автозамены.

**Реальная скорость** на MacBook M1 / 16 ГБ, встречи 30–90 минут: от 1 до 8 минут на запись. Часовая встреча — порядка 6–8 минут.

## 

02Требования

- Mac на Apple Silicon (M1/M2/M3/M4), от 16 ГБ оперативной памяти. Вычисления идут через Metal (MPS), часть операций падает на процессор — это нормально.
- Около 10 ГБ свободного места: модели весят прилично.
- Бесплатный аккаунт на [huggingface.co](https://huggingface.co) — нужен для скачивания модели диаризации.

На Intel-Mac и Windows стек тоже поднимается, но без MPS-ускорения работает в разы медленнее. Здесь описан вариант для Apple Silicon.

## 

03Базовое окружение

Откройте **Терминал** (Программы → Утилиты → Терминал) и выполняйте команды по одной.

### Homebrew

Менеджер пакетов для macOS. Если уже установлен — пропустите.

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

### ffmpeg и Python 3.11

brew install ffmpeg python@3.11

Нужен именно Python **3.11**. На 3.12 и выше часть зависимостей GigaAM и pyannote может не собраться.

### Рабочая папка и изолированное окружение

Всё живёт в одной папке — так проще переносить и удалять. Ниже примером взята ~/Documents/Transcription; назовите как удобно, но дальше по тексту подставляйте свой путь.

mkdir -p ~/Documents/Transcription

cd ~/Documents/Transcription

/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv gigaam-env

source gigaam-env/bin/activate

После активации в начале строки терминала появится (gigaam-env). Все следующие команды выполняются с активированным окружением.

## 

04Установка моделей

pip install --upgrade pip

pip install torch torchaudio

pip install git+https://github.com/salute-developers/GigaAM.git

pip install pyannote.audio

Проверка, что PyTorch видит графическое ядро Mac:

python -c "import torch; print('MPS доступен:', torch.backends.mps.is_available())"

Должно вывести MPS доступен: True. Если False — считать будет на процессоре, в несколько раз медленнее.

### Проверенная связка версий

На этих версиях стек отработал на реальных встречах. Если свежие версии сломаются — откатывайтесь на них.

Python

3.11.15

torch

2.12.1

torchaudio

2.11.0

gigaam

0.1.0

pyannote.audio

4.0.5

numpy

2.4.6

## 

05Токен Hugging Face

Модель диаризации закрытая (gated): без принятия условий она просто не скачается.

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co).
2. Откройте [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) и нажмите **Agree and access repository**.
3. То же самое для [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) — первая модель тянет вторую, и если условия не приняты, запуск падает с ошибкой 401.
4. Создайте токен: **Settings → Access Tokens → New token**, тип **Read**. Скопируйте строку вида hf_xxxxxxxx.
5. Положите токен в файл .env в рабочей папке:

echo "HF_TOKEN=hf_ВАШ_ТОКЕН" > ~/Documents/Transcription/.env

Токен — это пароль от вашего аккаунта. Файл .env не выкладывайте в общие папки и репозитории.

## 

06Скрипт транскрибации

Положите файл transcribe.py в рабочую папку рядом с .env. Что он делает по шагам:

1. Находит в своей папке все поддерживаемые медиафайлы.
2. Через ffmpeg приводит звук к 16 кГц моно во временный файл _tmp16k.wav, который удаляется в конце.
3. Параллельно запускает GigaAM (текст со словными таймкодами) и pyannote (интервалы спикеров).
4. Накладывает слова на интервалы спикеров и склеивает их в связные реплики.
5. Кладёт рядом с исходником <имя>_<дата_время>.txt, пишет transcribe_log.txt и показывает уведомление macOS.

**Форматы:** аудио — wav, mp3, m4a, aac, flac, ogg, opus, wma, aiff; видео (звук извлекается сам) — mp4, mov, mkv, avi, m4v, webm.

### Первый запуск

Модели скачиваются при первом старте — это несколько минут и несколько гигабайт.

cd ~/Documents/Transcription

source gigaam-env/bin/activate

python3 transcribe.py

### Настройки внутри скрипта

Блок «НАСТРОЙКИ» в начале файла — единственное, что нужно трогать.

|Параметр|По умолчанию|Когда менять|
|---|---|---|
|INPUT_DIR|папка со скриптом|если файлы должны браться из другой папки|
|ASR_MODEL|v3_e2e_rnnt|поставьте v3_e2e_ctc, если нужна скорость, а не максимум точности|
|NUM_SPEAKERS|None (авто)|поставьте число, если точно знаете количество участников — часто заметно улучшает разбивку|
|min_flicker_duration|1.0 сек|увеличьте до 2.0, если текст рвётся на куски из-за коротких «угу»|

Полный исходник transcribe.py

#!/usr/bin/env python3

"""

transcribe.py — транскрибация записей встреч на русском языке.

GigaAM-v3 (RNNT)  — распознавание речи

pyannote.audio    — диаризация (кто и когда говорил)

Оба шага идут параллельно, результаты объединяются.

Скрипт обрабатывает все поддерживаемые медиафайлы, лежащие в той же папке,

где лежит сам скрипт. Результат кладётся рядом с исходником:

    <имя>_<дата_время>.txt   — текст, разбитый по спикерам

Лог времени обработки — transcribe_log.txt в той же папке.

Токен Hugging Face берётся из файла .env рядом со скриптом:

    HF_TOKEN=hf_xxxxxxxxxxxxxxxxx

"""

import os

import sys

import time

import traceback

import subprocess

from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

from datetime import datetime

# Когда скрипт запущен через ярлык (Automator .app), он не подгружает

# ~/.zprofile, поэтому ffmpeg из Homebrew может быть не виден.

# Добавляем стандартные пути Homebrew в PATH вручную.

for _extra_path in ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"):

    if _extra_path not in os.environ.get("PATH", ""):

        os.environ["PATH"] = _extra_path + os.pathsep + os.environ.get("PATH", "")

# ---------- НАСТРОЙКИ ----------

# Рабочая папка = папка, в которой лежит этот скрипт.

# Если нужна другая — замените строку на: INPUT_DIR = Path("/полный/путь/к/папке")

INPUT_DIR = Path(__file__).resolve().parent

# Переключатель точность/скорость:

#   "v3_e2e_rnnt" — точнее, медленнее (по умолчанию)

#   "v3_e2e_ctc"  — быстрее, чуть менее точно

ASR_MODEL = "v3_e2e_rnnt"

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

ENV_FILE = INPUT_DIR / ".env"

LOG_FILE = INPUT_DIR / "transcribe_log.txt"

# Ограничить число спикеров (иногда сильно улучшает результат).

# None = определять автоматически. Пример: NUM_SPEAKERS = 4

NUM_SPEAKERS = None

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff"}

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

SUPPORTED_EXT = AUDIO_EXT | VIDEO_EXT

def load_env():

    """Подгружает HF_TOKEN из файла .env в рабочей папке."""

    if ENV_FILE.exists():

        for line in ENV_FILE.read_text().splitlines():

            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:

                continue

            key, value = line.split("=", 1)

            os.environ[key.strip()] = value.strip()

def find_input_files():

    """Все поддерживаемые медиафайлы в рабочей папке (без подпапок).

    Файлы *_tmp16k.wav — наши временные, их пропускаем.

    """

    files = [

        p for p in INPUT_DIR.iterdir()

        if p.is_file()

        and p.suffix.lower() in SUPPORTED_EXT

        and not p.stem.endswith("_tmp16k")

    ]

    return sorted(files)

def to_wav(src_path: Path) -> Path:

    """Конвертирует исходник во временный wav 16 кГц моно."""

    wav_path = src_path.parent / f"{src_path.stem}_tmp16k.wav"

    cmd = [

        "ffmpeg", "-y", "-i", str(src_path),

        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",

        str(wav_path),

    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return wav_path

def run_asr(wav_path_str: str, model_name: str):

    """Отдельный процесс: распознавание речи GigaAM."""

    import torch

    import gigaam

    model = gigaam.load_model(model_name)

    if torch.backends.mps.is_available():

        model = model.to("mps")

    result = model.transcribe_longform(wav_path_str, word_timestamps=True)

    words = []

    for seg in result.segments:

        if not seg.words:

            words.append((float(seg.start), float(seg.end), seg.text))

            continue

        for w in seg.words:

            words.append((float(w.start), float(w.end), w.text))

    return words

def run_diarization(wav_path_str: str, hf_token: str, model_name: str, num_speakers=None):

    """Отдельный процесс: диаризация pyannote."""

    import torch

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)

    if torch.backends.mps.is_available():

        try:

            pipeline.to(torch.device("mps"))

        except Exception:

            pass  # если перенос не поддержан — остаёмся на CPU

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}

    diarize_output = pipeline(wav_path_str, **kwargs)

    # pyannote.audio 4.x возвращает объект с полем .speaker_diarization,

    # версии 3.x — сразу Annotation. Поддерживаем оба варианта.

    annotation = getattr(diarize_output, "speaker_diarization", diarize_output)

    turns = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):

        turns.append((turn.start, turn.end, speaker))

    return turns

def assign_speaker(start, end, turns):

    """Подбирает спикера с максимальным перекрытием по времени."""

    best_speaker, best_overlap = "Неизвестно", 0.0

    for t_start, t_end, speaker in turns:

        overlap = min(end, t_end) - max(start, t_start)

        if overlap > best_overlap:

            best_overlap = overlap

            best_speaker = speaker

    return best_speaker

def merge_results(words, turns, min_flicker_duration=1.0, min_flicker_words=2):

    """Накладывает слова GigaAM на интервалы спикеров pyannote.

    Диаризация на уровне отдельных слов шумит: короткие перебивания

    могут на 1–2 слова «перескочить» на другого спикера. Такие короткие

    вспышки склеиваются с соседней репликой, а не рвут текст.

    """

    raw_words = []

    for start, end, text in words:

        text = text.strip()

        if not text:

            continue

        raw_words.append({

            "speaker": assign_speaker(start, end, turns),

            "start": start, "end": end, "text": text,

        })

    if not raw_words:

        return []

    # Группируем подряд идущие слова одного спикера

    runs = []

    for w in raw_words:

        if runs and runs[-1]["speaker"] == w["speaker"]:

            runs[-1]["end"] = w["end"]

            runs[-1]["text"] += " " + w["text"]

            runs[-1]["word_count"] += 1

        else:

            runs.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"],

                         "text": w["text"], "word_count": 1})

    # Убираем короткие «вспышки»

    changed = True

    while changed and len(runs) > 1:

        changed = False

        for i, run in enumerate(runs):

            duration = run["end"] - run["start"]

            if duration < min_flicker_duration and run["word_count"] <= min_flicker_words:

                if i > 0:

                    prev = runs[i - 1]

                    prev["end"] = run["end"]

                    prev["text"] += " " + run["text"]

                    prev["word_count"] += run["word_count"]

                    runs.pop(i)

                else:

                    nxt = runs[i + 1]

                    nxt["start"] = run["start"]

                    nxt["text"] = run["text"] + " " + nxt["text"]

                    nxt["word_count"] += run["word_count"]

                    runs.pop(i)

                changed = True

                break

    # Соседние забеги одного спикера склеиваем

    merged_runs = []

    for run in runs:

        if merged_runs and merged_runs[-1]["speaker"] == run["speaker"]:

            merged_runs[-1]["end"] = run["end"]

            merged_runs[-1]["text"] += " " + run["text"]

        else:

            merged_runs.append(dict(run))

    # SPEAKER_00 → «Спикер 1» по порядку появления

    speaker_map = {}

    next_id = 1

    merged = []

    for run in merged_runs:

        raw_speaker = run["speaker"]

        if raw_speaker not in speaker_map:

            speaker_map[raw_speaker] = f"Спикер {next_id}"

            next_id += 1

        merged.append({

            "speaker": speaker_map[raw_speaker],

            "start": run["start"],

            "end": run["end"],

            "text": run["text"].strip(),

        })

    return merged

def format_time_hms(seconds: float) -> str:

    total = int(round(seconds))

    hours, rem = divmod(total, 3600)

    minutes, secs = divmod(rem, 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def write_txt(merged, out_path: Path):

    lines = [

        f"[{format_time_hms(seg['start'])}] **{seg['speaker']}:** {seg['text']}"

        for seg in merged

    ]

    out_path.write_text("\n\n".join(lines), encoding="utf-8")

def log(message: str):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as f:

        f.write(f"[{timestamp}] {message}\n")

    print(message, flush=True)

def notify(title: str, message: str):

    """Всплывающее уведомление macOS."""

    def esc(s: str) -> str:

        return s.replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{esc(message)}" with title "{esc(title)}"'

    try:

        subprocess.run(["osascript", "-e", script], check=False)

    except Exception:

        pass

def process_file(src_path: Path, hf_token: str):

    total_start = time.time()

    log(f"Начало обработки: {src_path.name}")

    log("Конвертация в wav 16 кГц моно...")

    wav_path = to_wav(src_path)

    log("Запуск распознавания (GigaAM) и диаризации (pyannote) параллельно...")

    asr_start = time.time()

    with ProcessPoolExecutor(max_workers=2) as executor:

        asr_future = executor.submit(run_asr, str(wav_path), ASR_MODEL)

        diar_future = executor.submit(run_diarization, str(wav_path), hf_token,

                                      DIARIZATION_MODEL, NUM_SPEAKERS)

        utterances = asr_future.result()

        turns = diar_future.result()

    log(f"Распознавание + диаризация завершены за {(time.time() - asr_start) / 60:.1f} мин")

    merged = merge_results(utterances, turns)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    txt_path = src_path.parent / f"{src_path.stem}_{timestamp}.txt"

    write_txt(merged, txt_path)

    if wav_path.exists():

        wav_path.unlink()

    elapsed = time.time() - total_start

    log(f"Готово: {txt_path.name} (общее время обработки: {elapsed / 60:.1f} мин)")

    notify("Транскрибация завершена", f"{src_path.name} — {elapsed / 60:.1f} мин")

def main():

    load_env()

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:

        log("ОШИБКА: не найден HF_TOKEN. Проверь файл .env в рабочей папке "

            f"({ENV_FILE}), там должна быть строка HF_TOKEN=hf_...")

        sys.exit(1)

    files = find_input_files()

    if not files:

        log("Не найдено поддерживаемых медиафайлов в рабочей папке.")

        return

    for src_path in files:

        try:

            process_file(src_path, hf_token)

        except Exception as e:

            log(f"ОШИБКА при обработке {src_path.name}: {e}")

            log(traceback.format_exc())

            notify("Ошибка транскрибации", f"{src_path.name}: {e}")

if __name__ == "__main__":

    main()

07Корректировка фраз

Даже лучшая модель стабильно перевирает узкую терминологию. Живой пример: слово «ЭДО» в одной часовой встрече было распознано шестью способами — _Едо_, _ИДО_, _ВДО_, _ОДО_, _EDO_ и как фраза «в До». Искать это глазами в стостраничном транскрипте бессмысленно — работает словарь автозамены.

python3 clean_transcript.py "Встреча_2026-08-27_11-29.txt"

На выходе два файла: ..._clean.txt с исправленным текстом и ..._clean.log.txt — отчёт о том, что было заменено и что осталось на ручную проверку.

### Отчёт с реальной встречи

АВТОЗАМЕН ВЫПОЛНЕНО: 25

----------------------------------------

     6 ×  ЭДО (вар. «Едо»)

     3 ×  ЭДО (вар. «ИДО»)

     3 ×  ЭДО (вар. «ВДО»)

     3 ×  ЭДО (фраза «в До»)

     2 ×  ЭДО (вар. «ОДО»)

     2 ×  1С (хвост-мусор после «1С»)

     1 ×  NDA (вар. «Inday»)

     1 ×  Excel (вар. «Excelital»)

НА РУЧНУЮ ПРОВЕРКУ: 6 мест

----------------------------------------

  стр.215: [вероятно «ДС» (допсоглашение), а не «НДС» (налог)]

  стр.137: [вероятно «вишлист» (wish-list)]

  стр.207: [возможно «Visio» (схемы процесса)]

### Как устроен словарь

В скрипте два блока. Оба правятся как обычный текст, код трогать не нужно.

**REPLACEMENTS** — заменяем автоматически. Сюда попадает только то, в чём вы уверены на сто процентов:

(r"\bЕдо\b", "ЭДО", "ЭДО (вар. «Едо»)"),

#  ^ шаблон   ^ замена  ^ подпись в отчёте

\b — граница слова, чтобы правило не задело середину другого слова. Регистр не важен: одно правило ловит «едо», «Едо» и «ЕДО».

**FLAGS** — только подсветить, ничего не менять. Для случаев, где решает контекст:

(r"договор\s+НДС", "вероятно «ДС» (допсоглашение), а не «НДС» (налог)"),

**Правило ведения словаря.** После первых двух-трёх встреч по новой теме пробегите транскрипт глазами, выпишите переврáнные термины и заведите их в REPLACEMENTS. Дальше словарь работает сам и накапливается под вашу предметную область — у каждого заказчика она своя. Если замена может оказаться неоднозначной, заводите её в FLAGS: испорченный автозаменой смысл дороже, чем ручная правка шести мест.

Полный исходник clean_transcript.py

#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

clean_transcript.py — постобработка транскриптов (GigaAM + pyannote).

Словарь автозамены чинит термины, которые модель распознаёт неустойчиво

(ЭДО, 1С, NDA, Excel и т.п.), и подсвечивает спорные места для ручной проверки.

Использование:

    python clean_transcript.py входной.txt

    python clean_transcript.py входной.txt вычищенный.txt

На выходе:

    <имя>_clean.txt      — вычищенный текст

    <имя>_clean.log.txt  — лог: какое правило сколько раз сработало

                           + список мест, помеченных на ручную проверку

Как пополнять словарь: ниже два блока — REPLACEMENTS (безопасная автозамена)

и FLAGS (только подсветить, не трогать). Добавляйте строки по образцу.

Менять код не нужно.

"""

import re

import sys

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────

# 1) АВТОЗАМЕНА — только высокая уверенность. Формат строки:

#    (r"шаблон", "на что заменить", "комментарий для лога")

#

#    \b — граница слова (чтобы не задеть середину других слов).

#    Регистр при поиске игнорируется (IGNORECASE), поэтому

#    «едо», «Едо», «ЕДО» ловятся одним правилом.

#

#    Ниже — рабочий пример для встреч по ЭДО/1С. Свои термины добавляйте

#    по итогам первых 2–3 транскриптов: выпишите, что модель переврала,

#    и заведите правило.

# ──────────────────────────────────────────────────────────────────────────

REPLACEMENTS = [

    # --- ЭДО: один термин, который модель пишет 6 способами ---

    (r"\bЕдо\b",             "ЭДО",          "ЭДО (вар. «Едо»)"),

    (r"\bИДО\b",             "ЭДО",          "ЭДО (вар. «ИДО»)"),

    (r"\bВДО\b",             "ЭДО",          "ЭДО (вар. «ВДО»)"),

    (r"\bОДО\b",             "ЭДО",          "ЭДО (вар. «ОДО»)"),

    (r"\bEDO\b",             "ЭДО",          "ЭДО (вар. «EDO»)"),

    # «в До» как отдельная фраза — только в этом виде, чтобы не задеть «до конца»

    (r"\bв\s+До\b",          "в ЭДО",        "ЭДО (фраза «в До»)"),

    # --- 1С ---

    (r"\b1С[а-яa-z]{1,3}\b", "1С",           "1С (хвост-мусор после «1С»)"),

    # --- латиница ---

    (r"\bInday\b",           "NDA",          "NDA (вар. «Inday»)"),

    (r"\bExcelital\b",       "Excel",        "Excel (вар. «Excelital»)"),

    # --- частые искажения обычных слов ---

    (r"\bполичаются\b",      "отличаются",   "слово «отличаются»"),

    (r"\bтравтологию\b",     "тавтологию",   "слово «тавтологию»"),

    (r"\bвозразделение\b",   "подразделение","слово «подразделение»"),

]

# ──────────────────────────────────────────────────────────────────────────

# 2) FLAGS — НЕ заменяем автоматически (зависит от контекста или имя собственное).

#    Скрипт просто покажет в логе, где это встретилось, для ручной правки.

#    Формат: (r"шаблон", "почему помечено")

# ──────────────────────────────────────────────────────────────────────────

FLAGS = [

    (r"\bND\b",              "возможно NDA — проверить по смыслу"),

    (r"договор\s+НДС",       "вероятно «ДС» (допсоглашение), а не «НДС» (налог)"),

    (r"\bпо\s+винс\b",       "возможно «Visio» (схемы процесса)"),

    (r"\bкак-лист\b",        "вероятно «вишлист» (wish-list)"),

]

# ──────────────────────────────────────────────────────────────────────────

# Технический блок ниже менять не нужно

# ──────────────────────────────────────────────────────────────────────────

def clean_text(text):

    """Возвращает (вычищенный_текст, статистика_замен)."""

    stats = []

    for pattern, repl, desc in REPLACEMENTS:

        rx = re.compile(pattern, re.IGNORECASE | re.UNICODE)

        count = len(rx.findall(text))

        if count:

            text = rx.sub(repl, text)

            stats.append((desc, count))

    return text, stats

def find_flags(text):

    """Ищет места, помеченные на ручную проверку."""

    lines = text.splitlines()

    hits = []

    for pattern, why in FLAGS:

        rx = re.compile(pattern, re.IGNORECASE | re.UNICODE)

        for i, line in enumerate(lines, start=1):

            if rx.search(line):

                snippet = line.strip()

                if len(snippet) > 120:

                    snippet = snippet[:117] + "..."

                hits.append(f"  стр.{i}: [{why}] → {snippet}")

    return hits

def main():

    if len(sys.argv) < 2:

        print("Использование: python clean_transcript.py входной.txt [выходной.txt]")

        sys.exit(1)

    src = Path(sys.argv[1])

    if not src.exists():

        print(f"Файл не найден: {src}")

        sys.exit(1)

    text = src.read_text(encoding="utf-8")

    cleaned, stats = clean_text(text)

    flags = find_flags(cleaned)

    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else src.with_name(src.stem + "_clean.txt")

    log = out.with_name(out.stem + ".log.txt")

    out.write_text(cleaned, encoding="utf-8")

    total = sum(c for _, c in stats)

    log_lines = [

        f"Исходный файл: {src.name}",

        f"Вычищенный:    {out.name}",

        "",

        f"АВТОЗАМЕН ВЫПОЛНЕНО: {total}",

        "-" * 40,

    ]

    for desc, c in sorted(stats, key=lambda x: -x[1]):

        log_lines.append(f"  {c:>4} ×  {desc}")

    if not stats:

        log_lines.append("  (ничего не заменено)")

    log_lines += ["", f"НА РУЧНУЮ ПРОВЕРКУ: {len(flags)} мест", "-" * 40]

    log_lines.extend(flags if flags else ["  (нет)"])

    log.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Готово. Автозамен: {total}, помечено на проверку: {len(flags)}")

    print(f"  → {out}")

    print(f"  → {log}")

if __name__ == "__main__":

    main()

08Запуск в один клик

Чтобы не открывать терминал каждый раз, делается ярлык-приложение.

1. Откройте **Automator** (Программы → Automator).
2. Выберите **Новый документ → Программа**.
3. В поиске слева найдите действие **«Запустить shell-скрипт»** и перетащите его вправо.
4. Вставьте команду ниже, поправив путь к папке, если он у вас другой.
5. Сохраните как run.app в рабочую папку.

osascript -e 'tell application "Terminal" to do script "export PATH=\"/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH\"; source ~/Documents/Transcription/gigaam-env/bin/activate; cd ~/Documents/Transcription; python3 transcribe.py"'

**Зачем здесь** **export PATH.** Приложение из Automator не подхватывает профиль оболочки и не видит ffmpeg, установленный через Homebrew. Эта строка чинит проблему; то же самое продублировано внутри transcribe.py на случай запуска другими способами.

### Что должно лежать в рабочей папке

Рабочая папка/ ├── .env ← токен Hugging Face (создаёте сами) ├── transcribe.py ← основной скрипт ├── clean_transcript.py ← постобработка и словарь автозамены ├── gigaam-env/ ← окружение Python (шаг 03) ├── run.app ← ярлык запуска (шаг 08) └── transcribe_log.txt ← лог времени обработки (появляется сам)

09Типовые ошибки

|Симптом|Причина и что делать|
|---|---|
|ОШИБКА: не найден HF_TOKEN|нет файла .env рядом со скриптом или в нём опечатка. Строка должна быть ровно HF_TOKEN=hf_..., без кавычек и пробелов|
|Ошибка 401 / gated repo|не приняты условия на страницах speaker-diarization-3.1 **и** segmentation-3.0 под тем же аккаунтом, чей токен лежит в .env|
|ffmpeg: command not found|не прописан export PATH в команде Automator — см. шаг 08|
|AttributeError на .speaker_diarization|разошлись версии pyannote: в 4.x пайплайн возвращает объект с полем .speaker_diarization, в 3.x — сразу Annotation. В приложенном скрипте поддержаны оба варианта|
|Все реплики свалились в одного спикера|запись в один микрофон из переговорки. Помогает только раздельная запись участников; частично — задать NUM_SPEAKERS вручную|
|Текст рвётся на мелкие куски|увеличьте min_flicker_duration в функции merge_results, например до 2.0|
|Обработка идёт очень долго|проверьте, что MPS доступен: True; при нехватке времени переключите ASR_MODEL на v3_e2e_ctc|
|Не найдено медиафайлов|файл лежит в подпапке (скрипт не ходит вглубь) или его формат не в списке поддерживаемых|

10Порядок работы

После установки повседневный цикл выглядит так:

1. Положить запись встречи в рабочую папку.
2. Двойной клик по run и дождаться уведомления «Транскрибация завершена».
3. Прогнать результат через clean_transcript.py.
4. Проставить имена вместо «Спикер 1 / 2 / 3» поиском-заменой в любом редакторе.
5. Пополнить словарь новыми искажениями, если встретились.

Инструкция собрана по рабочей установке на MacBook M1: версии, тайминги и примеры отчётов взяты из реальных прогонов, а не из документации.