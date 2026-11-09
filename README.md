# Hackathon ML Research

Исследовательский учебный проект о связи характеристик хакатонных проектов с результатами. Завершены аудит данных, EDA, ретроспективный metadata baseline и его независимый review; объективное оценивание проектов не является целью.

**Статус:** Data Audit, EDA, Technology ML Baseline и Independent Review завершены.

## Исследовательская гипотеза

Проверить, добавляют ли структурные характеристики репозитория информацию к простым метаданным проекта при анализе связи с победой на хакатоне. Это наблюдательное исследование: оно не определяет «объективного победителя» и не устанавливает причинность.

## Источник и обработка

Используется [twangodev/devpost-hacks на Hugging Face](https://huggingface.co/datasets/twangodev/devpost-hacks), закреплённая revision `3cb29f15841c7b61fd78a20ff12127e0846dc227` (последнее изменение по HF API: 2026-04-28). Скрипт получает JSON-строки с HF Dataset Viewer API и сохраняет локальную копию в `data/raw/`.

Сырые записи и встроенные README не публикуются: у исходных материалов нет единой лицензии. В репозитории остаются только код аудита, агрегированная таблица и отчёт.

## Результаты Data Audit

- В закреплённой версии: **2 222 строки**, **2 222 уникальных `project_id`**, **9 хакатонов**; точных дубликатов строк не найдено.
- В карточке источника указано 363 победителя. Аудит исключает 61 запись HackTech 2026: результаты не были опубликованы на момент сбора. После исключения остаются **363 winner и 1 798 non-winner** в восьми событиях с обоими классами.
- В audit snapshot обнаружены README у **1 354 проектов**; карточка датасета указывает 1 358. Причина расхождения в четыре строки неизвестна.
- Четыре ограниченные попытки сверить метки на страницах Devpost получили HTTP 403. Независимая проверка разметки не подтверждена.
- Метки и тексты не имеют достаточных временных отметок для доказательства, что все возможные признаки существовали до объявления результатов.

Подробные распределения классов, GitHub-ссылок, README, пропусков и leakage-индикаторов приведены в [`data_audit_report.md`](data_audit_report.md) и [`data_audit_summary.csv`](data_audit_summary.csv).

## Воспроизведение

Требуется Python 3.10+; используются только модули стандартной библиотеки, дополнительные зависимости не нужны.

```bash
python data_audit.py
```

При первом запуске скрипт загружает все строки закреплённой revision в `data/raw/`, затем использует локальный кэш. Он также делает не более четырёх однократных запросов к Devpost для ограниченной проверки меток; при повторном запуске результаты проверки берутся из локального кэша. Запросы не обходят ограничения доступа.

Чтобы перескачать закреплённые строки:

```bash
python data_audit.py --refresh
```

Чтобы пропустить запросы к Devpost:

```bash
python data_audit.py --no-devpost-check
```

Скрипт формирует `data_audit_report.md` и `data_audit_summary.csv`. Локальный каталог `data/` исключён из Git.

## EDA и Technology ML Baseline

**EDA — completed.** Отчёт: [`data_quality_eda_report.md`](data_quality_eda_report.md). В нём есть графики [`eda_event_winner_share.svg`](eda_event_winner_share.svg), [`eda_top_technologies.svg`](eda_top_technologies.svg), [`eda_text_lengths.svg`](eda_text_lengths.svg), [`eda_link_coverage.svg`](eda_link_coverage.svg), а также агрегаты и feature inventory. Исходные тексты, URL-проверки и построчная аналитическая таблица остаются в игнорируемом `data/`.

**Technology ML Baseline — completed.** Сравнение `DummyClassifier(strategy="prior")` и `LogisticRegression` по восьми leave-one-hackathon-out folds: **2 161 проект**, **macro AP 0,243 против 0,178**; у LogisticRegression **recall 0,047 при пороге 0,5**. Детали: [`technology_baseline_report.md`](technology_baseline_report.md), [результаты по folds](technology_baseline_results.csv), [коэффициенты](technology_baseline_coefficients.csv), [сводка технологий](technology_tag_summary.csv).

**Independent Review — completed.** Read-only аудит и план самостоятельной NumPy-реализации: [`baseline_review.md`](baseline_review.md). Подтверждённых ошибок текущих метрик не найдено; pooled AP отмечен как описательная сводка из-за объединения scores отдельных fold-моделей.

## Воспроизведение EDA и baseline

При отсутствии локального кэша получить закреплённый snapshot и пропустить Devpost-запросы:

```bash
python data_audit.py --no-devpost-check
```

EDA использует Python 3.10+ и стандартную библиотеку; эта команда создаёт локальную аналитическую таблицу для baseline и публикуемые агрегаты. Чтобы не отправлять HTTP-запросы при повторе:

```bash
python data_quality_eda.py --skip-link-checks
```

Baseline требует Python 3.12 и версии NumPy, pandas и scikit-learn из [`requirements-ml.txt`](requirements-ml.txt):

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\python.exe -m pip install -r requirements-ml.txt
.venv\\Scripts\\python.exe technology_baseline.py
# macOS/Linux: используйте .venv/bin/python вместо .venv\\Scripts\\python.exe
```

Исходный датасет, кэши и индивидуальные строки не публикуются; каталог `data/` исключён из Git.


## Итог и ограничения

Это ретроспективное исследование snapshot, а не подтверждённый прогноз победителей до судейства. Историческая доступность `built_with` не подтверждена; независимая проверка меток Devpost не состоялась из-за HTTP 403. В оценке только **8 событий**, поэтому результат нельзя обобщать на новые хакатоны. AP выше у LogisticRegression во всех восьми folds, но recall при фиксированном пороге низкий; причинные выводы не делаются.
