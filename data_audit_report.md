# Data audit: twangodev/devpost-hacks

## Краткий вывод

**Вердикт: GO условно для metadata-only baseline; PIVOT для metadata + repository structure; STOP для metadata + LLM features сейчас.** В источнике 2 222 строки, но 61 строка `hacktech-by-caltech-2026` исключена из labels: dataset card говорит, что результаты не были опубликованы при сборе. Остаются 2 161 известных исходов в 8 событиях, каждое содержит оба класса. Это годится для пилотной group-based оценки, но не для сильных выводов о переносимости. Все четыре запроса Devpost получили HTTP 403; spot-check не подтвердил метки. Модели не обучались.

**Дата аудита:** 2026-11-08T10:55:36+00:00.
**Источник:** [Hugging Face dataset](https://huggingface.co/datasets/twangodev/devpost-hacks)
**Закреплённая revision:** `3cb29f15841c7b61fd78a20ff12127e0846dc227`; Hugging Face API сообщает последнее изменение `2026-04-28T04:55:48Z`.
**Rows API:** `https://datasets-server.huggingface.co/rows?dataset=twangodev/devpost-hacks&config=all&split=train&revision=3cb29f15841c7b61fd78a20ff12127e0846dc227`.
**Source cache:** `data\raw\devpost-hacks-3cb29f15841c7b61fd78a20ff12127e0846dc227.jsonl` — исходные строки JSONL отдельно от кода; файл данных не коммитить.

## Подтверждённые результаты

- Строки: **2222**; уникальные `project_id`: **2222**; точные дубликаты строк: **0**.
- Events: **9**; known labels после исключения unreleased event: `{"non_winner": 1798, "winner": 363, "unknown": 61}`.
- Raw missing `results`: **0**; raw missing `is_winner`: **0**. Полные поля не означают валидный target.
- Повторные группы `project_id`: **0**; повторные Devpost URL: **0**.
- Несовпадений булева `is_winner` с наличием слова `winner` в `results`: **0**; это только грубая consistency-проверка.
- Загружены строки закреплённой revision; GitHub репозитории не скачивались.

## Сводка по hackathon

`winner/non-winner` учитывают только known outcomes. `unknown_result` включает raw missing и результаты, withheld из-за статуса публикации. Доля winner рассчитана среди известных outcomes.

| Хакатон | Всего | Победители | Непобедители | Неизвестно | Доля побед среди известных | Пропуски results | Проекты с GitHub URL | Проекты с README | Уникальные README-репозитории | Уникальные GitHub-репозитории |
|---|---|---|---|---|---|---|---|---|---|---|
| cal-hacks-12-0 | 694 | 100 | 594 | 0 | 14.4% | 0 | 459 | 351 | 360 | 480 |
| hackgt-12 | 272 | 25 | 247 | 0 | 9.2% | 0 | 216 | 170 | 175 | 227 |
| hacktech-by-caltech-2026 | 61 | 0 | 0 | 61 | UNKNOWN | 0 | 51 | 45 | 47 | 53 |
| madhacks | 55 | 10 | 45 | 0 | 18.2% | 0 | 47 | 32 | 32 | 49 |
| madhacks-fall-2025 | 111 | 13 | 98 | 0 | 11.7% | 0 | 93 | 77 | 79 | 99 |
| pennapps-xxv | 98 | 25 | 73 | 0 | 25.5% | 0 | 76 | 63 | 64 | 80 |
| treehacks-2024 | 319 | 58 | 261 | 0 | 18.2% | 0 | 263 | 192 | 201 | 287 |
| treehacks-2025 | 248 | 70 | 178 | 0 | 28.2% | 0 | 230 | 184 | 192 | 250 |
| treehacks-2026 | 364 | 62 | 302 | 0 | 17.0% | 0 | 313 | 240 | 249 | 326 |

**Исход не опубликован при сборе:** `hacktech-by-caltech-2026`, 61 проект. В raw rows `results` заполнен, но карточка датасета указывает, что результаты не были объявлены на Devpost при сборе. Поэтому группа консервативно полностью исключена из целевой переменной. Raw пропусков `results` = 0; withheld по источнику = 61. Это разные случаи.

### Схема и пропуски

| Поле | Наблюдаемые JSON-типы | Null | Пустые строки | Пустые списки |
|---|---|---|---|---|
| built_with | list[str] | 0 | 0 | 0 |
| description | str | 0 | 0 | 0 |
| hackathon | str | 0 | 0 | 0 |
| is_winner | bool | 0 | 0 | 0 |
| other_links | list[str], list[unknown] | 0 | 0 | 263 |
| project_id | str | 0 | 0 | 0 |
| readmes | list[struct{content:str, repo:str, truncated:bool}], list[unknown] | 0 | 0 | 864 |
| results | str | 0 | 0 | 0 |
| tagline | str | 0 | 0 | 0 |
| title | str | 0 | 0 | 0 |
| url | str | 0 | 0 | 0 |
| video_link | str | 478 | 0 | 0 |

Null означает `None`; пустые строки и пустые списки показаны раздельно. Типы рекурсивно показывают вложенные list/struct поля в JSON-представлении, не заменяют Arrow schema.

## GitHub

`projects_with_github_links` — проекты с GitHub-ссылкой в `other_links`; `projects_with_readme` — проекты с непустым `readmes[].content`; `unique_readme_repositories` — уникальные embedded README repo; `unique_github_link_repositories` — уникальные нормализованные owner/repo из ссылок. Это разные определения. Внешняя доступность ссылок не проверялась; «доступный» здесь означает только README в pinned snapshot.

По pinned raw rows число проектов с README = 1 354. Dataset card заявляет 1 358; расхождение **4 строки**. Использованы пересчитанные строки закреплённой revision; карточка может отражать другой snapshot или несовпадающий подсчёт. Причина расхождения UNKNOWN.

## Проверка меток на Devpost

Детерминированная выборка: по одному known winner и non-winner из первых двух по имени событий с обоими классами, максимум **4 страницы**. Запросы без авторизации, повторов и обхода антибот-защиты.

| Хакатон | Ожидалась метка | Статус | Наблюдение | URL |
|---|---|---|---|---|
| cal-hacks-12-0 | winner | http_403 | UNKNOWN | https://devpost.com/software/sweet-by-metabolic-company-of-calhacks |
| cal-hacks-12-0 | non_winner | http_403 | UNKNOWN | https://devpost.com/software/vodka |
| hackgt-12 | winner | http_403 | UNKNOWN | https://devpost.com/software/the-kick-is-good |
| hackgt-12 | non_winner | http_403 | UNKNOWN | https://devpost.com/software/forma-hqvane |

Все четыре запроса получили HTTP 403. Они не подтвердили и не опровергли labels; независимая проверка ground truth остаётся UNKNOWN. Запросы не повторялись.

## Дубликаты и label consistency

- Точные дубликаты сравнивались по SHA-256 канонического JSON.
- Повторные `project_id` и нормализованные Devpost URL не обнаружены. Семантические дубли и одинаковые репозитории с разными ссылками отдельно не проверялись.
- Ноль расхождений с грубым текстовым тестом не является подтверждением корректности Devpost-исходов.

## Leakage screening

Лексические индикаторы: `{"readme_keyword": 75, "description_keyword": 106}`. Поиск слов winner/winning/won/award/prize/finalist в `title/tagline/description`, README и `built_with`; `results` — target, поэтому не включается в screening признаков.

Это screening, а не доказанная временная утечка. Исторические timestamps текстов отсутствуют; текст/README могли быть написаны или изменены после результатов (**UNKNOWN**). До прогноза необходимо определить cutoff и исключить target-derived `results` и тексты без подтверждённой временной доступности.

## Group split по hackathon

| Held-out event | Winners | Non-winners | Unknown | Есть оба известных класса |
|---|---:|---:|---:|---|
| cal-hacks-12-0 | 100 | 594 | 0 | yes |
| hackgt-12 | 25 | 247 | 0 | yes |
| hacktech-by-caltech-2026 | 0 | 0 | 61 | no |
| madhacks | 10 | 45 | 0 | yes |
| madhacks-fall-2025 | 13 | 98 | 0 | yes |
| pennapps-xxv | 25 | 73 | 0 | yes |
| treehacks-2024 | 58 | 261 | 0 | yes |
| treehacks-2025 | 70 | 178 | 0 | yes |
| treehacks-2026 | 62 | 302 | 0 | yes |

Остаётся **8** групп с обоими классами; HackTech 2026 withheld. `GroupKFold` допускает максимум 8 folds и не гарантирует стратификацию. Leave-one-event-out удерживает по одному событию, и все восемь held-out событий содержат оба класса. Это не заменяет оценки разброса между событиями: групп мало, доля winners различается. Ни модели, ни fold-метрики не рассчитывались.

## GO / PIVOT / STOP

| Сценарий | Решение | Почему |
|---|---|---|
| Metadata-only baseline | **GO условно** | 2 161 known labels / 8 событий с обоими классами. Исключить HackTech 2026, зафиксировать cutoff, проверить доступность каждого признака на этот момент и заранее определить event-group split. Вывод ограничить этим набором. |
| Metadata + repository structure | **PIVOT** | README содержится только для поднабора; event/class coverage на общем A vs A+B subset надо посчитать до моделей. Внешняя GitHub доступность не проверена. Массовая загрузка repo не нужна. |
| Metadata + LLM features | **STOP сейчас** | Временная доступность текстов сомнительна, leakage не разрешён, value B ещё неизвестна; LLM не исправит малое число независимых событий. API не запускался. |

**Ответ на главный вопрос:** для ограниченного metadata-only пилотного ML — **достаточно по объёму, условно**; качество labels независимой верификацией **не подтверждено** (четыре HTTP 403). Для обобщения на «все хакатоны» — **нет**. Repository-structure сравнительная ветка ждёт общего subset audit.

## Воспроизведение

Python 3.10+; только standard library, установка пакетов не нужна.

```bash
python data_audit.py
```

Скрипт запрашивает Hugging Face rows API с revision `3cb29f15841c7b61fd78a20ff12127e0846dc227` и кэширует JSONL в `data/raw/`. Повторный запуск использует cache. Для повторной загрузки той же revision:

```bash
python data_audit.py --refresh
```

Результаты записываются в `data_audit_summary.csv` и `data_audit_report.md`. Spot-check ограничен четырьмя запросами и сохраняется в pinned cache; при повторе новые Devpost-запросы не отправляются. `--no-devpost-check` пропускает их.

## Ограничения

1. Devpost spot-check не дал подтверждения из-за HTTP 403.
2. GitHub URL availability не проверялась; embedded README — только сохранённый снимок.
3. Duplicate audit точный; семантические повторы не искались.
4. Leakage screening лексический; временные метки текста отсутствуют.
5. Модели, метрики, power/calibration не оценивались.
