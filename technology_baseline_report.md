# Технологический metadata baseline: retrospective LOHO

**Дата:** 2026-11-09T10:53:12+00:00<br>
**Источник:** `twangodev/devpost-hacks`, pinned revision `3cb29f15841c7b61fd78a20ff12127e0846dc227`; использован уже загруженный cache и подготовленная таблица из 2 161 проекта.<br>
**Статус:** ретроспективный эксперимент на сохранённом snapshot. Это не проверенная модель прогноза до судейства.

## Кратко о данных и признаках

- **2 161** проектов: 363 winners и 1 798 non-winners, 8 событий. HackTech 2026 не включён: результаты были неизвестны при сборе.
- `y` — бинарная метка: winner=1, non-winner=0. `X` содержит бинарные indicators по technology tags и число оставшихся тегов. `project_id`, `hackathon`, `results`, описания и любые target-derived поля не передаются в модели.
- Обработка тегов минимальна: строки `strip().casefold()`, дубликаты внутри проекта удалены; пустые, нестроковые и punctuation-only значения отброшены. Outcome-похожие значения проверяются и исключаются из indicators и count.
- **3151** уникальных обычных technology tags; среднее 8.20, медиана 7; проектов без пригодных тегов: 1. Нестроковых записей отброшено: 0; punctuation-only entries: 1.

Top tags (доля отдельно не равна качеству проекта):

| Tag | Projects | Winners | Non-winners |
|---|---:|---:|---:|
| `python` | 1188 | 216 | 972 |
| `react` | 772 | 131 | 641 |
| `typescript` | 626 | 119 | 507 |
| `javascript` | 597 | 87 | 510 |
| `openai` | 417 | 73 | 344 |
| `fastapi` | 317 | 57 | 260 |
| `node.js` | 305 | 50 | 255 |
| `css` | 299 | 35 | 264 |
| `next.js` | 288 | 59 | 229 |
| `claude` | 285 | 42 | 243 |

Outcome-похожие значения в `built_with`:

| Tag | Projects | Winners | Non-winners |
|---|---:|---:|---:|
| Нет совпадений | 0 | 0 | 0 |

**Важная оговорка:** это лексический фильтр по очевидным словам (`winner`, `award`, `prize`, `finalist`, места, champion/victory); он снижает прямую утечку, но не доказывает, что остальные tags доступны до окончания события или не кодируют результат косвенно.

## Дизайн

Использован leave-one-hackathon-out: на каждом из 8 проходов один hackathon целиком является test, остальные 7 — train. `hackathon` задаёт только группы и не входит в `X`.

На каждом fold заново, только на train:

1. `MultiLabelBinarizer` строит словарь технологий и бинарные columns. Неизвестные test tags игнорируются, а их число отражено в результатах.
2. `StandardScaler` обучается на train count и применяется к test. Индикаторы остаются 0/1.
3. `DummyClassifier(strategy="prior")` и `LogisticRegression` обучаются на одной и той же матрице. Для LogisticRegression фиксированы sklearn defaults, `max_iter=1000`, `random_state=0`; гиперпараметры не подбирались.

Threshold — фиксированный 0.5 (без подстройки по test). `fit(X_train, y_train)` оценивает параметры только на train; `predict` возвращает бинарный класс, а `predict_proba` — оценку вероятности для positive-класса. Она нужна для `average_precision` и не гарантированно калибрована. Dummy prior предсказывает train prior; при малой доле winners его class prediction — non-winner для всех объектов.

### Учебные понятия

- `X` — строки проектов × столбцы признаков; `y` — winner=1 / non-winner=0. ID и event slug не входят в `X`.
- Каждый technology indicator равен 1, если tag есть в проекте, иначе 0. `technology_count` — число оставшихся уникальных tags.
- В каждом LOHO fold test — одно целое мероприятие; train — семь остальных. Это проверяет перенос между event-группами, а не случайное перемешивание проектов.
- Preprocessing: `MultiLabelBinarizer.fit` строит vocabulary на train; неизвестные test tags не получают отдельный indicator. `StandardScaler.fit` также видит только train count; затем теми же преобразованиями меняются train и test.
- Precision — доля верных winners среди предсказанных winners; recall — доля найденных winners среди всех winners; F1 объединяет precision и recall. Average precision оценивает ранжирование по вероятностям, winner share — базовая доля положительного класса в test.
- Logistic coefficient — изменение log-odds при изменении признака при прочих фиксированных; это не причинный эффект. В этой задаче count является суммой бинарных indicators в train и потому алгебраически избыточен; регуляризация задаёт решение, но не делает эти веса независимыми эффектами.

## Метрики по каждому held-out hackathon

Precision/recall/F1 относятся к положительному классу winner. `Winner share` — доля winners в test; vocabulary/unseen columns показывают переносимость списка тегов.

| Held-out event | Model | Test N | Winners | Winner share | Precision | Recall | F1 | Average precision | FP | FN | Train vocab | Unseen test tag entries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cal-hacks-12-0 | dummy_prior | 694 | 100 | 14.4% | 0.000 | 0.000 | 0.000 | 0.144 | 0 | 100 | 2496 | 911 |
| cal-hacks-12-0 | logistic_regression | 694 | 100 | 14.4% | 0.250 | 0.070 | 0.109 | 0.200 | 21 | 93 | 2496 | 911 |
| hackgt-12 | dummy_prior | 272 | 25 | 9.2% | 0.000 | 0.000 | 0.000 | 0.092 | 0 | 25 | 2860 | 339 |
| hackgt-12 | logistic_regression | 272 | 25 | 9.2% | 0.000 | 0.000 | 0.000 | 0.110 | 0 | 25 | 2860 | 339 |
| madhacks | dummy_prior | 55 | 10 | 18.2% | 0.000 | 0.000 | 0.000 | 0.182 | 0 | 10 | 3124 | 27 |
| madhacks | logistic_regression | 55 | 10 | 18.2% | 0.000 | 0.000 | 0.000 | 0.262 | 0 | 10 | 3124 | 27 |
| madhacks-fall-2025 | dummy_prior | 111 | 13 | 11.7% | 0.000 | 0.000 | 0.000 | 0.117 | 0 | 13 | 3038 | 113 |
| madhacks-fall-2025 | logistic_regression | 111 | 13 | 11.7% | 0.000 | 0.000 | 0.000 | 0.146 | 0 | 13 | 3038 | 113 |
| pennapps-xxv | dummy_prior | 98 | 25 | 25.5% | 0.000 | 0.000 | 0.000 | 0.255 | 0 | 25 | 3079 | 120 |
| pennapps-xxv | logistic_regression | 98 | 25 | 25.5% | 0.000 | 0.000 | 0.000 | 0.402 | 0 | 25 | 3079 | 120 |
| treehacks-2024 | dummy_prior | 319 | 58 | 18.2% | 0.000 | 0.000 | 0.000 | 0.182 | 0 | 58 | 2894 | 402 |
| treehacks-2024 | logistic_regression | 319 | 58 | 18.2% | 0.000 | 0.000 | 0.000 | 0.245 | 5 | 58 | 2894 | 402 |
| treehacks-2025 | dummy_prior | 248 | 70 | 28.2% | 0.000 | 0.000 | 0.000 | 0.282 | 0 | 70 | 2891 | 347 |
| treehacks-2025 | logistic_regression | 248 | 70 | 28.2% | 0.556 | 0.071 | 0.127 | 0.348 | 4 | 65 | 2891 | 347 |
| treehacks-2026 | dummy_prior | 364 | 62 | 17.0% | 0.000 | 0.000 | 0.000 | 0.170 | 0 | 62 | 2419 | 816 |
| treehacks-2026 | logistic_regression | 364 | 62 | 17.0% | 0.217 | 0.081 | 0.118 | 0.230 | 18 | 57 | 2419 | 816 |

### Сводка по событиям

Macro average — простое среднее восьми fold-метрик, каждое мероприятие имеет одинаковый вес.

| Model | Macro precision | Macro recall | Macro F1 | Macro average precision |
|---|---:|---:|---:|---:|
| dummy_prior | 0.000 | 0.000 | 0.000 | 0.178 |
| logistic_regression | 0.128 | 0.028 | 0.044 | 0.243 |

Pooled out-of-fold метрики объединяют все 2 161 test-предсказание (каждый проект предсказан только в fold своего hackathon).

| Model | Pooled precision | Pooled recall | Pooled F1 | Pooled average precision |
|---|---:|---:|---:|---:|
| dummy_prior | 0.000 | 0.000 | 0.000 | 0.147 |
| logistic_regression | 0.262 | 0.047 | 0.079 | 0.220 |

LogisticRegression превзошла Dummy по F1 в **3/8** events и по average precision в **8/8**. Диапазоны её fold F1: 0.000–0.127; average precision: 0.110–0.402. При пороге 0.5 модель отметила **17/363** winners, пропустила 346 и ошибочно отметила non-winners 48 раз. Суммарно **3075/17713 (17.4%)** project-tag entries не встречались в train vocabulary; это затронуло **1267/2161** test-проектов. Неизвестные tags не добавлялись в indicators, но остаются учтены в count.

## Ошибки и коэффициенты

Ошибки — out-of-fold false positives/false negatives; ID и прогнозы сохраняются отдельно в `data/processed/technology_baseline_errors.csv` (локальная таблица под `/data/`, не исходные тексты). Counts сгруппированы ниже.

| Held-out event | Model | Error | Projects |
|---|---|---|---:|
| cal-hacks-12-0 | dummy_prior | false_negative | 100 |
| cal-hacks-12-0 | logistic_regression | false_negative | 93 |
| cal-hacks-12-0 | logistic_regression | false_positive | 21 |
| hackgt-12 | dummy_prior | false_negative | 25 |
| hackgt-12 | logistic_regression | false_negative | 25 |
| madhacks | dummy_prior | false_negative | 10 |
| madhacks | logistic_regression | false_negative | 10 |
| madhacks-fall-2025 | dummy_prior | false_negative | 13 |
| madhacks-fall-2025 | logistic_regression | false_negative | 13 |
| pennapps-xxv | dummy_prior | false_negative | 25 |
| pennapps-xxv | logistic_regression | false_negative | 25 |
| treehacks-2024 | dummy_prior | false_negative | 58 |
| treehacks-2024 | logistic_regression | false_negative | 58 |
| treehacks-2024 | logistic_regression | false_positive | 5 |
| treehacks-2025 | dummy_prior | false_negative | 70 |
| treehacks-2025 | logistic_regression | false_negative | 65 |
| treehacks-2025 | logistic_regression | false_positive | 4 |
| treehacks-2026 | dummy_prior | false_negative | 62 |
| treehacks-2026 | logistic_regression | false_negative | 57 |
| treehacks-2026 | logistic_regression | false_positive | 18 |

Полная таблица fitted coefficients по fold лежит в `technology_baseline_coefficients.csv`. Здесь показаны средние коэффициенты технологий, встречавшихся минимум в 4 train vocabularies и имеющих support не менее 20 train projects в каждом таком fold. Порог нужен только для читаемости диагностики; он не меняет модель.

Средние наиболее положительные коэффициенты:

| Tag | Mean coefficient | Folds present | Min train support |
|---|---:|---:|---:|
| `fetch.ai` | 1.194 | 8/8 | 22 |
| `unity` | 0.711 | 8/8 | 22 |
| `elasticsearch` | 0.676 | 8/8 | 21 |
| `letta` | 0.603 | 7/8 | 49 |
| `cerebras` | 0.566 | 7/8 | 23 |
| `pytorch` | 0.543 | 8/8 | 81 |
| `render` | 0.433 | 8/8 | 22 |
| `css3` | 0.426 | 8/8 | 33 |

Средние наиболее отрицательные коэффициенты:

| Tag | Mean coefficient | Folds present | Min train support |
|---|---:|---:|---:|
| `uvicorn` | -0.715 | 8/8 | 28 |
| `api` | -0.681 | 8/8 | 47 |
| `streamlit` | -0.673 | 8/8 | 26 |
| `html5` | -0.660 | 8/8 | 43 |
| `google` | -0.654 | 8/8 | 29 |
| `anthropic` | -0.584 | 8/8 | 42 |
| `tailwind-css` | -0.570 | 8/8 | 22 |
| `websockets` | -0.492 | 8/8 | 39 |

Средний коэффициент стандартизованного technology count: 0.261; знак положительный в 8/8 folds. На train count математически равен сумме бинарных indicators; из-за этой зависимости коэффициенты не являются независимыми эффектами. Дополнительно tags коррелируют между собой и зависят от события. Знак/величина не доказывают, что технология повышает или снижает шанс победы. Convergence warnings: **0**.

## Интерпретация и ограничения

- Метрики сравнивают переносимость между этими восемью событиями; группы малы и различаются по winner share. Они не оценивают будущие хакатоны вне источника.
- `built_with` был взят из текущего snapshot. Его доступность до окончания события **не подтверждена**. Результаты — только связь/классификация по snapshot, не pre-event prediction.
- Метки не прошли независимую ручную/Devpost верификацию: предыдущие четыре spot-check получили HTTP 403. Это ограничение качества target.
- Нет гиперпараметрического поиска, threshold tuning, class weighting, отбора признаков по test, repository features, текста, LLM или внешних API.
- Pooled scores и macro scores отвечают на разные вопросы: pooled весит проекты, macro — события. Ни один не устраняет малое число независимых групп.

## Воспроизведение

Python 3.12; версии и только прямые зависимости: `requirements-ml.txt`.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements-ml.txt
.venv/Scripts/python.exe technology_baseline.py
```

Скрипт использует локальные `data/raw/devpost-hacks-3cb29f15841c7b61fd78a20ff12127e0846dc227.jsonl` и `data/processed/analytical_projects.csv`; ничего не скачивает. Результаты: `technology_baseline_results.csv`, `technology_baseline_coefficients.csv`, `technology_tag_summary.csv`, `technology_tag_screening.csv`; подробные per-project ошибки остаются в игнорируемой `data/processed/`.

**Версии запуска:** numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1.
