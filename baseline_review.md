# Независимый аудит первого ML baseline

**Вердикт: PASS WITH FINDINGS.** Код и метрики воспроизводимы; подтверждённой ошибки, искажающей текущий эксперимент, не найдено. Главная оговорка — pooled Average Precision сопоставляет score разных fold-моделей. Baseline пригоден как технический ориентир для NumPy-реализации при точном совпадении preprocessing и objective; полученные метрики не доказывают качество предсказаний на будущих хакатонах.

## 1. ML pipeline

`technology_baseline.py` читает закреплённый JSONL snapshot и уже подготовленную `data/processed/analytical_projects.csv`. Из исходного JSONL в эксперимент переносится только `project_id` и `built_with`; эти данные соединяются one-to-one по ID. Метка берётся из `outcome` аналитической таблицы. Исходные `results`, описания, README, URL и другие поля не входят в матрицу признаков.

Независимое сопоставление ID подтвердило: в источнике 2 222 уникальных проекта, из них 61 HackTech 2026; в аналитической таблице 2 161 проект, нет HackTech ID, все остальные 2 161 source ID присутствуют, лишних ID нет. Классы совпали с аудитом источника: 363 winners и 1 798 non-winners.

## 2. X, y и groups

- **`y`**: `1` для winner, `0` для non-winner.
- **`X`**: train-fold vocabulary бинарных признаков технологий из `built_with` плюс один стандартизованный признак `technology_count`.
- **`groups`**: значение `hackathon`; используется только для формирования train/test, не как predictor. `project_id` нужен для join и диагностики ошибок, но не передаётся модели.

В коде `X` строится явной сборкой матриц из `x_train_tags`/`x_test_tags` и scaled count. Поэтому `is_winner`, `outcome`, `results`, ID и название хакатона не просачиваются в predictor columns.

## 3. Preprocessing, fit и predict_proba

Теги `strip().casefold()`-нормализуются и дедуплицируются внутри проекта; пустые, нестроковые и punctuation-only значения исключены. Лексический фильтр удаляет очевидные outcome-похожие теги из индикаторов и счётчика; в текущем snapshot таких тегов не найдено. Это снижает риск прямой лексической утечки, но не исключает семантическую или временную утечку.

На каждом fold `MultiLabelBinarizer.fit_transform` обучает vocabulary только на train. Теги тестового мероприятия, не встреченные в train, исключаются из one-hot матрицы и подсчитываются. `StandardScaler` также fit-ится только на train count. Эти fitted преобразования затем применяются к обеим частям; количество неизвестных test-тегов остаётся частью count.

В каждом вызове fold-функции создаются новые `DummyClassifier(strategy="prior")` и `LogisticRegression(max_iter=1000, random_state=0)`. Оба fit-ятся на одних train строках и матрице. Порог классификации фиксирован на 0.5; гиперпараметры и порог по test не выбирались. `predict_proba` даёт непрерывный score для AP; его нельзя автоматически считать калиброванной вероятностью.

## 4. Leave-one-hackathon-out

Для каждого из 8 мероприятий все его проекты попадают в test, остальные 7 — в train. Код проверяет наличие обоих классов в train и test. В результате получено 16 строк метрик (8 групп × 2 модели), и каждый проект имеет ровно один out-of-fold прогноз от каждой модели. HackTech 2026 полностью исключён до группового разбиения.

Это оценка переноса на удержанные группы из данного snapshot, а не временная проверка на будущих соревнованиях. Три группы — последовательные TreeHacks; восемь event-групп не следует считать гарантированно независимыми репликациями.

## 5. Метрики и повтор воспроизведения

Precision, recall и F1 рассчитаны для положительного класса winner при фиксированном пороге 0.5. AP рассчитан по непрерывному `predict_proba`. Macro — простое среднее восьми event-метрик; pooled precision/recall/F1 рассчитаны по объединённым out-of-fold решениям.

Повторный запуск неизменённого `technology_baseline.py` на тех же локальных данных и pinned зависимостях дал:

| Проверка | Результат |
|---|---:|
| Проекты / winners / non-winners | 2 161 / 363 / 1 798 |
| Held-out hackathons | 8 |
| Macro AP: LogisticRegression / Dummy | 0,243 / 0,178 |
| Pooled F1: LogisticRegression | 0,079 |
| LogisticRegression: TP / FN / FP | 17 / 346 / 48 |
| Fold-метрики, где LR лучше Dummy: AP / F1 | 8/8 / 3/8 |
| Convergence warnings | 0 |

Macro AP и суммы TP/FN/FP независимо пересчитаны по CSV. Pooled F1 из этих сумм равен 0,07944, что округляется до 0,079. Dummy получает AP, равный доле winners в каждом fold, поскольку его prior-score внутри fold постоянен.

**Почему AP выше F1:** AP учитывает ранжирование scores по всему диапазону порогов. F1 здесь проверяется при одном заранее фиксированном пороге 0.5. В большинстве fold LogisticRegression не предсказала winners при этом пороге, но scores всё же чаще ранжировали winners выше non-winners. Поэтому прирост AP не означает хороший выбор победителей при пороге 0.5: модель нашла 17 из 363 winners и пропустила 346.

## 6. Подтверждённые ошибки и ограничения

**Подтверждённых ошибок кода или текущих численных результатов не найдено; исправления не вносились.** Независимый read-only Reviewer также не выявил code defect. Его основной finding совпал с методологической оговоркой ниже.

| Severity | Location | Finding | Classification and action |
|---|---|---|---|
| P2 | `technology_baseline.py:334-354`; `technology_baseline_report.md:90-97` | Pooled AP ranks scores from eight separately fitted models. Fold score scales/calibration are not guaranteed comparable; `DummyClassifier` scores are each fold's train prior. Values 0.220 and 0.147 are correctly computed over pooled scores but can reflect between-fold score differences, not only within-event ranking. | Methodological limitation confirmed by Reviewer. No code correction; use macro AP as the cleaner primary summary and treat pooled AP as descriptive, not global ranking ability. |
| P2 | `technology_baseline.py:206-214`; `data_audit_report.md:90-104` | There are only 8 event groups, winner share varies 9.2–28.2%, and TreeHacks appears in three years. | External validity is limited. Do not infer performance on arbitrary/future hackathons or claim precise between-event uncertainty. |
| P2 | `technology_baseline_report.md:158-160`; `data_quality_eda_report.md:97-103` | Historical availability of `built_with` is unconfirmed; previous Devpost label spot-checks returned HTTP 403. | Already disclosed in prior reports. Retrospective snapshot classification only; no pre-event prediction claim and no independent ground-truth confirmation. |
| P3 | `technology_baseline.py:128,228-234,297-309`; `technology_baseline_report.md:154` | In train, `technology_count` equals the sum of binary indicators and is algebraically redundant. Many tags are rare; correlated-feature coefficients are not independent or causal effects. | Interpretation limitation already disclosed. Does not itself invalidate predictions; coefficients are educational diagnostics, not technology rankings. |
| P3 | `technology_baseline.py:96-111,125-133` | `load_data()` checks row count, classes, HackTech absence and one-to-one merge, but does not assert exact equality between analytical IDs and source IDs excluding HackTech. | Future-input guard gap only. Independent exact-ID comparison found no missing or extra IDs in this run; no observed result defect, so no code change. |

Сравнение с `DummyClassifier(strategy="prior")` корректно для заданного trivial baseline: те же строки и folds, train-only prior, никакой информации из `X`. При доле победителей ниже 50% он предсказывает всех как non-winner, поэтому F1=0 — ожидаемое свойство baseline, а не свидетельство высокой практической ценности LR. Это отправная точка, не сильный конкурент.

## 7. Итог и план NumPy-реализации

**PASS WITH FINDINGS.** Техническая схема годится как эталон для самостоятельной реализации: данные, группы, признаки, train-only preprocessing, фиксированный порог и метрики определены и воспроизводятся. Для численного совпадения с sklearn нужно отдельно согласовать порядок признаков, intercept, L2-регуляризацию и её нормировку, параметры solver/tolerance и scaling; `requirements-ml.txt` фиксирует sklearn 1.9.1. Разные оптимизаторы могут давать небольшие различия коэффициентов без ошибки.

Рекомендуемый учебный порядок — без написания реализации в рамках этого аудита:

1. Определить форму `X`, вектор `y`, линейный score `z = Xw + b` и устойчивую sigmoid.
2. Вывести binary cross-entropy и градиенты по `w` и `b`; затем добавить L2 с той же конвенцией штрафа и не штрафовать intercept, если это соответствует выбранному sklearn reference.
3. Реализовать фиксированный оптимизатор и критерий остановки; отделить `fit`, score/probability и threshold-based `predict`.
4. Проверить вручную вычислимую loss, конечность значений на больших logits, градиент конечными разностями, уменьшение loss на одном обновлении, формы массивов и корректность `predict_proba`.
5. Сначала сравнить на маленьком фиксированном наборе с sklearn при совпадающих objective и preprocessing; затем повторить те же 8 LOHO folds и сравнить train-fitted preprocessing, probabilities, confusion counts, fold AP/F1 и macro-метрики. Не менять порог по test и не подгонять формулу под итоговые метрики.

**Результат аудита:** исходные данные и настройки эксперимента не менялись; новый файл этого отчёта — единственный новый артефакт. Повторный запуск пересоздал существующие generated outputs с теми же метриками.