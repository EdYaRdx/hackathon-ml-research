#!/usr/bin/env python3
"""Retrospective technology-metadata baseline with leave-one-hackathon-out CV.

The script reads the pinned local JSONL cache and the already prepared 2,161-row
analytical table. It never downloads data, repository contents, or external URLs.
"""
from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression

REVISION = "3cb29f15841c7b61fd78a20ff12127e0846dc227"
ROOT = Path(__file__).resolve().parent
PINNED_JSONL = ROOT / "data" / "raw" / f"devpost-hacks-{REVISION}.jsonl"
ANALYTICAL_CSV = ROOT / "data" / "processed" / "analytical_projects.csv"
RESULTS_CSV = ROOT / "technology_baseline_results.csv"
COEFFICIENTS_CSV = ROOT / "technology_baseline_coefficients.csv"
TAG_SUMMARY_CSV = ROOT / "technology_tag_summary.csv"
TAG_SCREEN_CSV = ROOT / "technology_tag_screening.csv"
ERRORS_CSV = ROOT / "data" / "processed" / "technology_baseline_errors.csv"
REPORT_MD = ROOT / "technology_baseline_report.md"
EXPECTED_ROWS = 2161
EXPECTED_WINNERS = 363
UNPUBLISHED_EVENT = "hacktech-by-caltech-2026"

# These terms are not technologies in this task; treat them as possible direct
# outcome labels and exclude matching tags from both the indicator features and count.
OUTCOME_LIKE = re.compile(
    r"(?<![a-z0-9])(?:winner|winning|won|award(?:ed|s)?|prize(?:s)?|finalist|"
    r"champion(?:s|ship)?|victory|first[\s_-]+place|second[\s_-]+place|third[\s_-]+place)(?![a-z0-9])",
    flags=re.IGNORECASE,
)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a predictable UTF-8 CSV; empty outputs still have a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_and_screen_tags(value) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    """Normalize tags, discard punctuation-only noise, and screen outcome labels."""
    if not isinstance(value, list):
        return (), (), 0, 0
    normalized = set()
    non_string = 0
    punctuation_only = 0
    for item in value:
        if not isinstance(item, str):
            non_string += 1
            continue
        tag = item.strip().casefold()
        if not tag:
            continue
        if not any(character.isalnum() for character in tag):
            punctuation_only += 1
            continue
        normalized.add(tag)
    suspicious = tuple(sorted(tag for tag in normalized if OUTCOME_LIKE.search(tag)))
    technologies = tuple(sorted(tag for tag in normalized if tag not in suspicious))
    return technologies, suspicious, non_string, punctuation_only


def load_data() -> tuple[pd.DataFrame, dict]:
    """Join cached built_with values to the prepared, labeled analytical table."""
    if not PINNED_JSONL.is_file():
        raise FileNotFoundError(f"Pinned cache not found: {PINNED_JSONL}; run the Data Audit first")
    if not ANALYTICAL_CSV.is_file():
        raise FileNotFoundError(f"Prepared analytical table not found: {ANALYTICAL_CSV}; run Data Quality EDA first")

    # The cache contains U+2028/U+2029 inside JSON text in some records. Split
    # only on the physical newline, just as data_audit.py does.
    source_records = [json.loads(line) for line in PINNED_JSONL.read_text(encoding="utf-8").split("\n") if line]
    if len(source_records) != 2222:
        raise RuntimeError(f"Pinned cache has {len(source_records)} rows; expected 2222")
    source_ids = [str(row.get("project_id") or "") for row in source_records]
    if any(not project_id for project_id in source_ids) or len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Pinned cache must have unique, non-empty project_id values")

    analytical = pd.read_csv(ANALYTICAL_CSV, dtype={"project_id": "string"})
    if len(analytical) != EXPECTED_ROWS:
        raise RuntimeError(f"Prepared table has {len(analytical)} rows; expected {EXPECTED_ROWS}")
    if analytical["project_id"].isna().any() or analytical["project_id"].duplicated().any():
        raise RuntimeError("Prepared table must have one non-empty row per project_id")
    if UNPUBLISHED_EVENT in set(analytical["hackathon"].astype(str)):
        raise RuntimeError("Unpublished HackTech 2026 must not enter the labeled table")

    # Parse only the requested technology list from the source; do not carry
    # descriptions, README, result text, URLs, or other fields into the experiment.
    source_features = pd.DataFrame([
        {"project_id": str(row.get("project_id") or ""), "built_with": row.get("built_with")}
        for row in source_records
    ])
    data = analytical.merge(source_features, on="project_id", how="left", validate="one_to_one")
    if data["built_with"].isna().all():
        raise RuntimeError("No built_with values joined to the analytical projects")

    known_outcomes = data["outcome"].map({"winner": 1, "non_winner": 0})
    if known_outcomes.isna().any():
        raise RuntimeError("Analytical table contains an unexpected outcome value")
    data["is_winner"] = known_outcomes.astype("int8")
    if int(data["is_winner"].sum()) != EXPECTED_WINNERS:
        raise RuntimeError(f"Expected {EXPECTED_WINNERS} winners; got {int(data['is_winner'].sum())}")
    if set(data["is_winner"].unique()) != {0, 1}:
        raise RuntimeError("The labeled dataset must contain both classes")

    tag_rows = [normalize_and_screen_tags(value) for value in data["built_with"]]
    data["technology_tags"] = [item[0] for item in tag_rows]
    data["outcome_like_tags"] = [item[1] for item in tag_rows]
    data["non_string_tag_entries"] = [item[2] for item in tag_rows]
    data["punctuation_only_tag_entries"] = [item[3] for item in tag_rows]
    data["technology_count"] = data["technology_tags"].map(len).astype("int32")

    # No identifier, event name, results text, or output-derived field is in X.
    if len(data) != EXPECTED_ROWS or data["project_id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError("The joined experiment table is not one row per prepared project")
    return data, {"source_rows": len(source_records), "project_rows": len(data)}


def summarize_tags(data: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    technology_counts = Counter()
    technology_winners = Counter()
    technology_non_winners = Counter()
    suspicious_counts = Counter()
    suspicious_winners = Counter()
    suspicious_non_winners = Counter()
    for row in data.itertuples(index=False):
        for tag in row.technology_tags:
            technology_counts[tag] += 1
            if row.is_winner:
                technology_winners[tag] += 1
            else:
                technology_non_winners[tag] += 1
        for tag in row.outcome_like_tags:
            suspicious_counts[tag] += 1
            if row.is_winner:
                suspicious_winners[tag] += 1
            else:
                suspicious_non_winners[tag] += 1

    winners = int(data["is_winner"].sum())
    non_winners = len(data) - winners
    tag_summary = []
    for tag, projects in technology_counts.most_common():
        tag_summary.append({
            "technology": tag,
            "projects": projects,
            "winner_projects": technology_winners[tag],
            "non_winner_projects": technology_non_winners[tag],
            "winner_prevalence": technology_winners[tag] / winners,
            "non_winner_prevalence": technology_non_winners[tag] / non_winners,
        })
    screen_summary = [{
        "suspicious_tag": tag,
        "projects": count,
        "winner_projects": suspicious_winners[tag],
        "non_winner_projects": suspicious_non_winners[tag],
        "excluded_from_features": True,
    } for tag, count in sorted(suspicious_counts.items())]
    stats = {
        "unique_technology_tags": len(technology_counts),
        "technology_tag_entries": sum(technology_counts.values()),
        "suspicious_tags": len(suspicious_counts),
        "projects_with_suspicious_tags": int(data["outcome_like_tags"].map(bool).sum()),
        "suspicious_tag_entries": sum(suspicious_counts.values()),
        "non_string_tag_entries": int(data["non_string_tag_entries"].sum()),
        "punctuation_only_tag_entries": int(data["punctuation_only_tag_entries"].sum()),
        "technology_count_mean": float(data["technology_count"].mean()),
        "technology_count_median": float(data["technology_count"].median()),
        "zero_technology_projects": int((data["technology_count"] == 0).sum()),
        "top_tags": tag_summary[:10],
        "outcome_like_tag_summary": screen_summary,
    }
    return tag_summary, screen_summary, stats


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_winner: np.ndarray) -> dict:
    """Use the fixed 0.5 class threshold; AP uses continuous winner scores."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, p_winner),
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
    }


def fit_one_fold(data: pd.DataFrame, heldout_event: str, coefficient_rows: list[dict],
                 error_rows: list[dict], prediction_rows: list[dict], warning_rows: list[dict]) -> list[dict]:
    """Fit preprocessing and two fixed models using only the other events."""
    train = data.loc[data["hackathon"] != heldout_event].reset_index(drop=True)
    test = data.loc[data["hackathon"] == heldout_event].reset_index(drop=True)
    if len(test) == 0 or set(test["is_winner"].unique()) != {0, 1}:
        raise RuntimeError(f"Held-out event {heldout_event!r} must contain both classes")
    if set(train["is_winner"].unique()) != {0, 1}:
        raise RuntimeError(f"Training data without {heldout_event!r} must contain both classes")

    # Fit the indicator vocabulary on train only. A technology unseen in train
    # is omitted from test indicators, never added by looking at the test event.
    encoder = MultiLabelBinarizer()
    x_train_tags = encoder.fit_transform(train["technology_tags"])
    training_vocabulary = set(encoder.classes_)
    test_tags_seen_in_train = [
        tuple(tag for tag in tags if tag in training_vocabulary)
        for tags in test["technology_tags"]
    ]
    x_test_tags = encoder.transform(test_tags_seen_in_train)
    unseen_counts = [sum(tag not in training_vocabulary for tag in tags) for tags in test["technology_tags"]]

    # Standardize only the numeric count, fitting mean and standard deviation
    # on train. The technology indicators remain binary 0/1 columns.
    count_scaler = StandardScaler()
    x_train_count = count_scaler.fit_transform(train[["technology_count"]])
    x_test_count = count_scaler.transform(test[["technology_count"]])
    x_train = np.column_stack((x_train_tags, x_train_count))
    x_test = np.column_stack((x_test_tags, x_test_count))
    y_train = train["is_winner"].to_numpy(dtype="int8")
    y_test = test["is_winner"].to_numpy(dtype="int8")

    # Same fold and same feature matrix for both models. No class weights,
    # threshold search, or hyperparameter selection uses the held-out event.
    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=0),
    }
    fold_results = []
    predictions_by_model = {}
    for model_name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x_train, y_train)
        for warning in caught:
            warning_rows.append({"heldout_hackathon": heldout_event, "model": model_name,
                                 "warning_category": type(warning.message).__name__,
                                 "warning": str(warning.message)})
        predictions = model.predict(x_test)
        class_one_index = list(model.classes_).index(1)
        p_winner = model.predict_proba(x_test)[:, class_one_index]
        predictions_by_model[model_name] = (predictions, p_winner)
        metrics = classification_metrics(y_test, predictions, p_winner)
        fold_results.append({
            "heldout_hackathon": heldout_event,
            "model": model_name,
            "n_train": len(train),
            "n_test": len(test),
            "test_winners": int(y_test.sum()),
            "test_non_winners": int(len(y_test) - y_test.sum()),
            "winner_share": float(y_test.mean()),
            "training_winner_share": float(y_train.mean()),
            "train_technology_vocabulary": len(encoder.classes_),
            "test_projects_with_unseen_technology": int(sum(count > 0 for count in unseen_counts)),
            "unseen_test_tag_entries": int(sum(unseen_counts)),
            "threshold": 0.5,
            **metrics,
        })
        for index, (actual, predicted, probability) in enumerate(zip(y_test, predictions, p_winner)):
            prediction_rows.append({
                "project_id": str(test.loc[index, "project_id"]),
                "heldout_hackathon": heldout_event,
                "model": model_name,
                "actual_is_winner": int(actual),
                "predicted_is_winner": int(predicted),
                "p_winner": float(probability),
                "unseen_technology_tags": int(unseen_counts[index]),
            })
            if actual != predicted:
                error_rows.append({
                    "project_id": str(test.loc[index, "project_id"]),
                    "heldout_hackathon": heldout_event,
                    "model": model_name,
                    "error_type": "false_negative" if actual == 1 else "false_positive",
                    "actual_is_winner": int(actual),
                    "predicted_is_winner": int(predicted),
                    "p_winner": float(probability),
                    "technology_count": int(test.loc[index, "technology_count"]),
                    "unseen_technology_tags": int(unseen_counts[index]),
                })

    logistic = models["logistic_regression"]
    names = list(encoder.classes_) + ["technology_count_standardized"]
    values = logistic.coef_.ravel()
    train_tag_counts = Counter(tag for tags in train["technology_tags"] for tag in tags)
    for feature, coefficient in zip(names, values):
        coefficient_rows.append({
            "heldout_hackathon": heldout_event,
            "feature": feature,
            "feature_type": "technology_indicator" if feature != "technology_count_standardized" else "numeric_count",
            "coefficient": float(coefficient),
            "train_project_support": train_tag_counts.get(feature, ""),
            "training_projects": len(train),
            "training_vocabulary_size": len(encoder.classes_),
        })

    # Keep the comparison's shared test rows explicit; this assertion guards
    # against accidentally evaluating the two models on different samples.
    if len(predictions_by_model) != 2 or any(len(pair[0]) != len(test) for pair in predictions_by_model.values()):
        raise RuntimeError("Models were not evaluated on the same held-out rows")
    return fold_results


def metric_report(results: pd.DataFrame, predictions: pd.DataFrame, data: pd.DataFrame,
                  tag_stats: dict, warning_rows: list[dict]) -> str:
    event_order = sorted(data["hackathon"].astype(str).unique())
    table_rows = []
    for event in event_order:
        for model in ("dummy_prior", "logistic_regression"):
            row = results.loc[(results["heldout_hackathon"] == event) & (results["model"] == model)].iloc[0]
            table_rows.append(
                f"| {event} | {model} | {int(row.n_test)} | {int(row.test_winners)} | {row.winner_share:.1%} | "
                f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.average_precision:.3f} | "
                f"{int(row.false_positive)} | {int(row.false_negative)} | {int(row.train_technology_vocabulary)} | "
                f"{int(row.unseen_test_tag_entries)} |"
            )
    per_event_table = "\n".join(table_rows)

    macro = results.groupby("model")[["precision", "recall", "f1", "average_precision"]].mean()
    y_all = predictions.drop_duplicates("project_id").set_index("project_id")["actual_is_winner"]
    macro_rows = []
    pooled_rows = []
    for model in ("dummy_prior", "logistic_regression"):
        model_rows = predictions.loc[predictions["model"] == model].drop_duplicates("project_id").set_index("project_id")
        model_rows = model_rows.loc[y_all.index]
        pooled = classification_metrics(
            model_rows["actual_is_winner"].to_numpy(dtype="int8"),
            model_rows["predicted_is_winner"].to_numpy(dtype="int8"),
            model_rows["p_winner"].to_numpy(dtype=float),
        )
        macro_row = macro.loc[model]
        macro_rows.append(
            f"| {model} | {macro_row['precision']:.3f} | {macro_row['recall']:.3f} | {macro_row['f1']:.3f} | "
            f"{macro_row['average_precision']:.3f} |"
        )
        pooled_rows.append(
            f"| {model} | {pooled['precision']:.3f} | {pooled['recall']:.3f} | {pooled['f1']:.3f} | "
            f"{pooled['average_precision']:.3f} |"
        )
    macro_table = "\n".join(macro_rows)
    pooled_table = "\n".join(pooled_rows)

    event_metrics = results.pivot(index="heldout_hackathon", columns="model", values=["f1", "average_precision"])
    lr_f1_better = int((event_metrics[("f1", "logistic_regression")] > event_metrics[("f1", "dummy_prior")]).sum())
    lr_ap_better = int((event_metrics[("average_precision", "logistic_regression")] > event_metrics[("average_precision", "dummy_prior")]).sum())
    lr_f1 = event_metrics[("f1", "logistic_regression")]
    lr_ap = event_metrics[("average_precision", "logistic_regression")]
    error_frame = pd.read_csv(ERRORS_CSV) if ERRORS_CSV.exists() else pd.DataFrame()
    error_lines = []
    if not error_frame.empty:
        by_event = error_frame.groupby(["heldout_hackathon", "model", "error_type"]).size()
        for (event, model, error_type), count in by_event.items():
            error_lines.append(f"| {event} | {model} | {error_type} | {count} |")
    error_table = "\n".join(error_lines) or "| — | — | — | 0 |"

    suspicious_lines = "\n".join(
        f"| `{item['suspicious_tag']}` | {item['projects']} | {item['winner_projects']} | {item['non_winner_projects']} |"
        for item in tag_stats["outcome_like_tag_summary"]
    ) or "| Нет совпадений | 0 | 0 | 0 |"
    top_tags = "\n".join(
        f"| `{item['technology']}` | {item['projects']} | {item['winner_projects']} | {item['non_winner_projects']} |"
        for item in tag_stats["top_tags"]
    )

    coef_frame = pd.read_csv(COEFFICIENTS_CSV)
    tech_coefficients = coef_frame.loc[coef_frame["feature_type"] == "technology_indicator"]
    tech_coefficients = tech_coefficients.copy()
    tech_coefficients["train_project_support"] = pd.to_numeric(
        tech_coefficients["train_project_support"], errors="coerce"
    )
    coef_group = tech_coefficients.groupby("feature").agg(
        mean=("coefficient", "mean"),
        median=("coefficient", "median"),
        folds_present=("coefficient", "count"),
        min_train_support=("train_project_support", "min"),
    )
    coef_group = coef_group.loc[
        (coef_group["folds_present"] >= 4) & (coef_group["min_train_support"] >= 20)
    ]
    positive = coef_group.sort_values("mean", ascending=False).head(8)
    negative = coef_group.sort_values("mean", ascending=True).head(8)
    positive_table = "\n".join(
        f"| `{tag}` | {row['mean']:.3f} | {int(row['folds_present'])}/8 | {int(row['min_train_support'])} |"
        for tag, row in positive.iterrows()
    ) or "| Нет | — | — | — |"
    negative_table = "\n".join(
        f"| `{tag}` | {row['mean']:.3f} | {int(row['folds_present'])}/8 | {int(row['min_train_support'])} |"
        for tag, row in negative.iterrows()
    ) or "| Нет | — | — | — |"
    count_coef = coef_frame.loc[coef_frame["feature"] == "technology_count_standardized", "coefficient"]
    count_coef_line = (
        f"Средний коэффициент стандартизованного technology count: {count_coef.mean():.3f}; "
        f"знак положительный в {(count_coef > 0).sum()}/8 folds."
        if len(count_coef) else "Коэффициент technology count отсутствует."
    )
    convergence = len(warning_rows)
    unseen_total = int(results.loc[results["model"] == "logistic_regression", "unseen_test_tag_entries"].sum())
    unseen_projects_total = int(results.loc[results["model"] == "logistic_regression", "test_projects_with_unseen_technology"].sum())
    unseen_share = unseen_total / tag_stats["technology_tag_entries"] if tag_stats["technology_tag_entries"] else 0.0
    logistic_rows = results.loc[results["model"] == "logistic_regression"]
    logistic_true_positives = int(logistic_rows["true_positive"].sum())
    logistic_false_positives = int(logistic_rows["false_positive"].sum())
    logistic_false_negatives = int(logistic_rows["false_negative"].sum())
    return f'''# Технологический metadata baseline: retrospective LOHO

**Дата:** {datetime.now(timezone.utc).isoformat(timespec="seconds")}<br>
**Источник:** `twangodev/devpost-hacks`, pinned revision `{REVISION}`; использован уже загруженный cache и подготовленная таблица из 2 161 проекта.<br>
**Статус:** ретроспективный эксперимент на сохранённом snapshot. Это не проверенная модель прогноза до судейства.

## Кратко о данных и признаках

- **2 161** проектов: 363 winners и 1 798 non-winners, 8 событий. HackTech 2026 не включён: результаты были неизвестны при сборе.
- `y` — бинарная метка: winner=1, non-winner=0. `X` содержит бинарные indicators по technology tags и число оставшихся тегов. `project_id`, `hackathon`, `results`, описания и любые target-derived поля не передаются в модели.
- Обработка тегов минимальна: строки `strip().casefold()`, дубликаты внутри проекта удалены; пустые, нестроковые и punctuation-only значения отброшены. Outcome-похожие значения проверяются и исключаются из indicators и count.
- **{tag_stats['unique_technology_tags']}** уникальных обычных technology tags; среднее {tag_stats['technology_count_mean']:.2f}, медиана {tag_stats['technology_count_median']:.0f}; проектов без пригодных тегов: {tag_stats['zero_technology_projects']}. Нестроковых записей отброшено: {tag_stats['non_string_tag_entries']}; punctuation-only entries: {tag_stats['punctuation_only_tag_entries']}.

Top tags (доля отдельно не равна качеству проекта):

| Tag | Projects | Winners | Non-winners |
|---|---:|---:|---:|
{top_tags}

Outcome-похожие значения в `built_with`:

| Tag | Projects | Winners | Non-winners |
|---|---:|---:|---:|
{suspicious_lines}

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
{per_event_table}

### Сводка по событиям

Macro average — простое среднее восьми fold-метрик, каждое мероприятие имеет одинаковый вес.

| Model | Macro precision | Macro recall | Macro F1 | Macro average precision |
|---|---:|---:|---:|---:|
{macro_table}

Pooled out-of-fold метрики объединяют все 2 161 test-предсказание (каждый проект предсказан только в fold своего hackathon).

| Model | Pooled precision | Pooled recall | Pooled F1 | Pooled average precision |
|---|---:|---:|---:|---:|
{pooled_table}

LogisticRegression превзошла Dummy по F1 в **{lr_f1_better}/8** events и по average precision в **{lr_ap_better}/8**. Диапазоны её fold F1: {lr_f1.min():.3f}–{lr_f1.max():.3f}; average precision: {lr_ap.min():.3f}–{lr_ap.max():.3f}. При пороге 0.5 модель отметила **{logistic_true_positives}/{EXPECTED_WINNERS}** winners, пропустила {logistic_false_negatives} и ошибочно отметила non-winners {logistic_false_positives} раз. Суммарно **{unseen_total}/{tag_stats['technology_tag_entries']} ({unseen_share:.1%})** project-tag entries не встречались в train vocabulary; это затронуло **{unseen_projects_total}/{EXPECTED_ROWS}** test-проектов. Неизвестные tags не добавлялись в indicators, но остаются учтены в count.

## Ошибки и коэффициенты

Ошибки — out-of-fold false positives/false negatives; ID и прогнозы сохраняются отдельно в `data/processed/technology_baseline_errors.csv` (локальная таблица под `/data/`, не исходные тексты). Counts сгруппированы ниже.

| Held-out event | Model | Error | Projects |
|---|---|---|---:|
{error_table}

Полная таблица fitted coefficients по fold лежит в `technology_baseline_coefficients.csv`. Здесь показаны средние коэффициенты технологий, встречавшихся минимум в 4 train vocabularies и имеющих support не менее 20 train projects в каждом таком fold. Порог нужен только для читаемости диагностики; он не меняет модель.

Средние наиболее положительные коэффициенты:

| Tag | Mean coefficient | Folds present | Min train support |
|---|---:|---:|---:|
{positive_table}

Средние наиболее отрицательные коэффициенты:

| Tag | Mean coefficient | Folds present | Min train support |
|---|---:|---:|---:|
{negative_table}

{count_coef_line} На train count математически равен сумме бинарных indicators; из-за этой зависимости коэффициенты не являются независимыми эффектами. Дополнительно tags коррелируют между собой и зависят от события. Знак/величина не доказывают, что технология повышает или снижает шанс победы. Convergence warnings: **{convergence}**.

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

Скрипт использует локальные `data/raw/devpost-hacks-{REVISION}.jsonl` и `data/processed/analytical_projects.csv`; ничего не скачивает. Результаты: `technology_baseline_results.csv`, `technology_baseline_coefficients.csv`, `technology_tag_summary.csv`, `technology_tag_screening.csv`; подробные per-project ошибки остаются в игнорируемой `data/processed/`.

**Версии запуска:** numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}.
'''


def main() -> int:
    data, source_info = load_data()
    tag_summary, tag_screen, tag_stats = summarize_tags(data)
    write_csv(TAG_SUMMARY_CSV, tag_summary,
              ["technology", "projects", "winner_projects", "non_winner_projects", "winner_prevalence", "non_winner_prevalence"])
    write_csv(TAG_SCREEN_CSV, tag_screen,
              ["suspicious_tag", "projects", "winner_projects", "non_winner_projects", "excluded_from_features"])

    events = sorted(data["hackathon"].astype(str).unique())
    if len(events) != 8:
        raise RuntimeError(f"Expected 8 labeled hackathons; got {len(events)}")
    coefficient_rows: list[dict] = []
    error_rows: list[dict] = []
    prediction_rows: list[dict] = []
    warning_rows: list[dict] = []
    result_rows = []
    for event in events:
        result_rows.extend(fit_one_fold(data, event, coefficient_rows, error_rows, prediction_rows, warning_rows))

    results = pd.DataFrame(result_rows)
    if len(results) != 16:
        raise RuntimeError(f"Expected 16 (8 events × 2 models) result rows; got {len(results)}")
    predictions = pd.DataFrame(prediction_rows)
    prediction_counts = predictions.groupby("project_id")["model"].nunique()
    expected_models = {"dummy_prior", "logistic_regression"}
    if (len(predictions) != 2 * EXPECTED_ROWS
            or len(prediction_counts) != EXPECTED_ROWS
            or not prediction_counts.eq(2).all()
            or set(predictions["model"]) != expected_models
            or predictions.duplicated(["project_id", "model"]).any()):
        raise RuntimeError("Expected one out-of-fold prediction per project and model")
    coefficients = pd.DataFrame(coefficient_rows)
    errors = pd.DataFrame(error_rows, columns=[
        "project_id", "heldout_hackathon", "model", "error_type", "actual_is_winner",
        "predicted_is_winner", "p_winner", "technology_count", "unseen_technology_tags",
    ])
    write_csv(RESULTS_CSV, result_rows, [
        "heldout_hackathon", "model", "n_train", "n_test", "test_winners", "test_non_winners",
        "winner_share", "training_winner_share", "train_technology_vocabulary",
        "test_projects_with_unseen_technology", "unseen_test_tag_entries", "threshold",
        "precision", "recall", "f1", "average_precision", "true_positive", "false_positive",
        "true_negative", "false_negative",
    ])
    write_csv(COEFFICIENTS_CSV, coefficient_rows, [
        "heldout_hackathon", "feature", "feature_type", "coefficient", "train_project_support",
        "training_projects", "training_vocabulary_size",
    ])
    write_csv(ERRORS_CSV, errors.to_dict(orient="records"), [
        "project_id", "heldout_hackathon", "model", "error_type", "actual_is_winner",
        "predicted_is_winner", "p_winner", "technology_count", "unseen_technology_tags",
    ])
    report = metric_report(results, predictions, data, tag_stats, warning_rows)
    report = report.replace("{REVISION}", REVISION)
    report = report.replace("{source_info['source_rows']}", str(source_info["source_rows"]))
    REPORT_MD.write_text(report, encoding="utf-8")
    print(json.dumps({
        "revision": REVISION,
        "source_rows": source_info["source_rows"],
        "projects": len(data),
        "winners": int(data["is_winner"].sum()),
        "non_winners": int(len(data) - data["is_winner"].sum()),
        "events": events,
        "unique_clean_technologies": tag_stats["unique_technology_tags"],
        "outcome_like_tags": tag_stats["outcome_like_tag_summary"],
        "logistic_f1_better_folds": int((results.pivot(index="heldout_hackathon", columns="model", values="f1")["logistic_regression"] > results.pivot(index="heldout_hackathon", columns="model", values="f1")["dummy_prior"]).sum()),
        "logistic_ap_better_folds": int((results.pivot(index="heldout_hackathon", columns="model", values="average_precision")["logistic_regression"] > results.pivot(index="heldout_hackathon", columns="model", values="average_precision")["dummy_prior"]).sum()),
        "error_rows": len(errors),
        "coefficient_rows": len(coefficients),
        "warnings": warning_rows,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "report": str(REPORT_MD),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
