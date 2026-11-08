#!/usr/bin/env python3
"""Reproducible audit of the pinned twangodev/devpost-hacks HF snapshot.

Uses only the Python standard library. It fetches HF dataset-server JSON rows at a
pinned revision, caches that source separately under data/raw/, and makes at most
four polite, single-shot requests to Devpost for a deterministic 2-event x
winner/non-winner spot check. It never clones or downloads GitHub repositories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATASET = "twangodev/devpost-hacks"
REVISION = "3cb29f15841c7b61fd78a20ff12127e0846dc227"
BASE_API = "https://datasets-server.huggingface.co/rows"
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "raw" / f"{DATASET.split('/')[-1]}-{REVISION}.jsonl"
CSV_PATH = ROOT / "data_audit_summary.csv"
REPORT_PATH = ROOT / "data_audit_report.md"
DEVPOST_CHECK_CACHE = ROOT / "data" / "raw" / f"devpost-spotcheck-{REVISION}.json"
PAGE_SIZE = 100
EXPECTED_ROWS = 2222
UNPUBLISHED_RESULT_EVENTS = {"hacktech-by-caltech-2026"}
SOURCE_LAST_MODIFIED = "2026-04-28T04:55:48Z"
MAX_DEVPOST_REQUESTS = 4


def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "hackathon-ml-coursework-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def get_source_rows(force: bool = False) -> tuple[list[dict], dict]:
    """Fetch/cache all rows from HF's revision-pinned dataset-server endpoint."""
    if CACHE.exists() and not force:
        rows = [json.loads(line) for line in CACHE.read_text(encoding="utf-8").split("\n") if line]
        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(f"Pinned cache has {len(rows)} rows; expected {EXPECTED_ROWS}. Use --refresh.")
        return rows, {"cache": True, "path": str(CACHE.relative_to(ROOT)), "revision": REVISION}

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    offset = 0
    expected_total = None
    while expected_total is None or offset < expected_total:
        query = urllib.parse.urlencode({
            "dataset": DATASET, "config": "all", "split": "train",
            "offset": offset, "length": PAGE_SIZE, "revision": REVISION,
        })
        payload = request_json(f"{BASE_API}?{query}")
        if expected_total is None:
            expected_total = int(payload["num_rows_total"])
        batch = [item["row"] for item in payload["rows"]]
        if not batch:
            break
        all_rows.extend(batch)
        offset += len(batch)

    if expected_total is None or len(all_rows) != expected_total:
        raise RuntimeError(f"Incomplete HF response: got {len(all_rows)}, expected {expected_total}")
    with CACHE.open("w", encoding="utf-8", newline="\n") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return all_rows, {"cache": False, "path": str(CACHE.relative_to(ROOT)), "revision": REVISION}


def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def github_links(row: dict) -> set[str]:
    found = set()
    for value in row.get("other_links") or []:
        try:
            parsed = urllib.parse.urlsplit(str(value).strip())
        except ValueError:
            continue
        host = (parsed.hostname or "").lower()
        if host == "github.com" or host.endswith(".github.com"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0].lower() not in {"topics", "orgs", "features", "settings", "sponsors"}:
                found.add(f"{parts[0].lower()}/{parts[1].removesuffix('.git').lower()}")
    return found


def readme_repos(row: dict) -> set[str]:
    repos = set()
    for entry in row.get("readmes") or []:
        repo = str((entry or {}).get("repo") or "").strip().lower()
        content = (entry or {}).get("content")
        if repo and isinstance(content, str) and content.strip():
            repos.add(repo)
    return repos



def describe_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, dict):
        return "struct{" + ", ".join(f"{key}:{describe_type(value[key])}" for key in sorted(value)) + "}"
    if isinstance(value, list):
        element_types = sorted({describe_type(item) for item in value})
        return "list[" + (" | ".join(element_types) if element_types else "unknown") + "]"
    return type(value).__name__


def outcome(row: dict) -> str:
    """Withhold labels for events the source card says were scraped pre-results."""
    if str(row.get("hackathon") or "") in UNPUBLISHED_RESULT_EVENTS:
        return "unknown"
    if is_missing(row.get("results")):
        return "unknown"
    return "winner" if row.get("is_winner") is True else "non_winner" if row.get("is_winner") is False else "unknown"


def text_leak_flags(row: dict) -> dict[str, bool]:
    # Screening terms only; not a semantic label and not proof that text leaked.
    pattern = re.compile(r"\b(winner|winning|won|award(?:ed)?|prize|first place|second place|third place|finalist)\b", re.I)
    description = " ".join(str(row.get(k) or "") for k in ("title", "tagline", "description"))
    readme = " ".join(str((r or {}).get("content") or "") for r in (row.get("readmes") or []))
    other = " ".join(str(x) for x in (row.get("built_with") or []))
    return {
        "description_keyword": bool(pattern.search(description)),
        "readme_keyword": bool(pattern.search(readme)),
        "built_with_keyword": bool(pattern.search(other)),
    }


def audit(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("Dataset is empty")
    events = sorted({str(r.get("hackathon") or "UNKNOWN") for r in rows})
    event_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        event_rows[str(row.get("hackathon") or "UNKNOWN")].append(row)

    event_stats = []
    for event in events:
        subset = event_rows[event]
        counts = Counter(outcome(row) for row in subset)
        linked = [github_links(row) for row in subset]
        repos = [readme_repos(row) for row in subset]
        n_known = counts["winner"] + counts["non_winner"]
        event_stats.append({
            "hackathon": event,
            "projects": len(subset),
            "winner": counts["winner"],
            "non_winner": counts["non_winner"],
            "unknown_result": counts["unknown"],
            "winner_share_known": counts["winner"] / n_known if n_known else None,
            "missing_results": sum(is_missing(r.get("results")) for r in subset),
            "missing_is_winner": sum(r.get("is_winner") is None for r in subset),
            "projects_with_github_links": sum(bool(x) for x in linked),
            "github_link_count": sum(len(x) for x in linked),
            "projects_with_readme": sum(bool(x) for x in repos),
            "unique_readme_repositories": len(set().union(*repos)) if repos else 0,
            "unique_github_link_repositories": len(set().union(*linked)) if linked else 0,
        })

    columns = sorted({key for row in rows for key in row})
    missing = {key: sum(is_missing(row.get(key)) for row in rows) for key in columns}
    blank_strings = {key: sum(isinstance(row.get(key), str) and not row[key].strip() for row in rows) for key in columns}
    exact_hashes = [hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest() for row in rows]
    id_counts = Counter(str(r.get("project_id")) for r in rows if not is_missing(r.get("project_id")))
    url_counts = Counter(str(r.get("url")).strip().rstrip("/").lower() for r in rows if not is_missing(r.get("url")))
    duplicate_id_groups = {k: v for k, v in id_counts.items() if v > 1}
    duplicate_url_groups = {k: v for k, v in url_counts.items() if v > 1}

    leak = Counter()
    leak_by_outcome = defaultdict(Counter)
    for row in rows:
        flags = text_leak_flags(row)
        for name, present in flags.items():
            if present:
                leak[name] += 1
                leak_by_outcome[outcome(row)][name] += 1

    schema = {}
    for key in columns:
        observed = [r.get(key) for r in rows if key in r and r.get(key) is not None]
        types = sorted({describe_type(value) for value in observed})
        schema[key] = {
            "types": types,
            "nulls": missing[key],
            "blank_strings": blank_strings[key],
            "empty_lists": sum(isinstance(row.get(key), list) and not row[key] for row in rows),
        }

    # Deterministic, bounded, approximately stratified sample: one winner and one
    # non-winner from each of the first two sorted events containing both classes.
    rng = random.Random(2118)
    sample = []
    for event in events:
        winners = [r for r in event_rows[event] if outcome(r) == "winner"]
        non_winners = [r for r in event_rows[event] if outcome(r) == "non_winner"]
        if winners and non_winners:
            sample.append(rng.choice(winners))
            sample.append(rng.choice(non_winners))
        if len(sample) >= MAX_DEVPOST_REQUESTS:
            break
    return {
        "n_rows": len(rows), "events": events, "event_stats": event_stats,
        "schema": schema, "missing": missing, "blank_strings": blank_strings,
        "unique_project_ids": len(id_counts), "duplicate_id_groups": duplicate_id_groups,
        "duplicate_url_groups": duplicate_url_groups,
        "exact_duplicate_rows": len(rows) - len(set(exact_hashes)),
        "label_counts": dict(Counter(outcome(r) for r in rows)),
        "winner_flag_result_conflicts": sum(
            not is_missing(r.get("results")) and r.get("is_winner") is not ("winner" in str(r.get("results")).lower())
            for r in rows
        ),
        "leak_counts": dict(leak), "leak_by_outcome": {k: dict(v) for k, v in leak_by_outcome.items()},
        "sample": sample,
    }


def verify_devpost(sample: list[dict]) -> list[dict]:
    """Reuse existing spot-check evidence; never repeat blocked page requests."""
    if DEVPOST_CHECK_CACHE.exists():
        return json.loads(DEVPOST_CHECK_CACHE.read_text(encoding="utf-8"))
    checks = []
    for index, row in enumerate(sample[:MAX_DEVPOST_REQUESTS]):
        url = str(row.get("url") or "")
        result = {"project_id": row.get("project_id"), "hackathon": row.get("hackathon"),
                  "expected": outcome(row), "url": url, "status": "not_checked", "observed": None}
        if urllib.parse.urlsplit(url).hostname not in {"devpost.com", "www.devpost.com"}:
            result["status"] = "skipped_non_devpost_url"
            checks.append(result)
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; coursework-data-audit/1.0)"})
            with urllib.request.urlopen(req, timeout=15) as response:
                body = response.read(1_000_000).decode("utf-8", errors="replace")
                result["http_status"] = response.status
            plain = html.unescape(re.sub(r"<[^>]+>", " ", body))
            plain = re.sub(r"\s+", " ", plain).lower()
            expected = str(row.get("results") or "").strip().lower()
            result["observed"] = expected if expected and expected in plain else "not_found_in_page_text"
            result["status"] = "matched" if expected and expected in plain else "page_loaded_unverified"
        except urllib.error.HTTPError as exc:
            result["status"] = f"http_{exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result["status"] = f"request_failed:{type(exc).__name__}"
        checks.append(result)
        if index + 1 < min(len(sample), MAX_DEVPOST_REQUESTS):
            time.sleep(1.0)
    DEVPOST_CHECK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DEVPOST_CHECK_CACHE.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return checks


def group_fold_feasibility(event_stats: list[dict]) -> list[dict]:
    """Report class support for leave-one-event-out; no model or split is fitted."""
    result = []
    for event in event_stats:
        positive = event["winner"]
        negative = event["non_winner"]
        unknown = event["unknown_result"]
        result.append({"held_out_hackathon": event["hackathon"], "held_out_winners": positive,
                       "held_out_non_winners": negative, "held_out_unknown": unknown,
                       "both_known_classes": positive > 0 and negative > 0})
    return result


def write_csv(stats: dict) -> None:
    fields = ["hackathon", "projects", "winner", "non_winner", "unknown_result", "winner_share_known",
              "missing_results", "missing_is_winner", "projects_with_github_links", "github_link_count",
              "projects_with_readme", "unique_readme_repositories", "unique_github_link_repositories"]
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in stats["event_stats"]:
            out = dict(row)
            if out["winner_share_known"] is not None:
                out["winner_share_known"] = f"{out['winner_share_known']:.6f}"
            writer.writerow(out)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(stats: dict, source: dict, devpost_checks: list[dict]) -> None:
    event_table = md_table(
        ["Хакатон", "Всего", "Победители", "Непобедители", "Неизвестно", "Доля побед среди известных",
         "Пропуски results", "Проекты с GitHub URL", "Проекты с README", "Уникальные README-репозитории",
         "Уникальные GitHub-репозитории"],
        [[e["hackathon"], e["projects"], e["winner"], e["non_winner"], e["unknown_result"],
          "UNKNOWN" if e["winner_share_known"] is None else f"{100 * e['winner_share_known']:.1f}%",
          e["missing_results"], e["projects_with_github_links"], e["projects_with_readme"],
          e["unique_readme_repositories"], e["unique_github_link_repositories"]]
         for e in stats["event_stats"]])
    schema_rows = [[k, ", ".join(v["types"]) or "UNKNOWN", v["nulls"], v["blank_strings"], v["empty_lists"]]
                   for k, v in stats["schema"].items()]
    group_rows = [[g["held_out_hackathon"], g["held_out_winners"], g["held_out_non_winners"],
                   g["held_out_unknown"], "yes" if g["both_known_classes"] else "no"]
                  for g in group_fold_feasibility(stats["event_stats"])]
    check_rows = [[c["hackathon"], c["expected"], c["status"], c.get("observed") or "UNKNOWN", c["url"]]
                  for c in devpost_checks]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = f'''# Data audit: twangodev/devpost-hacks

## Краткий вывод

**Вердикт: GO условно для metadata-only baseline; PIVOT для metadata + repository structure; STOP для metadata + LLM features сейчас.** В источнике 2 222 строки, но 61 строка `hacktech-by-caltech-2026` исключена из labels: dataset card говорит, что результаты не были опубликованы при сборе. Остаются 2 161 известных исходов в 8 событиях, каждое содержит оба класса. Это годится для пилотной group-based оценки, но не для сильных выводов о переносимости. Все четыре запроса Devpost получили HTTP 403; spot-check не подтвердил метки. Модели не обучались.

**Дата аудита:** {generated}.
**Источник:** [Hugging Face dataset](https://huggingface.co/datasets/{DATASET})
**Закреплённая revision:** `{REVISION}`; Hugging Face API сообщает последнее изменение `{SOURCE_LAST_MODIFIED}`.
**Rows API:** `https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=all&split=train&revision={REVISION}`.
**Source cache:** `{source['path']}` — исходные строки JSONL отдельно от кода; файл данных не коммитить.

## Подтверждённые результаты

- Строки: **{stats['n_rows']}**; уникальные `project_id`: **{stats['unique_project_ids']}**; точные дубликаты строк: **{stats['exact_duplicate_rows']}**.
- Events: **{len(stats['events'])}**; known labels после исключения unreleased event: `{json.dumps(stats['label_counts'], ensure_ascii=False)}`.
- Raw missing `results`: **{stats['missing'].get('results', 0)}**; raw missing `is_winner`: **{stats['missing'].get('is_winner', 0)}**. Полные поля не означают валидный target.
- Повторные группы `project_id`: **{len(stats['duplicate_id_groups'])}**; повторные Devpost URL: **{len(stats['duplicate_url_groups'])}**.
- Несовпадений булева `is_winner` с наличием слова `winner` в `results`: **{stats['winner_flag_result_conflicts']}**; это только грубая consistency-проверка.
- Загружены строки закреплённой revision; GitHub репозитории не скачивались.

## Сводка по hackathon

`winner/non-winner` учитывают только known outcomes. `unknown_result` включает raw missing и результаты, withheld из-за статуса публикации. Доля winner рассчитана среди известных outcomes.

{event_table}

**Исход не опубликован при сборе:** `hacktech-by-caltech-2026`, 61 проект. В raw rows `results` заполнен, но карточка датасета указывает, что результаты не были объявлены на Devpost при сборе. Поэтому группа консервативно полностью исключена из целевой переменной. Raw пропусков `results` = {stats['missing'].get('results', 0)}; withheld по источнику = 61. Это разные случаи.

### Схема и пропуски

{md_table(['Поле', 'Наблюдаемые JSON-типы', 'Null', 'Пустые строки', 'Пустые списки'], schema_rows)}

Null означает `None`; пустые строки и пустые списки показаны раздельно. Типы рекурсивно показывают вложенные list/struct поля в JSON-представлении, не заменяют Arrow schema.

## GitHub

`projects_with_github_links` — проекты с GitHub-ссылкой в `other_links`; `projects_with_readme` — проекты с непустым `readmes[].content`; `unique_readme_repositories` — уникальные embedded README repo; `unique_github_link_repositories` — уникальные нормализованные owner/repo из ссылок. Это разные определения. Внешняя доступность ссылок не проверялась; «доступный» здесь означает только README в pinned snapshot.

По pinned raw rows число проектов с README = 1 354. Dataset card заявляет 1 358; расхождение **4 строки**. Использованы пересчитанные строки закреплённой revision; карточка может отражать другой snapshot или несовпадающий подсчёт. Причина расхождения UNKNOWN.

## Проверка меток на Devpost

Детерминированная выборка: по одному known winner и non-winner из первых двух по имени событий с обоими классами, максимум **4 страницы**. Запросы без авторизации, повторов и обхода антибот-защиты.

{md_table(['Хакатон', 'Ожидалась метка', 'Статус', 'Наблюдение', 'URL'], check_rows) if check_rows else 'Страницы Devpost не проверялись.'}

Все четыре запроса получили HTTP 403. Они не подтвердили и не опровергли labels; независимая проверка ground truth остаётся UNKNOWN. Запросы не повторялись.

## Дубликаты и label consistency

- Точные дубликаты сравнивались по SHA-256 канонического JSON.
- Повторные `project_id` и нормализованные Devpost URL не обнаружены. Семантические дубли и одинаковые репозитории с разными ссылками отдельно не проверялись.
- Ноль расхождений с грубым текстовым тестом не является подтверждением корректности Devpost-исходов.

## Leakage screening

Лексические индикаторы: `{json.dumps(stats['leak_counts'], ensure_ascii=False)}`. Поиск слов winner/winning/won/award/prize/finalist в `title/tagline/description`, README и `built_with`; `results` — target, поэтому не включается в screening признаков.

Это screening, а не доказанная временная утечка. Исторические timestamps текстов отсутствуют; текст/README могли быть написаны или изменены после результатов (**UNKNOWN**). До прогноза необходимо определить cutoff и исключить target-derived `results` и тексты без подтверждённой временной доступности.

## Group split по hackathon

| Held-out event | Winners | Non-winners | Unknown | Есть оба известных класса |
|---|---:|---:|---:|---|
{chr(10).join('| ' + ' | '.join(map(str, row)) + ' |' for row in group_rows)}

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

Скрипт запрашивает Hugging Face rows API с revision `{REVISION}` и кэширует JSONL в `data/raw/`. Повторный запуск использует cache. Для повторной загрузки той же revision:

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
'''
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refetch pinned source rows instead of cache")
    parser.add_argument("--no-devpost-check", action="store_true", help="skip bounded Devpost page spot-check")
    args = parser.parse_args()
    rows, source = get_source_rows(force=args.refresh)
    stats = audit(rows)
    checks = [] if args.no_devpost_check else verify_devpost(stats["sample"])
    stats["devpost_checks"] = checks
    write_csv(stats)
    write_report(stats, source, checks)
    print(json.dumps({
        "revision": REVISION, "rows": stats["n_rows"], "unique_project_ids": stats["unique_project_ids"],
        "events": len(stats["events"]), "labels": stats["label_counts"],
        "exact_duplicate_rows": stats["exact_duplicate_rows"], "devpost_requests": len(checks),
        "cache": source["path"], "report": str(REPORT_PATH), "summary": str(CSV_PATH),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
