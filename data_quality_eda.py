#!/usr/bin/env python3
"""Pinned-snapshot EDA for the hackathon coursework; standard library only.

Reads the existing Data Audit JSONL cache (never modifies it), builds a local
row-level feature table without source text/URLs, emits aggregate CSVs and SVGs,
and performs at most one HEAD request per GitHub/video URL in a deterministic
stratified sample of 16 known-outcome projects. URL results are cached locally.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REVISION = "3cb29f15841c7b61fd78a20ff12127e0846dc227"
DATASET_URL = "https://huggingface.co/datasets/twangodev/devpost-hacks"
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "raw" / f"devpost-hacks-{REVISION}.jsonl"
LINK_CACHE = ROOT / "data" / "raw" / f"eda-link-checks-{REVISION}.json"
OUT = ROOT / "data" / "processed"
EXPECTED_SOURCE_ROWS = 2222
UNPUBLISHED_EVENTS = {"hacktech-by-caltech-2026"}
MAX_SAMPLE_PROJECTS = 100
SAMPLE_PER_CLASS_EVENT = 1  # 8 eligible events × 2 classes = at most 16 projects
REQUEST_DELAY_SECONDS = 1.0


def read_rows() -> list[dict]:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Pinned source cache missing: {SOURCE}; run data_audit.py first")
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").split("\n") if line]
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"Pinned cache has {len(rows)} rows; expected {EXPECTED_SOURCE_ROWS}")
    return rows


def outcome(row: dict) -> str | None:
    if row.get("hackathon") in UNPUBLISHED_EVENTS or not str(row.get("results") or "").strip():
        return None
    if row.get("is_winner") is True:
        return "winner"
    if row.get("is_winner") is False:
        return "non_winner"
    return None


def text(value) -> str:
    return value if isinstance(value, str) else ""


def normalized_github_repos(row: dict) -> set[str]:
    repos = set()
    for value in row.get("other_links") or []:
        try:
            parsed = urllib.parse.urlsplit(str(value).strip())
        except ValueError:
            continue
        host = (parsed.hostname or "").lower()
        if host != "github.com" and not host.endswith(".github.com"):
            continue
        if parsed.scheme not in {"http", "https"}:
            continue
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0].lower() not in {"topics", "orgs", "features", "settings", "sponsors"}:
            repos.add(f"{parts[0].lower()}/{parts[1].removesuffix('.git').lower()}")
    return repos


def readme_repos(row: dict) -> set[str]:
    return {
        str((item or {}).get("repo") or "").strip().lower()
        for item in (row.get("readmes") or [])
        if str((item or {}).get("repo") or "").strip()
        and text((item or {}).get("content")).strip()
    }


def normalized_technologies(row: dict) -> set[str]:
    return {str(value).strip().casefold() for value in (row.get("built_with") or []) if str(value).strip()}


def make_project(row: dict) -> dict:
    github_repos = normalized_github_repos(row)
    readmes = row.get("readmes") or []
    has_readme_record = bool(readmes)
    has_readme_content = bool(readme_repos(row))
    has_readme_any_content = any(text((item or {}).get("content")).strip() for item in readmes)
    has_readme_any_repo = any(str((item or {}).get("repo") or "").strip() for item in readmes)
    github_urls = [str(value).strip() for value in (row.get("other_links") or []) if is_github_url(value)]
    video_url = text(row.get("video_link")).strip()
    description_raw = row.get("description")
    tagline_raw = row.get("tagline")
    description_missing = description_raw is None or (isinstance(description_raw, str) and not description_raw.strip())
    tagline_missing = tagline_raw is None or (isinstance(tagline_raw, str) and not tagline_raw.strip())
    description = text(description_raw)
    tagline = text(tagline_raw)
    technologies = normalized_technologies(row)
    return {
        "project_id": str(row.get("project_id") or ""),
        "hackathon": str(row.get("hackathon") or ""),
        "outcome": outcome(row),
        "built_with_count": len(technologies),
        "description_chars": None if description_missing else len(description),
        "description_words": None if description_missing else len(description.split()),
        "tagline_chars": None if tagline_missing else len(tagline),
        "tagline_words": None if tagline_missing else len(tagline.split()),
        "github_url_present": bool(github_urls),
        "github_url_count": len(github_urls),
        "unique_github_repositories": len(github_repos),
        "readme_record_present": has_readme_record,
        "readme_any_content": has_readme_any_content,
        "readme_any_repo": has_readme_any_repo,
        "readme_content_present": has_readme_content,
        "unique_readme_repositories": len(readme_repos(row)),
        "video_link_present": bool(video_url),
        "github_readme_video_all": bool(github_urls and has_readme_content and video_url),
        "technologies": technologies,
        "github_urls": github_urls,
        "video_url": video_url,
        "source": row,
    }


def is_github_url(value) -> bool:
    try:
        parsed = urllib.parse.urlsplit(str(value).strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and (host == "github.com" or host.endswith(".github.com"))


def percentile(values: list[int | float | None], q: float):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def mean(values: list[int | float | None]):
    observed = [value for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], projects: list[dict]) -> dict:
    known = [item for item in projects if item["outcome"] is not None]
    events = sorted({item["hackathon"] for item in known})
    event_summary = []
    event_outcomes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in known:
        event_outcomes[item["hackathon"]][item["outcome"]].append(item)
    for event in events:
        subset = [p for p in known if p["hackathon"] == event]
        winners = event_outcomes[event]["winner"]
        non_winners = event_outcomes[event]["non_winner"]
        counts = Counter(p["outcome"] for p in subset)
        event_summary.append({
            "hackathon": event,
            "projects": len(subset),
            "winners": counts["winner"],
            "non_winners": counts["non_winner"],
            "winner_share": rate(counts["winner"], len(subset)),
            "github_url_projects": sum(p["github_url_present"] for p in subset),
            "github_url_share": rate(sum(p["github_url_present"] for p in subset), len(subset)),
            "readme_records": sum(p["readme_record_present"] for p in subset),
            "readme_content_projects": sum(p["readme_content_present"] for p in subset),
            "video_projects": sum(p["video_link_present"] for p in subset),
            "github_readme_projects": sum(p["github_url_present"] and p["readme_content_present"] for p in subset),
            "all_three_projects": sum(p["github_readme_video_all"] for p in subset),
            "mean_technology_count_winner": mean([p["built_with_count"] for p in winners]),
            "mean_technology_count_non_winner": mean([p["built_with_count"] for p in non_winners]),
            "median_description_chars_winner": percentile([p["description_chars"] for p in winners], .5),
            "median_description_chars_non_winner": percentile([p["description_chars"] for p in non_winners], .5),
            "median_tagline_chars_winner": percentile([p["tagline_chars"] for p in winners], .5),
            "median_tagline_chars_non_winner": percentile([p["tagline_chars"] for p in non_winners], .5),
        })

    technology_stats = Counter()
    technology_by_outcome: dict[str, Counter] = defaultdict(Counter)
    technology_counts_by_outcome: dict[str, int] = Counter()
    for project in known:
        technology_counts_by_outcome[project["outcome"]] += 1
        for technology in project["technologies"]:
            technology_stats[technology] += 1
            technology_by_outcome[project["outcome"]][technology] += 1
    technology_summary = []
    for technology, total in technology_stats.most_common():
        winner_n = technology_by_outcome["winner"][technology]
        non_winner_n = technology_by_outcome["non_winner"][technology]
        technology_summary.append({
            "technology": technology,
            "project_count": total,
            "winner_projects": winner_n,
            "non_winner_projects": non_winner_n,
            "winner_prevalence": rate(winner_n, technology_counts_by_outcome["winner"]),
            "non_winner_prevalence": rate(non_winner_n, technology_counts_by_outcome["non_winner"]),
            "prevalence_difference_winner_minus_non_winner": (
                rate(winner_n, technology_counts_by_outcome["winner"])
                - rate(non_winner_n, technology_counts_by_outcome["non_winner"])
            ),
        })

    intersection_counts = Counter()
    for project in known:
        key = (project["github_url_present"], project["readme_content_present"], project["video_link_present"])
        intersection_counts[(key, project["outcome"])] += 1
    intersections = []
    for key in sorted((github, readme, video) for github in (False, True)
                      for readme in (False, True) for video in (False, True)):
        winner_n = intersection_counts[(key, "winner")]
        non_winner_n = intersection_counts[(key, "non_winner")]
        intersections.append({
            "github_url_present": key[0], "readme_content_present": key[1], "video_link_present": key[2],
            "projects": winner_n + non_winner_n, "winners": winner_n, "non_winners": non_winner_n,
            "winner_share": rate(winner_n, winner_n + non_winner_n),
        })

    known_by_outcome = defaultdict(list)
    for project in known:
        known_by_outcome[project["outcome"]].append(project)
    text_summary = []
    for label in ("winner", "non_winner"):
        subset = known_by_outcome[label]
        for feature in ("description_chars", "description_words", "tagline_chars", "tagline_words"):
            values = [item[feature] for item in subset]
            observed_values = [value for value in values if value is not None]
            text_summary.append({
                "outcome": label, "measure": feature, "projects": len(values),
                "observed": len(observed_values), "missing": len(values) - len(observed_values),
                "mean": mean(values), "p50": percentile(values, .5), "p75": percentile(values, .75),
                "p90": percentile(values, .9), "maximum": max(observed_values) if observed_values else None,
            })

    github_readme = [p for p in known if p["github_url_present"] and p["readme_content_present"]]
    github_readme_by_event = []
    for event in events:
        subset = [p for p in github_readme if p["hackathon"] == event]
        counts = Counter(p["outcome"] for p in subset)
        github_readme_by_event.append({"hackathon": event, "projects": len(subset),
                                       "winners": counts["winner"], "non_winners": counts["non_winner"]})

    readme_all = [p for p in projects if p["readme_record_present"]]
    readme_nonempty = [p for p in projects if p["readme_content_present"]]
    empty_readme_records = [p for p in readme_all if not p["readme_content_present"]]
    readme_unusable_reasons = Counter(
        "no non-empty content" if not p["readme_any_content"]
        else "no repo identifier" if not p["readme_any_repo"]
        else "repo and content not paired in one entry"
        for p in empty_readme_records
    )
    readme_any_content_count = sum(p["readme_any_content"] for p in projects)
    readme_any_repo_count = sum(p["readme_any_repo"] for p in projects)
    github_row_total = sum(p["github_url_present"] for p in projects)
    technology_count_distribution = []
    tech_count_counts = {label: Counter(p["built_with_count"] for p in known_by_outcome[label])
                         for label in ("winner", "non_winner")}
    for k in sorted(set(tech_count_counts["winner"]) | set(tech_count_counts["non_winner"])):
        technology_count_distribution.append({
            "built_with_count": k,
            "winner_projects": tech_count_counts["winner"][k],
            "non_winner_projects": tech_count_counts["non_winner"][k],
        })

    return {
        "n_source": len(rows), "n_known": len(known), "n_unknown": len(rows) - len(known),
        "projects": known, "events": event_summary, "technology_summary": technology_summary,
        "intersections": intersections, "text_summary": text_summary,
        "github_readme": github_readme, "github_readme_by_event": github_readme_by_event,
        "github_projects_source_all": github_row_total,
        "readme_record_count_all": len(readme_all), "readme_content_count_all": len(readme_nonempty),
        "readme_any_content_count_all": readme_any_content_count,
        "readme_any_repo_count_all": readme_any_repo_count,
        "empty_readme_records_all": len(empty_readme_records),
        "readme_unusable_reasons": dict(readme_unusable_reasons),
        "readme_record_count_known": sum(p["readme_record_present"] for p in known),
        "readme_content_count_known": sum(p["readme_content_present"] for p in known),
        "technology_count_distribution": technology_count_distribution,
        "technology_by_outcome": technology_by_outcome,
        "technology_n_by_outcome": technology_counts_by_outcome,
        "known_by_outcome": known_by_outcome,
    }


def build_stratified_sample(projects: list[dict]) -> list[dict]:
    rng = random.Random(2118)
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for project in projects:
        by_stratum[(project["hackathon"], project["outcome"])].append(project)
    sample = []
    for event in sorted({key[0] for key in by_stratum}):
        for label in ("winner", "non_winner"):
            group = sorted(by_stratum[(event, label)], key=lambda p: p["project_id"])
            if group:
                sample.append(rng.choice(group))
    if len(sample) > MAX_SAMPLE_PROJECTS:
        raise RuntimeError("Stratified sample exceeds the 100-project cap")
    return sample


def host_category(url: str, link_type: str) -> str:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if link_type == "github":
        return "github.com"
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        return "youtube"
    if host in {"vimeo.com", "www.vimeo.com", "player.vimeo.com"}:
        return "vimeo"
    return host or "unknown-host"


def check_one_url(url: str, link_type: str, project: dict) -> dict:
    parsed = urllib.parse.urlsplit(url)
    record = {"project_id": project["project_id"], "hackathon": project["hackathon"],
              "outcome": project["outcome"], "link_type": link_type,
              "host_category": host_category(url, link_type), "status": "not_checked"}
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        record["status"] = "invalid_url"
        return record
    req = urllib.request.Request(url, method="HEAD", headers={
        "User-Agent": "hackathon-ml-coursework-eda/1.0",
        "Accept": "*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            record["status"] = f"http_{response.status}"
            record["final_host_category"] = host_category(response.geturl(), link_type)
            response.close()
    except urllib.error.HTTPError as exc:
        record["status"] = f"http_{exc.code}"
        exc.close()
    except urllib.error.URLError as exc:
        record["status"] = f"network_error:{type(exc.reason).__name__}"
    except (TimeoutError, OSError) as exc:
        record["status"] = f"network_error:{type(exc).__name__}"
    return record


def perform_link_checks(projects: list[dict], refresh: bool, skip: bool) -> tuple[list[dict], int]:
    if skip:
        return [], 0
    if LINK_CACHE.exists() and not refresh:
        cached = json.loads(LINK_CACHE.read_text(encoding="utf-8"))
        return cached["checks"], int(cached["sample_projects"])
    sample = build_stratified_sample(projects)
    checks = []
    for project in sample:
        urls = []
        if project["github_urls"]:
            urls.append(("github", project["github_urls"][0]))
        if project["video_url"]:
            urls.append(("video", project["video_url"]))
        for link_type, url in urls:
            if checks:
                time.sleep(REQUEST_DELAY_SECONDS)
            checks.append(check_one_url(url, link_type, project))
    LINK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LINK_CACHE.write_text(json.dumps({"revision": REVISION, "sample_projects": len(sample), "checks": checks},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    return checks, len(sample)


def summarize_link_checks(checks: list[dict]) -> list[dict]:
    counts = Counter((item["link_type"], item["host_category"], item["status"]) for item in checks)
    return [{"link_type": kind, "host_category": host, "status": status, "checks": count}
            for (kind, host, status), count in sorted(counts.items())]


def feature_inventory(summary: dict) -> list[dict]:
    known_n = summary["n_known"]
    known_projects = summary["projects"]
    missing_built_with = sum(project["source"].get("built_with") is None for project in known_projects)
    missing_description = sum(project["description_chars"] is None for project in known_projects)
    missing_tagline = sum(project["tagline_chars"] is None for project in known_projects)
    pattern = re.compile(r"\b(winner|winning|won|award(?:ed)?|prize|first place|second place|third place|finalist)\b", re.I)
    lexical_description = sum(
        bool(pattern.search(" ".join(text(p["source"].get(key)) for key in ("title", "tagline", "description"))))
        for p in summary["projects"]
    )
    lexical_readme = sum(
        any(pattern.search(text((item or {}).get("content"))) for item in (p["source"].get("readmes") or []))
        for p in summary["projects"]
    )
    fields = [
        ("project_id", "Источник: project_id", "string/id", 0, "Идентификатор не outcome-derived, но может запомнить строку", "Доступен в snapshot; до события неизвестно", "Исключить", "Только ключ; не predictor."),
        ("hackathon", "Источник: hackathon slug", "categorical/group", 0, "Идентичность события может proxy-ить правила/судейство", "Event-level известен, но использовать для Group split", "Только группировка", "Не включать как predictor при оценке новых событий."),
        ("outcome", "is_winner с учетом результатов карточки", "binary target", 0, "Целевая метка; label audit не независим", "После события; не признак", "Только target", "61 HackTech 2026 исключены как unknown."),
        ("built_with_count", "Число уникальных casefold built_with tags", "integer", missing_built_with, "Прямой lexical leakage не ищется; возможен event confounding", "UNKNOWN: submission timestamp отсутствует", "Условно, только retrospective", "Для pre-end prediction недопустим до подтверждения cutoff."),
        ("technology_tags", "Бинарные indicators по built_with", "multi-hot categorical", missing_built_with, "Связи могут быть event/track-specific", "UNKNOWN: timestamp отсутствует", "Условно, только retrospective", "Число редких тегов и малые n осложняют сравнение."),
        ("description_chars/words", "Длина Devpost description", "integer", missing_description, f"{lexical_description}/{known_n} known rows содержат outcome-like terms в title/tagline/description", "UNKNOWN: текст мог быть заполнен/обновлён после объявления", "Исключить из strict baseline пока нет cutoff", "Не применять NLP до temporal audit."),
        ("tagline_chars/words", "Длина Devpost tagline", "integer", missing_tagline, "Текст может содержать outcome claims; частота отдельно от description не подменяется", "UNKNOWN: timestamp отсутствует", "Исключить из strict baseline пока нет cutoff", "Retrospective ассоциация возможна с явной оговоркой."),
        ("github_url_present", "Есть GitHub URL в other_links", "boolean", 0, "Ссылка может быть добавлена после события/награды", "UNKNOWN: дата добавления не записана", "Исключить из strict baseline", "Текущая HTTP доступность не равна исходному наличию."),
        ("readme_content_present", "Есть непустой embedded README в snapshot", "boolean", 0, f"{lexical_readme}/{known_n} known rows имеют README с award/win terms", "UNKNOWN; README обычно связан с репозиторием и мог меняться позже", "Исключить из strict baseline", "Наличие README не означает его pre-event snapshot."),
        ("video_link_present", "video_link непустой", "boolean", 0, "Видео/ссылка может быть оформлена при финальной submission", "UNKNOWN: дата добавления отсутствует", "Исключить из strict baseline", "Текущая доступность не использовать как предиктор."),
        ("github_readme_video_all", "AND трех availability flags", "boolean", 0, "Композит включает post-submission signals", "UNKNOWN", "Исключить из strict baseline", "Использовать только в EDA/selection bias анализе."),
        ("results/is_winner raw fields", "Поля источника", "string/bool", 0, "Target identity/direct leakage", "После события", "Исключить из X", "Сохраняются только для формирования target."),
        ("current_link_http_status", "Ограниченный HEAD check текущих ссылок", "categorical/status", None, "Future/volatile external state", "После snapshot, время проверки now", "Исключить", "Только data quality metric; не модельный признак."),
    ]
    return [{"feature": name, "extraction": extraction, "dtype": dtype,
             "missing_share_known": "UNKNOWN" if missing is None else f"{missing / known_n:.6f}",
             "possible_outcome_leakage": leakage, "known_available_before_hackathon_end": availability,
             "first_baseline": allowed, "notes": notes}
            for name, extraction, dtype, missing, leakage, availability, allowed, notes in fields]


def analytical_rows(projects: list[dict]) -> list[dict]:
    fields = ["project_id", "hackathon", "outcome", "built_with_count", "description_chars", "description_words",
              "tagline_chars", "tagline_words", "github_url_present", "github_url_count",
              "unique_github_repositories", "readme_record_present", "readme_content_present",
              "unique_readme_repositories", "video_link_present", "github_readme_video_all"]
    return [{key: (int(project[key]) if isinstance(project[key], bool) else project[key]) for key in fields}
            for project in projects]


def svg_text(x: float, y: float, value: str, cls: str = "", size: int = 14, anchor: str = "start") -> str:
    safe = html.escape(str(value), quote=True)
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}">{safe}</text>'


def write_svg(path: Path, width: int, height: int, title: str, body: str) -> None:
    style = "text{font-family:system-ui,sans-serif;fill:#111827;paint-order:stroke;stroke:#fff;stroke-width:2px;stroke-linejoin:round}.muted{fill:#64748b}.winner{fill:#2563eb}.nonwinner{fill:#d97706}.axis{stroke:#94a3b8;stroke-width:1}.track{fill:#cbd5e1;opacity:.45}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">'
           f'<style>{style}</style>{svg_text(24, 30, title, size=20)}{body}</svg>')
    path.write_text(svg, encoding="utf-8")


def chart_events(events: list[dict]) -> None:
    width, height, x0, chart_w, row_h = 900, 430, 260, 440, 38
    body = [svg_text(x0, 60, "Winner share among known outcomes", "muted", 13)]
    for i, row in enumerate(events):
        y = 88 + i * row_h
        share = row["winner_share"] or 0
        body.append(svg_text(x0 - 14, y + 5, row["hackathon"], "", 13, "end"))
        body.append(f'<rect x="{x0}" y="{y - 10}" width="{chart_w}" height="17" rx="4" class="track"/>')
        body.append(f'<rect x="{x0}" y="{y - 10}" width="{chart_w * share / .35:.1f}" height="17" rx="4" class="winner"/>')
        body.append(svg_text(x0 + chart_w + 12, y + 5, f"{share:.1%}  ({row['winners']}/{row['projects']})", "", 13))
    body.append(svg_text(x0, height - 20, "HackTech 2026 omitted: results unpublished when collected.", "muted", 12))
    write_svg(ROOT / "eda_event_winner_share.svg", width, height, "Known winner share varies across hackathons", "".join(body))


def chart_technologies(technology_summary: list[dict], total_known: int) -> None:
    top = technology_summary[:12]
    width, height, x0, chart_w = 980, 130 + 38 * len(top), 250, 420
    body = [svg_text(x0, 60, "Project prevalence by known outcome", "muted", 13),
            svg_text(x0, 82, "Winner", "winner", 12), svg_text(x0 + 100, 82, "Non-winner", "nonwinner", 12)]
    for i, row in enumerate(top):
        y = 110 + i * 38
        win_rate, non_rate = row["winner_prevalence"], row["non_winner_prevalence"]
        body.append(svg_text(x0 - 14, y + 7, row["technology"], "", 13, "end"))
        body.append(f'<rect x="{x0}" y="{y - 7}" width="{chart_w * win_rate / .8:.1f}" height="11" rx="3" class="winner"/>')
        body.append(f'<rect x="{x0}" y="{y + 7}" width="{chart_w * non_rate / .8:.1f}" height="11" rx="3" class="nonwinner"/>')
        body.append(svg_text(x0 + chart_w + 10, y + 3, f"W {win_rate:.1%}, n={row['winner_projects']}", "muted", 11))
        body.append(svg_text(x0 + chart_w + 10, y + 17, f"NW {non_rate:.1%}, n={row['non_winner_projects']}", "muted", 11))
    body.append(svg_text(x0, height - 14, f"Top tags by total projects; n={total_known}.", "muted", 12))
    write_svg(ROOT / "eda_top_technologies.svg", width, height, "Technology tags are uneven across classes", "".join(body))


def chart_availability(intersections: list[dict]) -> None:
    ordered = sorted(intersections, key=lambda r: (r["projects"], r["github_url_present"], r["readme_content_present"], r["video_link_present"]))
    width, height, x0, chart_w, row_h = 940, 80 + 35 * len(ordered), 270, 460, 35
    body = [svg_text(x0, 60, "Projects per URL/README/video combination", "muted", 13)]
    for i, row in enumerate(ordered):
        label = f"GitHub {'Y' if row['github_url_present'] else 'N'} / README {'Y' if row['readme_content_present'] else 'N'} / video {'Y' if row['video_link_present'] else 'N'}"
        y = 91 + i * row_h
        body.append(svg_text(x0 - 14, y + 5, label, "", 13, "end"))
        body.append(f'<rect x="{x0}" y="{y - 9}" width="{chart_w * row["projects"] / max(1, max(x["projects"] for x in ordered)):.1f}" height="18" rx="4" class="track"/>')
        body.append(svg_text(x0 + chart_w + 12, y + 5, f"{row['projects']} (W {row['winners']}, NW {row['non_winners']})", "", 12))
    write_svg(ROOT / "eda_link_coverage.svg", width, height, "Link and documentation coverage intersection", "".join(body))


def chart_lengths(text_summary: list[dict]) -> None:
    data = [row for row in text_summary if row["measure"] in {"description_chars", "tagline_chars"}]
    width, height, x0, chart_w = 900, 250, 250, 450
    maxval = max((row["p50"] or 0 for row in data), default=1)
    body = [svg_text(x0, 60, "Median characters; bars share a common scale", "muted", 13)]
    for i, row in enumerate(data):
        y = 100 + i * 42
        label = f"{row['measure'].replace('_chars', '')} / {row['outcome']}"
        barcls = "winner" if row["outcome"] == "winner" else "nonwinner"
        body.append(svg_text(x0 - 14, y + 5, label, "", 13, "end"))
        body.append(f'<rect x="{x0}" y="{y - 9}" width="{chart_w * (row["p50"] or 0) / maxval:.1f}" height="18" rx="4" class="{barcls}"/>')
        body.append(svg_text(x0 + chart_w + 12, y + 5, f"p50={row['p50']}  p90={row['p90']}  observed={row['observed']}", "", 12))


    write_svg(ROOT / "eda_text_lengths.svg", width, height, "Median description and tagline lengths by outcome", "".join(body))

def report_text(summary: dict, checks: list[dict], sample_n: int) -> str:
    n = summary["n_known"]
    winner_n = len(summary["known_by_outcome"]["winner"])
    non_n = len(summary["known_by_outcome"]["non_winner"])
    technology_summary = summary["technology_summary"]
    top_tech = technology_summary[:10]
    top_table = "\n".join(
        f"| {r['technology']} | {r['project_count']} | {r['winner_projects']} | {r['non_winner_projects']} | "
        f"{r['winner_prevalence']:.1%} | {r['non_winner_prevalence']:.1%} |"
        for r in top_tech)
    event_table = "\n".join(
        f"| {r['hackathon']} | {r['projects']} | {r['winners']} | {r['non_winners']} | {r['winner_share']:.1%} | "
        f"{r['github_url_projects']} | {r['readme_content_projects']} | {r['video_projects']} | {r['all_three_projects']} |"
        for r in summary["events"])
    intersection_table = "\n".join(
        f"| {'Y' if r['github_url_present'] else 'N'} | {'Y' if r['readme_content_present'] else 'N'} | "
        f"{'Y' if r['video_link_present'] else 'N'} | {r['projects']} | {r['winners']} | {r['non_winners']} |"
        for r in summary["intersections"])
    link_counts = Counter((c["link_type"], c["status"]) for c in checks)
    link_table = "\n".join(f"| {kind} | {status} | {count} |" for (kind, status), count in sorted(link_counts.items())) or "| UNKNOWN | skipped | 0 |"
    github_readme_count = len(summary["github_readme"])
    gr_winners = sum(p["outcome"] == "winner" for p in summary["github_readme"])
    gr_non = github_readme_count - gr_winners
    wshare = rate(winner_n, n)
    github_n = sum(r["github_url_projects"] for r in summary["events"])
    readme_n = sum(r["readme_content_projects"] for r in summary["events"])
    video_n = sum(r["video_projects"] for r in summary["events"])
    both_three = sum(r["all_three_projects"] for r in summary["events"])
    desc_row = {(r["outcome"], r["measure"]): r for r in summary["text_summary"]}
    d_w = desc_row[("winner", "description_chars")]
    d_n = desc_row[("non_winner", "description_chars")]
    t_w = desc_row[("winner", "tagline_chars")]
    t_n = desc_row[("non_winner", "tagline_chars")]
    screening_pattern = re.compile(r"\b(winner|winning|won|award(?:ed)?|prize|first place|second place|third place|finalist)\b", re.I)
    description_keyword_count = sum(
        bool(screening_pattern.search(" ".join(text(p["source"].get(key)) for key in ("title", "tagline", "description"))))
        for p in summary["projects"])
    readme_keyword_count = sum(
        any(screening_pattern.search(text((item or {}).get("content"))) for item in (p["source"].get("readmes") or []))
        for p in summary["projects"]
    )
    readme_reasons = summary["readme_unusable_reasons"]
    readme_reason_table = "; ".join(f"{reason}: {count}" for reason, count in sorted(readme_reasons.items())) or "none"
    desc_missing = sum(p["description_chars"] is None for p in summary["projects"])
    tagline_missing = sum(p["tagline_chars"] is None for p in summary["projects"])
    event_deltas = sorted(summary["events"], key=lambda r: r["winner_share"])
    return f'''# Data Quality Enrichment + EDA: Hackathon ML Research

**Дата:** {datetime.now(timezone.utc).isoformat(timespec="seconds")}<br>
**Источник:** [twangodev/devpost-hacks](https://huggingface.co/datasets/twangodev/devpost-hacks), pinned revision `{REVISION}`.<br>
**Unit:** один Devpost project; исходные строки не изменялись. HackTech 2026 исключён из target, потому что источник сообщает, что результатов ещё не было при сборе.

## Ключевые числа

- В pinned cache: **{summary['n_source']}** строк; аналитическая таблица с known outcomes: **{n}** проектов = **{winner_n} winners + {non_n} non-winners**; **{summary['n_unknown']}** withheld/unknown строк, в том числе 61 HackTech 2026.
- README: raw `readmes` record присутствует у **{summary['readme_record_count_all']}** строк; usable embedded README (repo identifier и непустой content в одной записи) — у **{summary['readme_content_count_all']}**. **[CONFIRMED]** Четыре raw records не проходят этот usability filter; breakdown: {readme_reason_table}. **[INFERRED]** Совпадение 1 358 raw records с числом dataset card правдоподобно объясняет 1 358 против 1 354; точное правило card **[UNKNOWN]**.
- В known-outcome таблице: GitHub URL в `other_links` **{github_n}/{n} ({rate(github_n,n):.1%})**; непустой README **{readme_n}/{n} ({rate(readme_n,n):.1%})**; video URL **{video_n}/{n} ({rate(video_n,n):.1%})**; все три одновременно **{both_three}/{n} ({rate(both_three,n):.1%})**.
- Подвыборка GitHub URL + непустое README: **{github_readme_count}** projects (**{gr_winners} winner / {gr_non} non-winner**; winner share {rate(gr_winners,github_readme_count):.1%}). Это кандидаты на дальнейший analysis subset, но доступность GitHub и пред-event timing не доказываются наличием ссылки/README.

## README discrepancy

**[CONFIRMED]** В pinned cache 1 358 строк имеют непустой `readmes` list, а 1 354 — хотя бы одну запись одновременно с repo identifier и непустым README content; разница — 4. Невалидные records распределены так: {readme_reason_table}. **[INFERRED]** Dataset card, вероятно, считает raw records, но его точное правило **[UNKNOWN]**; исходные данные не менялись.

## Результаты по событиям

Доли рассчитаны только среди known outcomes. Различия между мероприятиями — описательные, не причинные.

| Hackathon | N | Winners | Non-winners | Winner share | GitHub URL | README content | Video URL | All three |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{event_table}

Самая низкая/высокая доля winner среди учтённых events: `{event_deltas[0]['hackathon']}` ({event_deltas[0]['winner_share']:.1%}) и `{event_deltas[-1]['hackathon']}` ({event_deltas[-1]['winner_share']:.1%}). Технические теги и ссылки также различаются между events; это может отражать разные задачи, правила, submission patterns и состав участников.

![Доли winners по событиям](eda_event_winner_share.svg)

## Technologies и текстовая длина

Технологии нормализованы через `strip().casefold()` и считаются не более одного раза на проект. Распространённость — доля проектов в каждом классе, не доля среди всех тегов.

| Technology | Projects | Winners using | Non-winners using | Winner prevalence | Non-winner prevalence |
|---|---:|---:|---:|---:|---:|
{top_table}

Среднее количество `built_with` тегов: winners **{mean([p['built_with_count'] for p in summary['known_by_outcome']['winner']]):.2f}**, non-winners **{mean([p['built_with_count'] for p in summary['known_by_outcome']['non_winner']]):.2f}**; подробное распределение лежит в `eda_technology_count_distribution.csv`.

![Распространённость частых технологий](eda_top_technologies.svg)

Длины измерены в Unicode characters и whitespace-delimited words; это не semantic quality. Median description length: winners **{d_w['p50']} chars** (p90 {d_w['p90']}), non-winners **{d_n['p50']} chars** (p90 {d_n['p90']}); median tagline: winners **{t_w['p50']}**, non-winners **{t_n['p50']} chars**. Missing/blank values исключены из статистики: description **{desc_missing}/{n}**, tagline **{tagline_missing}/{n}**. Распределения асимметричны; mean/median не следует интерпретировать как качество.

![Длины описаний и tagline](eda_text_lengths.svg)

## GitHub, README, video и limited link check

Presence в snapshot отличается от текущей HTTP-доступности. В known subset: GitHub URL {github_n}; непустой README {readme_n}; video URL {video_n}. Таблица ниже показывает пересечение всех восьми комбинаций.

| GitHub URL | README content | Video URL | Projects | Winners | Non-winners |
|---|---|---|---:|---:|---:|
{intersection_table}

![Пересечение ссылок и материалов](eda_link_coverage.svg)

Limited check: детерминированно выбрано **{sample_n}** projects (по одному winner и non-winner на каждый из восьми eligible events); для каждого было проверено не более одной GitHub URL и одной video URL — **{len(checks)}** однократных `HEAD` запросов максимум. Пауза {REQUEST_DELAY_SECONDS:.1f}s, timeout 15s, без retries, авторизации, обхода anti-bot, repository cloning или video download. Link status cache хранит individual project IDs/URLs только под `data/raw/` (ignored, не публиковать).

| Link type | HTTP/network status | Checks |
|---|---|---:|
{link_table}

`403` — доступ запрещён; `405` — HEAD не поддержан; `429` — rate-limited; ошибки сети/timeout — ресурс не классифицирован. Никакие из этих ответов не меняют snapshot presence. HEAD `2xx` говорит только, что URL отвечал на момент проверки, а не что repo содержимое качественно/доступно для структурного парсинга. UNKNOWN statuses не трактуются как dead URLs.

## Потенциальные признаки и leakage

Лексический screening title/tagline/description выявил award/winner-подобные слова у **{description_keyword_count}/{n}** проектов, а embedded README — у **{readme_keyword_count}/{n}**. Это индикаторы ручного контроля, не label verification и не доказательство утечки. `results` и `is_winner` — target/source of label, не X.

| Сигнал | Момент доступности | Для pre-end baseline? |
|---|---|---|
| `project_id` | Snapshot; это ключ | Нет: только identifier |
| `hackathon` | Известно как group ID | Нет как predictor; использовать для group split |
| `built_with` / count / one-hot | Timestamp отсутствует | Пока нет; только retrospective EDA |
| Description/tagline lengths/text | Timestamp отсутствует; возможны post-event edits | Нет до cutoff audit; prose содержит outcome-style terms |
| GitHub URL / README / video presence | Timestamp отсутствует; submission material мог быть добавлен после результата | Нет до подтверждения temporal availability |
| `results` / `is_winner` | Outcome поле | Target only; исключить из X |
| Текущий HTTP status | Измерен после snapshot | Нет; volatile future state |

Полный feature inventory с dtype/missing/leakage/disposition — `eda_feature_inventory.csv`. По структуре dataset нет явного timestamp на поля проекта, поэтому из этой выгрузки нельзя доказать доступность feature до окончания хакатона. Любой baseline здесь должен быть помечен как retrospective association/prediction-on-snapshot, если не будет найден point-in-time snapshot.

## Структурная подвыборка и решение

GitHub URL + непустой embedded README дает **{github_readme_count}** known projects: {gr_winners} winners и {gr_non} non-winners, по событиям распределено следующим образом (таблица `eda_github_readme_by_event.csv`). Объём выглядит достаточным для следующего **feature-extraction pilot** без массовой загрузки репозиториев, но частичный README coverage может быть selection-biased. Перед сравнениями A vs A+B нужно сравнивать A на том же subset, зафиксировать критерии отбора и проверить local/pinned README usability. `available README` не подтверждает, что связанный GitHub repository всё ещё публичен.

## Ограничения и неизвестное

- Независимая outcome spot-check остаётся неподтверждённой: предыдущая ограниченная проверка Devpost получила HTTP 403.
- HTTP check в данном этапе ограничен {len(checks)} URLs; он не оценивает общую долю доступных repos/videos.
- Нет timestamps, judging criteria, contest-level metadata/prize pool, temporal snapshots и стабильного мерила качества.
- Event differences, prevalence и text length — descriptive; никаких причинных выводов.
- Модели не обучались.

## Воспроизведение

Python 3.10+; стандартная библиотека, без установки зависимостей.

```bash
python data_quality_eda.py
```

Скрипт использует existing pinned cache из `data/raw/`, пишет row-level table без описаний/URLs в ignored `data/processed/`, агрегаты и SVG-графики в корень. External link checks ограничены 16 projects/максимум 32 URL, результаты кэшируются под ignored `data/raw/`. `--refresh-link-checks` повторяет эту ограниченную проверку; `--skip-link-checks` пропускает сеть. Исходный cache не меняется.
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-link-checks", action="store_true", help="repeat bounded HEAD checks for the same 16-project sample")
    parser.add_argument("--skip-link-checks", action="store_true", help="do not send external HTTP requests")
    args = parser.parse_args()
    raw = read_rows()
    projects_all = [make_project(row) for row in raw]
    project_ids = [project["project_id"] for project in projects_all]
    if any(not project_id for project_id in project_ids) or len(project_ids) != len(set(project_ids)):
        raise RuntimeError("Source does not contain one unique non-empty project_id per row")
    projects = [project for project in projects_all if project["outcome"] is not None]
    summary = summarize(raw, projects_all)
    if len(projects) != 2161:
        raise RuntimeError(f"Known-outcome analytical subset has {len(projects)} rows; expected 2161")

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["project_id", "hackathon", "outcome", "built_with_count", "description_chars", "description_words",
              "tagline_chars", "tagline_words", "github_url_present", "github_url_count", "unique_github_repositories",
              "readme_record_present", "readme_content_present", "unique_readme_repositories", "video_link_present",
              "github_readme_video_all"]
    write_csv(OUT / "analytical_projects.csv", analytical_rows(projects), fields)

    event_fields = list(summary["events"][0])
    write_csv(ROOT / "eda_event_summary.csv", summary["events"], event_fields)
    tech_fields = list(summary["technology_summary"][0])
    write_csv(ROOT / "eda_technology_summary.csv", summary["technology_summary"], tech_fields)
    write_csv(ROOT / "eda_technology_count_distribution.csv", summary["technology_count_distribution"],
              ["built_with_count", "winner_projects", "non_winner_projects"])
    write_csv(ROOT / "eda_availability_intersections.csv", summary["intersections"],
              ["github_url_present", "readme_content_present", "video_link_present", "projects", "winners", "non_winners", "winner_share"])
    write_csv(ROOT / "eda_text_length_summary.csv", summary["text_summary"],
              ["outcome", "measure", "projects", "observed", "missing", "mean", "p50", "p75", "p90", "maximum"])
    write_csv(ROOT / "eda_github_readme_by_event.csv", summary["github_readme_by_event"],
              ["hackathon", "projects", "winners", "non_winners"])

    checks, sample_n = perform_link_checks(projects, args.refresh_link_checks, args.skip_link_checks)
    write_csv(ROOT / "eda_link_check_summary.csv", summarize_link_checks(checks),
              ["link_type", "host_category", "status", "checks"])
    write_csv(ROOT / "eda_feature_inventory.csv", feature_inventory(summary),
              ["feature", "extraction", "dtype", "missing_share_known", "possible_outcome_leakage",
               "known_available_before_hackathon_end", "first_baseline", "notes"])

    chart_events(summary["events"])
    chart_technologies(summary["technology_summary"], summary["n_known"])
    chart_availability(summary["intersections"])
    chart_lengths(summary["text_summary"])
    (ROOT / "data_quality_eda_report.md").write_text(report_text(summary, checks, sample_n), encoding="utf-8")
    print(json.dumps({
        "revision": REVISION, "source_rows": len(raw), "analytical_projects": len(projects),
        "winners": sum(p["outcome"] == "winner" for p in projects),
        "non_winners": sum(p["outcome"] == "non_winner" for p in projects),
        "events": len(summary["events"]), "github_readme_projects": len(summary["github_readme"]),
        "readme_records_all": summary["readme_record_count_all"],
        "readme_content_all": summary["readme_content_count_all"],
        "empty_readme_records_all": summary["empty_readme_records_all"],
        "link_sample_projects": sample_n, "link_checks": len(checks),
        "report": str(ROOT / "data_quality_eda_report.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
