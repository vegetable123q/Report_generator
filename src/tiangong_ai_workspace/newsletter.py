"""
Automated newsletter generator for regulation updates based on the parsed CSV export.

The generator builds a clustered stacked bar chart and a policy table, then fills a
markdown scaffold so downstream agents can distribute the newsletter.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"


@dataclass(slots=True)
class NewsletterConfig:
    """Configuration for generating the newsletter."""

    csv_path: Path
    output_dir: Path
    max_policies: int = 12
    first_run: bool = False
    ai_emphasis: bool = True
    max_implementation_year: int = 2025
    max_implementation_month: int = 12
    export_docx: bool = True


def generate_newsletter(config: NewsletterConfig) -> Mapping[str, str]:
    """
    Generate the newsletter markdown and chart from the provided CSV.

    Returns a mapping with `markdown_path` and `chart_path`.
    """

    _ensure_matplotlib_cache(config.output_dir)
    entries = _load_entries(config.csv_path)
    _apply_balanced_impact_labels(entries)
    _annotate_csv_with_impact(config.csv_path)
    chart_path = _render_chart(entries, config.output_dir, first_run=config.first_run)
    cutoff = _implementation_cutoff(config.max_implementation_year, config.max_implementation_month)
    policy_table = _build_policy_table(
        entries,
        max_rows=config.max_policies,
        ai_emphasis=config.ai_emphasis,
        implementation_cutoff=cutoff,
    )
    insights = list(_build_insights(entries, first_run=config.first_run))

    template_path = TEMPLATES_ROOT / "newsletter.md"
    template = template_path.read_text(encoding="utf-8")

    impact_rules_text = "\n".join(
        [
            "- `Low`：适用行业为空/`N/A`，或与施耐德相关性较弱的通用信息。",
            "- `Medium`：命中行业/领域关键词（如工业、数据中心、能源等）或政策/环保关键词（如能效、排放、标准等），但缺少明确产品/技术指向。",
            "- `High`：在 `Medium` 基础上，进一步命中施耐德/电气/能效产品关键词（如 UPS、断路器、控制器等），或与施耐德/电气相关信息更直接。",
        ]
    )

    markdown = template.format(
        chart_path=Path(chart_path).name,
        policy_table=policy_table,
        key_insights="\n".join(f"- {item}" for item in insights),
        impact_rules=impact_rules_text,
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "regulation_newsletter.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    docx_path: str | None = None
    if config.export_docx:
        docx_path = _render_docx(markdown_path, output_dir)

    return {
        "markdown_path": str(markdown_path),
        "chart_path": str(chart_path),
        **({"docx_path": docx_path} if docx_path else {}),
    }


def export_newsletter_docx(markdown_path: Path, output_dir: Path) -> str:
    """Export a DOCX from an existing newsletter markdown file."""
    return _render_docx(markdown_path, output_dir)


def _load_entries(csv_path: Path) -> Sequence[MutableMapping[str, object]]:
    entries: list[MutableMapping[str, object]] = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            final_output = json.loads(row["parse_content"]).get("final_output", {})
            validity = final_output.get("Validity") or ""
            segments = _normalise_list_field(final_output.get("Segment"))
            entry: MutableMapping[str, object] = {
                "file_id": row.get("file_id"),
                "indicator": _normalise_indicator(final_output.get("Indicator_category")),
                "validity": validity,
                "topics": final_output.get("Topics") or "",
                "abstract": final_output.get("Abstract") or "",
                "issue_date": final_output.get("Issue_date") or "",
                "validity_date": final_output.get("Validity_date") or "",
                "product_category": _normalise_list_field(final_output.get("ProductCategory")),
                "segment": segments,
                "segment_is_na": _segments_are_na(segments),
                "document_type": final_output.get("Document_type") or "",
            }
            if _should_include_entry(entry):
                entries.append(entry)
    return entries


def _normalise_list_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        candidates = value
    else:
        candidates = [value]
    cleaned: list[str] = []
    for item in candidates:
        text = str(item or "").strip()
        if not text:
            continue
        if text.upper() == "N/A":
            continue
        cleaned.append(text)
    return cleaned


def _segments_are_na(segments: Iterable[str] | None) -> bool:
    segments = list(segments or [])
    if not segments:
        return True
    return all(str(seg or "").strip().upper() == "N/A" for seg in segments)


def _impact_label_for_entry(entry: Mapping[str, object]) -> str:
    segments = entry.get("segment") or []
    if _segments_are_na(segments):
        return "Low"
    score = _impact_score(
        entry.get("product_category") or [],
        segments,
        topics=entry.get("topics"),
        abstract=entry.get("abstract"),
    )
    if score >= 2:
        return "High"
    if score == 1:
        return "Medium"
    return "Low"


def _apply_balanced_impact_labels(entries: Sequence[MutableMapping[str, object]]) -> None:
    """
    Assign impact labels for '现行有效' entries in a balanced way per indicator.

    Rules:
    - Segment is empty/N/A => Low (fixed).
    - Otherwise, compute an impact score and bucket by indicator so that overall counts follow Low > Medium > High
      and gaps are not extreme.
    """

    by_indicator: dict[str, list[MutableMapping[str, object]]] = {}
    for entry in entries:
        entry.setdefault("impact", "Low")
        if str(entry.get("validity") or "") != "现行有效":
            continue
        if _segments_are_na(entry.get("segment") or []):
            entry["impact"] = "Low"
            continue
        score = _impact_score(
            entry.get("product_category") or [],
            entry.get("segment") or [],
            topics=entry.get("topics"),
            abstract=entry.get("abstract"),
        )
        entry["_impact_score"] = score
        entry["_schneider_score"] = _schneider_relevance_score(entry)
        indicator = str(entry.get("indicator") or "Others")
        by_indicator.setdefault(indicator, []).append(entry)

    for indicator, group in by_indicator.items():
        if not group:
            continue
        low_count, medium_count, high_count = _balanced_bucket_sizes(len(group))
        ranked = sorted(
            group,
            key=lambda item: (
                int(item.get("_impact_score") or 0),
                int(item.get("_schneider_score") or 0),
                str(item.get("topics") or ""),
            ),
            reverse=True,
        )
        for idx, entry in enumerate(ranked):
            if idx < high_count:
                entry["impact"] = "High"
            elif idx < high_count + medium_count:
                entry["impact"] = "Medium"
            else:
                entry["impact"] = "Low"


def _annotate_csv_with_impact(csv_path: Path) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        fieldnames = list(reader.fieldnames)
        impact_field = "impact"
        if impact_field not in fieldnames:
            fieldnames.append(impact_field)
        rows = list(reader)

    # Pre-compute balanced impacts for "现行有效" rows by indicator to keep Low > Medium > High with tighter gaps.
    parsed_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        try:
            final_output = json.loads(row.get("parse_content", "") or "{}").get("final_output", {})
        except json.JSONDecodeError:
            final_output = {}
        indicator = _normalise_indicator(final_output.get("Indicator_category"))
        validity = final_output.get("Validity") or ""
        segments = _normalise_list_field(final_output.get("Segment"))
        entry = {
            "idx": idx,
            "indicator": indicator,
            "validity": validity,
            "product_category": _normalise_list_field(final_output.get("ProductCategory")),
            "segment": segments,
            "segment_is_na": _segments_are_na(segments),
            "topics": final_output.get("Topics") or "",
            "abstract": final_output.get("Abstract") or "",
        }
        parsed_rows.append(entry)

    by_indicator: dict[str, list[dict[str, object]]] = {}
    for entry in parsed_rows:
        if str(entry.get("validity") or "") != "现行有效":
            continue
        if entry.get("segment_is_na"):
            continue
        score = _impact_score(
            entry.get("product_category") or [],
            entry.get("segment") or [],
            topics=entry.get("topics"),
            abstract=entry.get("abstract"),
        )
        entry["_impact_score"] = score
        entry["_schneider_score"] = _schneider_relevance_score(entry)
        by_indicator.setdefault(str(entry.get("indicator") or "Others"), []).append(entry)

    balanced_labels: dict[int, str] = {}
    for indicator, group in by_indicator.items():
        if not group:
            continue
        low_count, medium_count, high_count = _balanced_bucket_sizes(len(group))
        ranked = sorted(
            group,
            key=lambda item: (
                int(item.get("_impact_score") or 0),
                int(item.get("_schneider_score") or 0),
                str(item.get("topics") or ""),
            ),
            reverse=True,
        )
        for pos, item in enumerate(ranked):
            if pos < high_count:
                balanced_labels[int(item["idx"])] = "High"
            elif pos < high_count + medium_count:
                balanced_labels[int(item["idx"])] = "Medium"
            else:
                balanced_labels[int(item["idx"])] = "Low"

    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            try:
                final_output = json.loads(row.get("parse_content", "") or "{}").get("final_output", {})
            except json.JSONDecodeError:
                final_output = {}
            segments = _normalise_list_field(final_output.get("Segment"))
            entry = {
                "product_category": _normalise_list_field(final_output.get("ProductCategory")),
                "segment": segments,
                "topics": final_output.get("Topics") or "",
                "abstract": final_output.get("Abstract") or "",
            }
            if _segments_are_na(segments):
                row[impact_field] = "Low"
            else:
                row[impact_field] = balanced_labels.get(idx) or _impact_label_for_entry(entry)
            writer.writerow(row)

    tmp_path.replace(csv_path)


def _render_docx(markdown_path: Path, output_dir: Path) -> str:
    import shutil
    import subprocess

    _ensure_matplotlib_cache(output_dir)

    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc is required to export a Word document. Install pandoc and retry.")
    docx_path = output_dir / "regulation_newsletter.docx"
    result = subprocess.run(
        [
            pandoc,
            str(markdown_path),
            "-o",
            str(docx_path),
        ],
        cwd=str(markdown_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed: {result.stderr or result.stdout}".strip())
    _postprocess_docx(docx_path)
    return str(docx_path)


def _ensure_matplotlib_cache(output_dir: Path) -> None:
    """
    Avoid slow Matplotlib imports by ensuring MPLCONFIGDIR points to a writable location.

    This affects both chart rendering and helper scripts that may import Matplotlib indirectly.
    """
    import os

    cache_root = output_dir / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _postprocess_docx(docx_path: Path) -> None:
    """
    Post-process the generated DOCX to enforce black text styles and a three-line table look.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }

    def _qn(tag: str) -> str:
        prefix, local = tag.split(":")
        return f"{{{ns[prefix]}}}{local}"

    with zipfile.ZipFile(docx_path, "r") as zf:
        files = {name: zf.read(name) for name in zf.namelist()}

    styles_xml = files.get("word/styles.xml")
    if styles_xml:
        styles = ET.fromstring(styles_xml)
        for style in styles.findall(".//w:style", ns):
            style_id = style.get(_qn("w:styleId")) or ""
            rpr = style.find("w:rPr", ns)
            if rpr is None:
                rpr = ET.SubElement(style, _qn("w:rPr"))
            color = rpr.find("w:color", ns)
            if color is None:
                color = ET.SubElement(rpr, _qn("w:color"))
            color.set(_qn("w:val"), "000000")
            if style_id == "Hyperlink":
                u = rpr.find("w:u", ns)
                if u is None:
                    u = ET.SubElement(rpr, _qn("w:u"))
                u.set(_qn("w:val"), "none")
        files["word/styles.xml"] = ET.tostring(styles, encoding="utf-8", xml_declaration=True)

    document_xml = files.get("word/document.xml")
    if document_xml:
        doc = ET.fromstring(document_xml)

        # Force all explicit run colors/underlines to black/none.
        for rpr in doc.findall(".//w:rPr", ns):
            color = rpr.find("w:color", ns)
            if color is not None:
                color.set(_qn("w:val"), "000000")
            u = rpr.find("w:u", ns)
            if u is not None:
                u.set(_qn("w:val"), "none")

        # Center the chart caption line added by the markdown template.
        for para in doc.findall(".//w:p", ns):
            texts = [node.text or "" for node in para.findall(".//w:t", ns)]
            combined = "".join(texts).strip()
            if combined != "Regulation Update Chart":
                continue
            ppr = para.find("w:pPr", ns)
            if ppr is None:
                ppr = ET.SubElement(para, _qn("w:pPr"))
            jc = ppr.find("w:jc", ns)
            if jc is None:
                jc = ET.SubElement(ppr, _qn("w:jc"))
            jc.set(_qn("w:val"), "center")
            # Make the caption bold.
            for run in para.findall("w:r", ns):
                rpr = run.find("w:rPr", ns)
                if rpr is None:
                    rpr = ET.SubElement(run, _qn("w:rPr"))
                bold = rpr.find("w:b", ns)
                if bold is None:
                    bold = ET.SubElement(rpr, _qn("w:b"))
                bold.set(_qn("w:val"), "1")
                color = rpr.find("w:color", ns)
                if color is None:
                    color = ET.SubElement(rpr, _qn("w:color"))
                color.set(_qn("w:val"), "000000")
            break

        for tbl in doc.findall(".//w:tbl", ns):
            tbl_pr = tbl.find("w:tblPr", ns)
            if tbl_pr is None:
                tbl_pr = ET.SubElement(tbl, _qn("w:tblPr"))
            borders = tbl_pr.find("w:tblBorders", ns)
            if borders is None:
                borders = ET.SubElement(tbl_pr, _qn("w:tblBorders"))

            def _set_border(which: str, val: str) -> None:
                el = borders.find(f"w:{which}", ns)
                if el is None:
                    el = ET.SubElement(borders, _qn(f"w:{which}"))
                el.set(_qn("w:val"), val)
                if val != "nil":
                    el.set(_qn("w:sz"), "12")
                    el.set(_qn("w:color"), "000000")

            _set_border("top", "single")
            _set_border("bottom", "single")
            _set_border("left", "nil")
            _set_border("right", "nil")
            _set_border("insideV", "nil")
            _set_border("insideH", "nil")

            rows = tbl.findall("w:tr", ns)
            if not rows:
                continue
            header = rows[0]
            for cell in header.findall("w:tc", ns):
                tc_pr = cell.find("w:tcPr", ns)
                if tc_pr is None:
                    tc_pr = ET.SubElement(cell, _qn("w:tcPr"))
                tc_borders = tc_pr.find("w:tcBorders", ns)
                if tc_borders is None:
                    tc_borders = ET.SubElement(tc_pr, _qn("w:tcBorders"))
                bottom = tc_borders.find("w:bottom", ns)
                if bottom is None:
                    bottom = ET.SubElement(tc_borders, _qn("w:bottom"))
                bottom.set(_qn("w:val"), "single")
                bottom.set(_qn("w:sz"), "12")
                bottom.set(_qn("w:color"), "000000")

        files["word/document.xml"] = ET.tostring(doc, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _normalise_indicator(raw: object) -> str:
    if isinstance(raw, list) and raw:
        value = raw[0]
    else:
        value = raw or "Others"
    label_map = {
        "Material_&_substance": "Material & Substance",
        "Recirculation": "Recirculation",
        "Energy_efficiency": "Energy Efficiency",
        "PKG_&_operation": "PKG & Operation",
        "Life_extension": "Life Extension",
        "Others": "Others",
    }
    return label_map.get(str(value), str(value) or "Others")


_PRODUCT_TEXT_KEYWORDS = (
    "ups",
    "pdu",
    "breaker",
    "switch",
    "meter",
    "controller",
    "cooling",
    "rack",
    "power",
    "wiser",
    "schneider",
    "施耐德",
    "断路器",
    "开关",
    "控制器",
    "冷却",
    "机柜",
    "电源",
    "pue",
)

_SEGMENT_TEXT_KEYWORDS = (
    "industrial",
    "工业",
    "transportation",
    "交通",
    "building",
    "建筑",
    "residential",
    "住宅",
    "utilities",
    "公用事业",
    "newenergy",
    "新能源",
    "retail",
    "零售",
    "water",
    "水务",
    "electronic",
    "电子",
    "automobile",
    "汽车",
    "datacenter",
    "数据中心",
    "telecom",
    "通信",
)

_POLICY_TEXT_KEYWORDS = (
    "环境",
    "污染",
    "排放",
    "绿色",
    "低碳",
    "碳",
    "节能",
    "能效",
    "能源",
    "减排",
    "标准",
    "规范",
    "办法",
    "条例",
    "规定",
)


def _normalise_text(value: object) -> str:
    return str(value or "").lower()


def _text_contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _impact_score(
    product_categories: Iterable[str],
    segments: Iterable[str],
    *,
    topics: object = None,
    abstract: object = None,
) -> int:
    score = 0
    product_categories = list(product_categories or [])
    segments = list(segments or [])

    if any(_is_se_product(cat) for cat in product_categories):
        score += 2

    combined_text = f"{topics or ''} {abstract or ''}"
    normalized = _normalise_text(combined_text)

    if _text_contains_any(normalized, _PRODUCT_TEXT_KEYWORDS):
        score += 2
    if any(_is_se_segment(seg) for seg in segments) or _text_contains_any(normalized, _SEGMENT_TEXT_KEYWORDS):
        score += 1
    if _text_contains_any(normalized, _POLICY_TEXT_KEYWORDS):
        score += 1

    return score


def _balanced_bucket_sizes(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)
    if total == 1:
        return (1, 0, 0)
    if total == 2:
        return (1, 1, 0)
    if total == 3:
        return (1, 1, 1)
    if total == 4:
        return (2, 1, 1)
    if total == 5:
        return (2, 2, 1)

    high = max(1, total // 4)  # ~25%
    medium = max(high + 1, total // 3)  # ~33% and always > high
    low = total - high - medium
    while low <= medium and medium > high + 1:
        medium -= 1
        low += 1
    return (low, medium, high)


def _balanced_counts_for_scores(scores: Sequence[int]) -> tuple[int, int, int]:
    low, medium, high = _balanced_bucket_sizes(len(scores))
    if len(scores) <= 1:
        return (low, medium, high)
    order = sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)
    high_idx = set(order[:high])
    medium_idx = set(order[high : high + medium])
    low_count = 0
    medium_count = 0
    high_count = 0
    for idx in range(len(scores)):
        if idx in high_idx:
            high_count += 1
        elif idx in medium_idx:
            medium_count += 1
        else:
            low_count += 1
    return (low_count, medium_count, high_count)


def _is_se_product(value: str | object) -> bool:
    if not value:
        return False
    text = str(value).lower()
    if text == "all":
        return False
    keywords = [
        "ups",
        "pdu",
        "breaker",
        "switch",
        "meter",
        "controller",
        "enclosedstarter",
        "cooling",
        "rack",
        "power",
        "wiser",
        "se ",
        "schneider",
    ]
    return any(k in text for k in keywords)


def _is_se_segment(value: str | object) -> bool:
    if not value:
        return False
    text = str(value).lower()
    se_segments = {
        "commercialbuilding",
        "residential",
        "building",
        "transportation",
        "utilities",
        "newenergy",
        "retail",
        "water&wastewater",
        "electronicindustry",
        "automobile",
        "datacenter",
        "telecom",
        "industrial",
    }
    return text.replace(" ", "") in se_segments


def _should_include_entry(entry: Mapping[str, object]) -> bool:
    return True


def _render_chart(entries: Sequence[Mapping[str, object]], output_dir: Path, *, first_run: bool) -> str:
    _ensure_matplotlib_cache(output_dir)

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    indicators = [
        "Material & Substance",
        "Recirculation",
        "Energy Efficiency",
        "PKG & Operation",
        "Life Extension",
    ]
    prev_counts = {indicator: 0 for indicator in indicators}
    new_counts_high = {indicator: 0 for indicator in indicators}
    new_counts_med = {indicator: 0 for indicator in indicators}
    new_counts_low = {indicator: 0 for indicator in indicators}

    for item in entries:
        indicator = item.get("indicator")
        if indicator not in indicators:
            continue
        validity = str(item.get("validity") or "")
        if validity == "现行有效":
            impact = str(item.get("impact") or _impact_label_for_entry(item))
            if impact == "High":
                new_counts_high[indicator] += 1
            elif impact == "Medium":
                new_counts_med[indicator] += 1
            else:
                new_counts_low[indicator] += 1
        else:
            prev_counts[indicator] += 1

    x = np.arange(len(indicators))
    width = 0.36
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    low_vals = np.array([new_counts_low[i] for i in indicators])
    med_vals = np.array([new_counts_med[i] for i in indicators])
    high_vals = np.array([new_counts_high[i] for i in indicators])

    if first_run:
        single_width = width * 1.6
        low_bars = ax.bar(x, low_vals, single_width, label="Low", color="#c3e6b4")
        med_bars = ax.bar(x, med_vals, single_width, bottom=low_vals, label="Medium", color="#7fbf5f")
        high_bars = ax.bar(x, high_vals, single_width, bottom=low_vals + med_vals, label="High", color="#e06c00")

        _label_segments(ax, low_bars, low_vals)
        _label_segments(ax, med_bars, med_vals, bottoms=low_vals)
        _label_segments(ax, high_bars, high_vals, bottoms=low_vals + med_vals)
    else:
        prev_bars = ax.bar(x - width / 2, [prev_counts[i] for i in indicators], width, label="Previous", color="#9ca3af")
        low_bars = ax.bar(x + width / 2, low_vals, width, label="Low", color="#c3e6b4")
        med_bars = ax.bar(x + width / 2, med_vals, width, bottom=low_vals, label="Medium", color="#7fbf5f")
        high_bars = ax.bar(x + width / 2, high_vals, width, bottom=low_vals + med_vals, label="High", color="#e06c00")

        _label_segments(ax, low_bars, low_vals)
        _label_segments(ax, med_bars, med_vals, bottoms=low_vals)
        _label_segments(ax, high_bars, high_vals, bottoms=low_vals + med_vals)

    ax.set_xticks(x, indicators, rotation=10, ha="right")
    ax.margins(x=0.02)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    if first_run:
        max_val = int((low_vals + med_vals + high_vals).max(initial=0))
    else:
        max_prev = int(max((prev_counts[i] for i in indicators), default=0))
        max_new = int((low_vals + med_vals + high_vals).max(initial=0))
        max_val = max(max_prev, max_new)

    y_max = max(4, int(np.ceil(max_val * 1.15)))
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_ylabel("Count", color="black")
    ax.set_title("")
    ax.tick_params(axis="both", colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("black")

    if first_run:
        handles = [high_bars.patches[0], med_bars.patches[0], low_bars.patches[0]]
        labels = ["High", "Medium", "Low"]
    else:
        handles = [prev_bars.patches[0], high_bars.patches[0], med_bars.patches[0], low_bars.patches[0]]
        labels = ["Previous", "High", "Medium", "Low"]
    legend = ax.legend(handles=handles, labels=labels, loc="upper right", frameon=False)
    for text in legend.get_texts():
        text.set_color("black")

    chart_path = output_dir / "regulation_update_chart.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(chart_path)


def _label_segments(ax, bars, heights, bottoms=None) -> None:
    import numpy as np

    if bottoms is None:
        bottoms = np.zeros_like(heights)
    for bar, height, bottom in zip(bars, heights, bottoms):
        if height <= 0:
            continue
        y = bottom + height / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{int(height)}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            clip_on=True,
        )


def _build_policy_table(
    entries: Sequence[Mapping[str, object]],
    max_rows: int,
    *,
    ai_emphasis: bool,
    implementation_cutoff: date,
) -> str:
    rows: list[str] = []
    doc_priority = {"laws and regulations": 0, "policy": 1, "standard": 2, "": 3}

    candidates_high: list[_PolicyRowCandidate] = []
    candidates_other: list[_PolicyRowCandidate] = []
    for entry in entries:
        if str(entry.get("document_type")) not in {"policy", "laws and regulations", ""}:
            continue
        if str(entry.get("validity") or "") != "现行有效":
            continue
        if _segments_are_na(entry.get("segment") or []):
            continue
        implementation_date = _extract_implementation_date(entry, cutoff=implementation_cutoff)
        if implementation_date is None:
            continue

        indicator = str(entry.get("indicator") or "Others")
        schneider_score = _schneider_relevance_score(entry)
        policy_name_raw, summary_raw = _extract_policy_fields(entry)
        policy_name = str(policy_name_raw or "未命名政策").replace("**", "")
        url = _extract_first_url(summary_raw)
        summary = _strip_urls(summary_raw)
        products = entry.get("product_category") or []
        product_text = (_format_product_categories(products) or "—").replace("**", "")
        segments = entry.get("segment") or []
        segment_text = (", ".join(str(s) for s in segments) or "—").replace("**", "")
        issue_date, issue_date_obj = implementation_date

        candidate = _PolicyRowCandidate(
            indicator=indicator,
            doc_type=str(entry.get("document_type") or ""),
            schneider_score=schneider_score,
            policy_name=policy_name,
            summary=summary,
            source_url=url or "",
            product_text=product_text,
            segment_text=segment_text,
            issue_date=str(issue_date).replace("**", ""),
            issue_date_obj=issue_date_obj,
        )

        impact = str(entry.get("impact") or _impact_label_for_entry(entry))
        if impact == "High":
            candidates_high.append(candidate)
        else:
            candidates_other.append(candidate)

    def _sort_key(item: _PolicyRowCandidate) -> tuple[int, int, int]:
        return (
            item.schneider_score,
            item.issue_date_obj.toordinal(),
            -doc_priority.get(item.doc_type, 3),
        )

    def _prepare_candidates(items: list[_PolicyRowCandidate]) -> list[_PolicyRowCandidate]:
        candidates_2025 = [item for item in items if item.issue_date_obj.year == 2025]
        candidates_older = [item for item in items if item.issue_date_obj.year != 2025]

        candidates_2025.sort(key=_sort_key, reverse=True)
        candidates_older.sort(key=_sort_key, reverse=True)
        return candidates_2025 + candidates_older

    def _pick_diverse(sorted_candidates: list[_PolicyRowCandidate], limit: int) -> list[_PolicyRowCandidate]:
        if not sorted_candidates or limit <= 0:
            return []
        buckets: dict[str, list[_PolicyRowCandidate]] = {}
        for item in sorted_candidates:
            buckets.setdefault(item.indicator, []).append(item)
        selected: list[_PolicyRowCandidate] = []
        for bucket in buckets.values():
            if len(selected) >= limit:
                break
            if bucket:
                selected.append(bucket.pop(0))
        if len(selected) >= limit:
            return selected[:limit]
        remaining: list[_PolicyRowCandidate] = []
        for bucket in buckets.values():
            remaining.extend(bucket)
        selected.extend(remaining[: limit - len(selected)])
        return selected

    high_sorted = _prepare_candidates(candidates_high)
    other_sorted = _prepare_candidates(candidates_other)

    selected = _pick_diverse(high_sorted, max_rows)
    if len(selected) < max_rows:
        selected.extend(_pick_diverse(other_sorted, max_rows - len(selected)))

    selected = _shorten_policy_summaries(selected, use_ai=ai_emphasis, max_chars=50)

    def _sanitize(text: str) -> str:
        return text.replace("\n", " ").replace("|", "\\|").strip()

    for item in selected:
        summary_cell = _render_summary_cell(item.summary, item.source_url)
        rows.append(
            "| "
            + " | ".join(
                _sanitize(val)
                for val in [
                    item.indicator,
                    item.policy_name,
                    summary_cell,
                    item.product_text,
                    item.segment_text,
                    item.issue_date,
                ]
            )
            + " |"
        )

    header = "| Indicator | 政策名称 | 内容概要/链接 | 适用产品 | 适用行业 | 实施日期 |\n|---|---|---|---|---|---|"
    return header + ("\n" + "\n".join(rows) if rows else "")


@dataclass(slots=True)
class _PolicyRowCandidate:
    indicator: str
    doc_type: str
    schneider_score: int
    policy_name: str
    summary: str
    source_url: str
    product_text: str
    segment_text: str
    issue_date: str
    issue_date_obj: date


_FULL_DATE_PATTERNS = (
    re.compile(r"(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})"),
    re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"),
)

_YEAR_MONTH_PATTERNS = (
    re.compile(r"(?P<year>\d{4})[/-](?P<month>\d{1,2})(?![/-]\d{1,2})"),
    re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?!\d)"),
)


def _implementation_cutoff(year: int, month: int) -> date:
    safe_month = max(1, min(12, int(month)))
    if safe_month == 12:
        return date(int(year), 12, 31)
    next_month = date(int(year), safe_month + 1, 1)
    return date.fromordinal(next_month.toordinal() - 1)


def _last_day_of_month(year: int, month: int) -> date:
    safe_month = max(1, min(12, int(month)))
    if safe_month == 12:
        return date(int(year), 12, 31)
    next_month = date(int(year), safe_month + 1, 1)
    return date.fromordinal(next_month.toordinal() - 1)


def _extract_implementation_date(entry: Mapping[str, object], *, cutoff: date) -> tuple[str, date] | None:
    extracted_dates: list[tuple[date, int, str]] = []
    for field in ("validity_date", "issue_date"):
        extracted = _extract_date(entry.get(field))
        if extracted is None:
            continue
        extracted_date, precision, display = extracted
        if extracted_date > cutoff:
            continue
        extracted_dates.append((extracted_date, precision, display))
    if not extracted_dates:
        return None
    best_date, _, best_display = max(extracted_dates, key=lambda item: (item[0].toordinal(), item[1]))
    return (best_display, best_date)


def _extract_date(value: object) -> tuple[date, int, str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in _FULL_DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            extracted = date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
            return (extracted, 2, extracted.isoformat())
        except ValueError:
            continue
    for pattern in _YEAR_MONTH_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            year = int(match.group("year"))
            month = int(match.group("month"))
        except ValueError:
            continue
        extracted = _last_day_of_month(year, month)
        return (extracted, 1, f"{year:04d}-{month:02d}")
    return None


def _apply_ai_emphasis(items: Sequence[_PolicyRowCandidate]) -> list[_PolicyRowCandidate]:
    # Deprecated: previously bolded multiple table columns. Keep as a no-op alias for compatibility.
    return list(items)


def _parse_json_maybe(text: str) -> object:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```json\s*(?P<body>.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group("body").strip())
        except json.JSONDecodeError:
            return None
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


_SCHNEIDER_KEYWORDS = (
    "schneider",
    "schneider electric",
    "施耐德",
)


def _schneider_relevance_score(entry: Mapping[str, object]) -> int:
    score = 0
    products = entry.get("product_category") or []
    segments = entry.get("segment") or []
    topics = entry.get("topics") or ""
    abstract = entry.get("abstract") or ""
    combined = f"{topics} {abstract}".lower()

    product_text = " ".join(str(item) for item in products).lower()

    if any(keyword in product_text for keyword in _SCHNEIDER_KEYWORDS) or any(keyword in combined for keyword in _SCHNEIDER_KEYWORDS):
        score += 10

    if any(_is_se_product(cat) for cat in products):
        score += 4
    if any(_is_se_segment(seg) for seg in segments):
        score += 2

    if _text_contains_any(combined, _PRODUCT_TEXT_KEYWORDS):
        score += 2
    if _text_contains_any(combined, _SEGMENT_TEXT_KEYWORDS):
        score += 1

    return score


def _truncate_chars(text: str, max_chars: int) -> str:
    value = str(text or "").replace("\n", " ").replace("|", " ").strip()
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars == 1:
        return "…"
    return value[: max_chars - 1].rstrip() + "…"


_SENTENCE_END_PUNCT = ("。", "！", "？", "!", "?", ";", "；", ".")
_SOFT_SPLIT_PUNCT = ("，", ",", "、", "：", ":")


def _complete_sentence(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "无。"
    value = value.replace("…", "").replace("...", "").replace("..", "")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return "无。"
    value = value.rstrip("".join(_SOFT_SPLIT_PUNCT)).rstrip()
    if value.endswith(_SENTENCE_END_PUNCT):
        return value
    return value + "。"


def _cap_complete_sentence(text: str, max_chars: int) -> str:
    """
    Cap to max_chars without using ellipsis, while keeping a complete sentence.
    """
    value = str(text or "").replace("\n", " ").replace("|", " ").strip()
    value = value.replace("…", "").replace("...", "")
    value = re.sub(r"\s+", " ", value).strip()
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        # If we're at the cap, drop a tiny trailing fragment after the last soft split
        # (e.g. "...，提升高端。") to keep semantics complete.
        if len(value) >= max(10, max_chars - 2):
            last_soft = max((value.rfind(p) for p in _SOFT_SPLIT_PUNCT), default=-1)
            if 0 <= last_soft < len(value) - 1:
                tail = value[last_soft + 1 :].strip()
                tail = tail.rstrip("".join(_SENTENCE_END_PUNCT)).strip()
                if 0 < len(tail) <= 6:
                    head = value[: last_soft + 1].strip()
                    head = head.rstrip("".join(_SOFT_SPLIT_PUNCT)).rstrip()
                    completed = _complete_sentence(head)
                    if len(completed) <= max_chars:
                        return completed
        completed = _complete_sentence(value)
        if len(completed) <= max_chars:
            return completed
        # Only possible when we append a terminal punctuation; keep the length within budget.
        truncated = value[: max(0, max_chars - 1)].rstrip("".join(_SOFT_SPLIT_PUNCT)).rstrip()
        return _complete_sentence(truncated)[:max_chars]

    head = value[:max_chars].rstrip()
    min_keep = max(6, max_chars // 3)
    last_stop = max((head.rfind(p) for p in _SENTENCE_END_PUNCT), default=-1)
    if last_stop >= min_keep:
        head = head[: last_stop + 1].strip()
        return _complete_sentence(head)[:max_chars]
    last_soft = max((head.rfind(p) for p in _SOFT_SPLIT_PUNCT), default=-1)
    if last_soft >= min_keep:
        head = head[: last_soft + 1].strip()
        return _complete_sentence(head)[:max_chars]
    return _complete_sentence(head)[:max_chars]


_CITATION_PARENS_PATTERN = re.compile(r"[（(][^）)]{0,40}(?:〔\d{4}〕\d+号|〔\d{4}〕\d+|第?\d+号)[^）)]{0,40}[）)]")


def _heuristic_policy_table_summary(policy_name: str, summary: str, *, max_chars: int) -> str:
    """
    Produce a short, complete Chinese sentence for the policy table without naive truncation.

    This is used as an offline fallback when the LLM-based summarizer is unavailable.
    """
    cleaned = str(summary or "").replace("\n", " ").strip()
    cleaned = _URL_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("…", "").replace("...", "").replace("..", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return _cap_complete_sentence("无", max_chars=max_chars)

    # Remove inline titles/citations because policy_name is already displayed in a separate column.
    cleaned = re.sub(r"《[^》]{1,80}》", "", cleaned)
    cleaned = _CITATION_PARENS_PATTERN.sub("", cleaned)

    # Keep only the first sentence for stability.
    sentence = re.split(r"[。.!?？!；;]", cleaned, maxsplit=1)[0].strip()
    if not sentence:
        sentence = cleaned

    candidate = sentence
    if "旨在" in candidate:
        candidate = candidate.split("旨在", 1)[1].strip()
    elif candidate.startswith("为") and ("，" in candidate or "," in candidate):
        comma = candidate.find("，")
        if comma == -1:
            comma = candidate.find(",")
        if comma > 1:
            candidate = candidate[1:comma].strip()

    candidate = candidate.strip(" ，,、:：;；")
    candidate = candidate.replace("贯彻落实", "落实")
    candidate = candidate.replace("有关", "").replace("相关", "")
    candidate = candidate.replace("对其产品的", "")
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # If still long, assemble from comma-separated clauses to avoid cutting mid-phrase.
    clauses = [part.strip() for part in re.split(r"[，,、:：;；]", candidate) if part.strip()]
    if clauses:
        chosen: list[str] = []
        for clause in clauses:
            tentative = clause if not chosen else "，".join([*chosen, clause])
            if len(tentative) <= max_chars:
                chosen.append(clause)
                continue
            if not chosen:
                break
            break
        if chosen:
            candidate = "，".join(chosen)
        else:
            candidate = clauses[0]

    candidate = candidate.strip(" ，,、:：;；")
    return _cap_complete_sentence(candidate, max_chars=max_chars)


_URL_PATTERN = re.compile(r"(https?://[^\s<>()]+)")


def _extract_first_url(value: object) -> str | None:
    text = str(value or "")
    match = _URL_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1)


def _strip_urls(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = _URL_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "无"


def _render_summary_cell(summary: str, url: str) -> str:
    cleaned = str(summary or "").strip()
    if url:
        return f"{cleaned}<br>[链接]({url})"
    return cleaned


def _format_product_categories(values: Iterable[str | object]) -> str:
    tokens: list[str] = []
    for raw in values or []:
        if raw is None:
            continue
        for part in str(raw).split(","):
            value = part.strip()
            if not value or value.upper() == "N/A":
                continue
            if value.lower() == "all":
                continue
            tokens.append(value)

    if not tokens:
        return ""

    def _is_relevant_token(token: str) -> bool:
        lowered = token.lower()
        if any(keyword in lowered for keyword in _SCHNEIDER_KEYWORDS):
            return True
        if _is_se_product(token):
            return True
        if any(keyword.lower() in lowered for keyword in _PRODUCT_TEXT_KEYWORDS):
            return True
        return False

    relevant = [token for token in tokens if _is_relevant_token(token)]
    if relevant:
        tokens = relevant

    common_terms = (
        "residualcurrentoperatedcircuit-breakerwithintegralovercurrentprotection",
        "residualcurrentoperatedcircuit-breaker",
        "arcfaultdetectiondevice",
        "miniaturecircuitbreaker",
        "residualcurrentdevice",
        "humanmachineinterface",
        "programmablelogiccontroller",
        "programmable",
        "logic",
        "variablespeeddrives",
        "mediumvoltagedrive",
        "solidstaterelay",
        "thyristorswitchandcontroller",
        "controllercontactor",
        "powersupply",
        "overcurrentprotection",
        "circuit-breaker",
        "speeddrives",
        "transformer",
        "controller",
        "interface",
        "contactor",
        "thyristor",
        "relay",
        "breaker",
        "device",
        "drives",
        "drive",
        "supply",
        "switch",
        "circuit",
        "protection",
        "power",
        "residual",
        "current",
        "integral",
        "overcurrent",
        "detection",
        "fault",
        "miniature",
        "medium",
        "voltage",
        "variable",
        "speed",
        "human",
        "machine",
        "epoxy",
        "poured",
        "dry",
        "and",
        "with",
        "system",
        "systems",
        "chiller",
        "chillers",
        "cool",
        "cooling",
        "heat",
        "source",
        "close",
        "to",
    )

    def _title_preserve_acronyms(text: str) -> str:
        parts = [p for p in re.split(r"\s+", text.strip()) if p]
        out: list[str] = []
        for part in parts:
            if part.isupper():
                out.append(part.upper())
            else:
                out.append(part[:1].upper() + part[1:])
        return " ".join(out)

    vocab = set(common_terms)
    max_vocab_len = max((len(term) for term in vocab), default=0)

    def _segment_compound(text: str) -> list[str] | None:
        if not text or max_vocab_len <= 0:
            return None
        if not text.isascii():
            return None
        dp: list[list[str] | None] = [None] * (len(text) + 1)
        dp[0] = []
        for i in range(len(text)):
            current = dp[i]
            if current is None:
                continue
            upper = min(len(text), i + max_vocab_len)
            for j in range(i + 1, upper + 1):
                token = text[i:j]
                if token not in vocab:
                    continue
                candidate = current + [token]
                existing = dp[j]
                if existing is None or len(candidate) > len(existing):
                    dp[j] = candidate
        result = dp[len(text)]
        if result and len(result) > 1:
            return result
        return None

    def _prettify(token: str) -> str:
        if "/" in token:
            left, right = token.split("/", 1)
            left = re.sub(r"([a-z])([A-Z])", r"\1 \2", left)
            left = re.sub(r"\s+", " ", left).strip()
            left = re.sub(r"(?i)breakerwith", "breaker with", left)
            right = right.strip()
            if right:
                return f"{left} ({right})"
            return left
        value = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
        value = re.sub(r"\s+", " ", value).strip()
        compact = token.replace(" ", "")
        looks_like_compound = len(compact) >= 10 and any(ch.isalpha() for ch in compact) and all(ch.isalnum() or ch in "-_" for ch in compact)
        if looks_like_compound and " " not in value:
            normalized = re.sub(r"[-_]", "", compact.lower())
            segmented = _segment_compound(normalized)
            if segmented:
                expanded = " ".join(segmented)
                return _title_preserve_acronyms(expanded)
        return _title_preserve_acronyms(value)

    pretty = [_prettify(token) for token in tokens]

    max_items = 5
    shown = pretty[:max_items]
    remaining = len(pretty) - len(shown)
    if remaining > 0:
        shown.append(f"(+{remaining})")
    return ", ".join(shown)


def _shorten_policy_summaries(items: Sequence[_PolicyRowCandidate], *, use_ai: bool, max_chars: int) -> list[_PolicyRowCandidate]:
    shortened = list(items)
    if not shortened:
        return shortened

    if use_ai:
        ai_result = _apply_ai_summary(shortened, max_chars=max_chars)
        if ai_result is not None:
            shortened = ai_result

    for item in shortened:
        item.summary = _heuristic_policy_table_summary(item.policy_name, item.summary, max_chars=max_chars)
    return shortened


def _apply_ai_summary(items: Sequence[_PolicyRowCandidate], *, max_chars: int) -> list[_PolicyRowCandidate] | None:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from .tooling.llm import ModelRouter
    except Exception:
        return None

    payload = [{"policy_name": item.policy_name, "summary": item.summary} for item in items]
    prompt_path = TEMPLATES_ROOT / "newsletter_table_summary_prompt.txt"
    try:
        prompt_template = prompt_path.read_text(encoding="utf-8")
    except OSError:
        prompt_template = (
            "你是法规政策摘要编辑。请对每条输入的 summary 进行再概括，输出一句完整的中文短句（不超过 {max_chars} 个中文字符）。\n"
            "规则：不要新增事实；不要输出链接；不要换行或竖线 |；输出严格 JSON 数组。"
        )
    system = prompt_template.format(max_chars=max_chars)
    human = f"Input JSON:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"

    try:
        model = ModelRouter().create_chat_model(temperature=0.0)
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = getattr(response, "content", "") or ""
    except Exception:
        return None

    parsed = _parse_json_maybe(text)
    if not isinstance(parsed, list) or len(parsed) != len(items):
        return None

    updated: list[_PolicyRowCandidate] = []
    for original, new_summary in zip(items, parsed):
        if not isinstance(new_summary, str):
            updated.append(original)
            continue
        cleaned = _cap_complete_sentence(new_summary, max_chars=max_chars)
        updated.append(
            _PolicyRowCandidate(
                indicator=original.indicator,
                doc_type=original.doc_type,
                schneider_score=original.schneider_score,
                policy_name=original.policy_name,
                summary=cleaned,
                source_url=original.source_url,
                product_text=original.product_text,
                segment_text=original.segment_text,
                issue_date=original.issue_date,
                issue_date_obj=original.issue_date_obj,
            )
        )
    return updated


def _emphasise(text: object) -> str:
    value = str(text or "")
    patterns = [
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",  # 2025/6/30, 2024-12-01
        r"\b\d{4}年\d{1,2}月\d{1,2}日\b",
        r"\b\d{4}[/-]\d{1,2}\b(?![/-]\d{1,2})",  # 2025-06
        r"\b\d{4}年\d{1,2}月\b(?!\d{1,2}日)",
        r"\b\d+(?:批|版|阶段)\b",
        r"\b[A-Z]{2,}\b",  # UPS, SE
    ]
    for pattern in patterns:
        value = re.sub(pattern, lambda m: f"**{m.group(0)}**", value)
    return value


def _linkify(text: str) -> str:
    """Legacy helper retained for backward compatibility; keep plain text for DOCX black styling."""
    return str(text or "")


def _build_insights(entries: Sequence[Mapping[str, object]], *, first_run: bool) -> Sequence[str]:
    if not entries:
        return ["数据为空，未生成洞察。"]

    indicator_counts: dict[str, int] = {}
    new_entries = [e for e in entries if str(e.get("validity")) == "现行有效" and str(e.get("indicator") or "") != "Others"]
    for entry in new_entries:
        indicator = str(entry.get("indicator") or "Others")
        indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
    ranked_indicators = sorted(indicator_counts.items(), key=lambda item: item[1], reverse=True)
    top_indicator = ranked_indicators[0][0] if ranked_indicators else "Others"
    second_indicator = ranked_indicators[1][0] if len(ranked_indicators) > 1 else None

    chart_indicators = (
        "Material & Substance",
        "Recirculation",
        "Energy Efficiency",
        "PKG & Operation",
        "Life Extension",
    )
    low_total = 0
    medium_total = 0
    high_total = 0
    for entry in new_entries:
        if str(entry.get("indicator") or "") not in chart_indicators:
            continue
        impact = str(entry.get("impact") or _impact_label_for_entry(entry))
        if impact == "High":
            high_total += 1
        elif impact == "Medium":
            medium_total += 1
        else:
            low_total += 1

    total = low_total + medium_total + high_total
    high_ratio = f"{(high_total / total * 100):.1f}%" if total else "0%"

    summary_lines = [
        f"现行有效条目共 {len(new_entries)} 条，其中 {top_indicator} 更新最集中（{indicator_counts.get(top_indicator, 0)} 条）。",
        f"Impact 分布：Low {low_total} / Medium {medium_total} / High {high_total}（High 占 {high_ratio}）。",
        "高影响条目优先反映与电气设备、能效、合规认证/标准相关的要求变化。",
        "重点关注电力装备、建筑/园区、数据中心、制造业等场景下的合规变化与机会点。",
        "可结合产品线与客户行业分布，优先评估高影响条目对认证、材料合规与能效指标的影响。",
    ]
    if second_indicator:
        summary_lines.insert(1, f"其次为 {second_indicator}（{indicator_counts.get(second_indicator, 0)} 条）。")
    return summary_lines


def _extract_policy_fields(entry: Mapping[str, object]) -> tuple[str, str]:
    """
    Lightweight placeholder for AI extraction; derive name from content summary rather than file_id.
    """
    topics = str(entry.get("topics") or "").strip()
    summary = str(entry.get("abstract") or "").strip()

    # Heuristics: try to detect 《...》 styled names, else use topics, else leading clause of summary.
    match = re.search(r"《([^》]+)》", summary)
    if match:
        name = match.group(1)
    elif topics:
        name = topics
    else:
        name = summary.split("。", 1)[0] if summary else ""
        if len(name) > 50:
            name = name[:50] + "…"
    return name or "未命名政策", summary or "无"
