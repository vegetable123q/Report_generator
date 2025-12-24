#!/usr/bin/env python3
"""
分析 CSV 中有多少条目的实施日期在 2024-2025 年范围内，并检查各种筛选条件
"""
import csv
import json
import re
from datetime import date
from pathlib import Path

csv_path = Path("tb_parse_result_detail_info.csv")

def extract_date(value):
    """提取日期"""
    text = str(value or "").strip()
    if not text:
        return None
    
    # 完整日期格式
    patterns = [
        re.compile(r"(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})"),
        re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
            except ValueError:
                continue
    
    # 年月格式
    patterns_ym = [
        re.compile(r"(?P<year>\d{4})[/-](?P<month>\d{1,2})(?![/-]\d{1,2})"),
        re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?!\d)"),
    ]
    for pattern in patterns_ym:
        match = pattern.search(text)
        if match:
            try:
                year = int(match.group("year"))
                month = int(match.group("month"))
                if month == 12:
                    return date(year, 12, 31)
                else:
                    next_month = date(year, month + 1, 1)
                    return date.fromordinal(next_month.toordinal() - 1)
            except ValueError:
                continue
    return None

def normalize_list_field(value):
    """标准化列表字段"""
    if value is None:
        return []
    if isinstance(value, list):
        candidates = value
    else:
        candidates = [value]
    cleaned = []
    for item in candidates:
        text = str(item or "").strip()
        if not text or text.upper() == "N/A":
            continue
        cleaned.append(text)
    return cleaned

def segments_are_na(segments):
    """检查segments是否为空或N/A"""
    segments = list(segments or [])
    if not segments:
        return True
    return all(str(seg or "").strip().upper() == "N/A" for seg in segments)

def analyze():
    entries_in_range = []
    strict_filtered_count = 0
    
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                final_output = json.loads(row.get("parse_content", "{}")).get("final_output", {})
            except json.JSONDecodeError:
                continue
            
            validity_date_str = final_output.get("Validity_date") or ""
            issue_date_str = final_output.get("Issue_date") or ""
            validity = final_output.get("Validity") or ""
            segment = normalize_list_field(final_output.get("Segment"))
            doc_type = final_output.get("Document_type") or ""
            indicator = final_output.get("Indicator_category") or []
            
            validity_date = extract_date(validity_date_str)
            issue_date = extract_date(issue_date_str)
            
            # 选择较晚的日期
            impl_date = None
            if validity_date and issue_date:
                impl_date = max(validity_date, issue_date)
            elif validity_date:
                impl_date = validity_date
            elif issue_date:
                impl_date = issue_date
            
            if impl_date and date(2024, 1, 1) <= impl_date <= date(2025, 12, 31):
                entry_info = {
                    "file_id": row.get("file_id"),
                    "impl_date": impl_date,
                    "validity": validity,
                    "topics": final_output.get("Topics", ""),
                    "segment": segment,
                    "segment_is_na": segments_are_na(segment),
                    "doc_type": doc_type,
                    "indicator": indicator,
                }
                entries_in_range.append(entry_info)
                
                # 检查是否通过 strict 过滤
                passes_strict = True
                if doc_type not in {"policy", "laws and regulations", ""}:
                    passes_strict = False
                if validity != "现行有效":
                    passes_strict = False
                if entry_info["segment_is_na"]:
                    passes_strict = False
                
                if passes_strict:
                    strict_filtered_count += 1
    
    print(f"2024-2025年范围内的总条目数: {len(entries_in_range)}")
    print(f"通过 strict 筛选的条目数: {strict_filtered_count}")
    print(f"date_only 模式可用的条目数: {len(entries_in_range)}")
    
    # 分析不能通过 strict 筛选的原因
    doc_type_issues = 0
    validity_issues = 0
    segment_issues = 0
    
    for entry in entries_in_range:
        if entry["doc_type"] not in {"policy", "laws and regulations", ""}:
            doc_type_issues += 1
        if entry["validity"] != "现行有效":
            validity_issues += 1
        if entry["segment_is_na"]:
            segment_issues += 1
    
    print(f"\n筛选失败原因分析:")
    print(f"  document_type 不符: {doc_type_issues}")
    print(f"  validity 不是'现行有效': {validity_issues}")
    print(f"  segment 为空/N/A: {segment_issues}")
    
    # 按年份分组
    by_year = {"2024": [], "2025": []}
    for entry in entries_in_range:
        year_key = str(entry["impl_date"].year)
        by_year[year_key].append(entry)
    
    print(f"\n按年份分组:")
    print(f"  2024年: {len(by_year['2024'])} 条")
    print(f"  2025年: {len(by_year['2025'])} 条")
    
    # 显示前30条（date_only 模式）
    print(f"\n前30条记录（按日期倒序，date_only模式可显示）:")
    print(f"{'实施日期':<12} {'有效性':<15} {'Segment?':<10} {'DocType':<20} {'政策主题':<50}")
    print("-" * 115)
    for entry in sorted(entries_in_range, key=lambda x: x["impl_date"], reverse=True)[:30]:
        topics = (entry["topics"][:47] + "...") if len(entry["topics"]) > 50 else entry["topics"]
        segment_status = "N/A" if entry["segment_is_na"] else f"{len(entry['segment'])}项"
        doc_type = entry["doc_type"][:17] if entry["doc_type"] else "空"
        print(f"{entry['impl_date']} {entry['validity']:<15} {segment_status:<10} {doc_type:<20} {topics}")

if __name__ == "__main__":
    analyze()
