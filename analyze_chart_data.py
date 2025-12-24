#!/usr/bin/env python3
"""
分析为什么图表没有显示所有89条2024-2025年数据
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

def analyze():
    entries_2024_2025 = []
    indicator_breakdown = {}
    
    chart_indicators = [
        "Material & Substance",
        "Recirculation",
        "Energy Efficiency",
        "PKG & Operation",
        "Life Extension",
    ]
    
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                final_output = json.loads(row.get("parse_content", "{}")).get("final_output", {})
            except json.JSONDecodeError:
                continue
            
            validity_date_str = final_output.get("Validity_date") or ""
            issue_date_str = final_output.get("Issue_date") or ""
            indicator_raw = final_output.get("Indicator_category") or []
            
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
                # 标准化 indicator
                if isinstance(indicator_raw, list):
                    indicators = indicator_raw
                else:
                    indicators = [indicator_raw] if indicator_raw else []
                
                # 清理 indicator 列表
                clean_indicators = []
                for ind in indicators:
                    ind_str = str(ind or "").strip()
                    if ind_str and ind_str != "N/A":
                        clean_indicators.append(ind_str)
                
                if not clean_indicators:
                    clean_indicators = ["Others"]
                
                entry = {
                    "file_id": row.get("file_id"),
                    "impl_date": impl_date,
                    "indicators": clean_indicators,
                    "topics": final_output.get("Topics", ""),
                }
                entries_2024_2025.append(entry)
                
                # 统计指标分布
                for ind in clean_indicators:
                    # 标准化指标名称（处理下划线）
                    ind_normalized = ind.replace("_", " ")
                    if ind_normalized in chart_indicators:
                        if ind_normalized not in indicator_breakdown:
                            indicator_breakdown[ind_normalized] = 0
                        indicator_breakdown[ind_normalized] += 1
    
    print(f"=== 2024-2025年数据分析 ===")
    print(f"总条目数: {len(entries_2024_2025)}")
    print(f"\n指标分布（图表中应该显示的）:")
    print(f"{'Indicator':<30} {'Count':<10}")
    print("-" * 40)
    
    chart_total = 0
    for ind in chart_indicators:
        count = indicator_breakdown.get(ind, 0)
        chart_total += count
        print(f"{ind:<30} {count:<10}")
    
    print("-" * 40)
    print(f"{'Total in chart':<30} {chart_total:<10}")
    
    print(f"\n未包含在图表中的原因:")
    print(f"- 没有明确 indicator 或 indicator 为'Others': {len([e for e in entries_2024_2025 if 'Others' in e['indicators'] or not e['indicators']])}")
    
    # 显示有'Others' indicator的条目
    others_count = 0
    for entry in entries_2024_2025:
        if "Others" in entry["indicators"] or not entry["indicators"]:
            others_count += 1
    
    print(f"- Indicator 为 'Others' 的条目: {others_count}")
    
    print(f"\n预期图表应该显示: {chart_total} / {len(entries_2024_2025)}")
    print(f"未被显示的条目: {len(entries_2024_2025) - chart_total}")

if __name__ == "__main__":
    analyze()
