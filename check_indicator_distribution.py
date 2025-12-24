#!/usr/bin/env python3
"""检查 2024-2025 年数据的 indicator 分布"""
import json
import csv
from pathlib import Path
from datetime import date
import re

csv_path = Path('tb_parse_result_detail_info.csv')

def extract_date(value):
    text = str(value or '').strip()
    if not text:
        return None
    patterns = [
        re.compile(r'(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})'),
        re.compile(r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日'),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return date(int(match.group('year')), int(match.group('month')), int(match.group('day')))
            except ValueError:
                continue
    patterns_ym = [
        re.compile(r'(?P<year>\d{4})[/-](?P<month>\d{1,2})(?![/-]\d{1,2})'),
        re.compile(r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?!\d)'),
    ]
    for pattern in patterns_ym:
        match = pattern.search(text)
        if match:
            try:
                year = int(match.group('year'))
                month = int(match.group('month'))
                if month == 12:
                    return date(year, 12, 31)
                else:
                    next_month = date(year, month + 1, 1)
                    return date.fromordinal(next_month.toordinal() - 1)
            except ValueError:
                continue
    return None

indicators = ['Material & Substance', 'Recirculation', 'Energy Efficiency', 'PKG & Operation', 'Life Extension']
indicator_map = {
    'Material_&_substance': 'Material & Substance',
    'Recirculation': 'Recirculation',
    'Energy_efficiency': 'Energy Efficiency',
    'PKG_&_operation': 'PKG & Operation',
    'Life_extension': 'Life Extension',
}

indicator_counts = {ind: 0 for ind in indicators}
others_count = 0
total_2024_2025 = 0

with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            final_output = json.loads(row.get('parse_content', '{}')).get('final_output', {})
        except json.JSONDecodeError:
            continue
        
        validity_date = extract_date(final_output.get('Validity_date'))
        issue_date = extract_date(final_output.get('Issue_date'))
        
        impl_date = None
        if validity_date and issue_date:
            impl_date = max(validity_date, issue_date)
        elif validity_date:
            impl_date = validity_date
        elif issue_date:
            impl_date = issue_date
        
        if impl_date and date(2024, 1, 1) <= impl_date <= date(2025, 12, 31):
            total_2024_2025 += 1
            indicator_raw = final_output.get('Indicator_category')
            
            if isinstance(indicator_raw, list):
                indicator_raw = indicator_raw[0] if indicator_raw else None
            
            indicator_normalized = indicator_map.get(str(indicator_raw), None)
            
            if indicator_normalized and indicator_normalized in indicators:
                indicator_counts[indicator_normalized] += 1
            else:
                others_count += 1

print(f'2024-2025年总条目数: {total_2024_2025}')
print(f'\n按指标分类（仅统计5大指标）:')
for ind in indicators:
    print(f'  {ind}: {indicator_counts[ind]}')
print(f'  Others（未统计）: {others_count}')
print(f'\n图表显示的总数: {sum(indicator_counts.values())}')
print(f'缺失的数据: {others_count} 条（占比 {others_count/total_2024_2025*100:.1f}%）')
