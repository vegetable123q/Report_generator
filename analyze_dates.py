#!/usr/bin/env python3
"""Analyze implementation dates in the CSV file."""

import csv
import json
import re
from datetime import date
from pathlib import Path


def extract_date(text):
    """Extract date from text field."""
    if not text:
        return None
    text = str(text).strip()
    
    # Full date patterns
    full_patterns = [
        re.compile(r'(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})'),
        re.compile(r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日'),
    ]
    
    for pattern in full_patterns:
        match = pattern.search(text)
        if match:
            try:
                year = int(match.group('year'))
                month = int(match.group('month'))
                day = int(match.group('day'))
                return date(year, month, day)
            except:
                continue
    
    # Year-month patterns
    ym_patterns = [
        re.compile(r'(?P<year>\d{4})[/-](?P<month>\d{1,2})(?![/-]\d{1,2})'),
        re.compile(r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?!\d)'),
    ]
    
    for pattern in ym_patterns:
        match = pattern.search(text)
        if match:
            try:
                year = int(match.group('year'))
                month = int(match.group('month'))
                # Use last day of month
                if month == 12:
                    return date(year, 12, 31)
                next_month = date(year, month + 1, 1)
                return date.fromordinal(next_month.toordinal() - 1)
            except:
                continue
    
    return None


def main():
    csv_path = Path('tb_parse_result_detail_info.csv')
    
    total = 0
    in_range = 0
    with_segment = 0
    valid_entries = []
    
    start = date(2024, 1, 1)
    end = date(2025, 12, 31)
    
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                final_output = json.loads(row['parse_content']).get('final_output', {})
            except:
                continue
            
            validity_date = final_output.get('Validity_date') or ''
            issue_date = final_output.get('Issue_date') or ''
            segments = final_output.get('Segment')
            validity = final_output.get('Validity') or ''
            
            # Extract dates
            impl_date = None
            for field in [validity_date, issue_date]:
                d = extract_date(field)
                if d and d <= end:
                    if impl_date is None or d > impl_date:
                        impl_date = d
            
            if impl_date and start <= impl_date <= end:
                in_range += 1
                
                # Check segment
                has_valid_seg = False
                if segments:
                    if isinstance(segments, list):
                        seg_list = segments
                    else:
                        seg_list = [segments]
                    has_valid_seg = any(
                        str(s).strip() and str(s).strip().upper() != 'N/A' 
                        for s in seg_list
                    )
                
                entry_info = {
                    'date': impl_date,
                    'topic': final_output.get('Topics', '未知')[:70],
                    'has_segment': has_valid_seg,
                    'validity': validity,
                    'segments': segments,
                }
                valid_entries.append(entry_info)
                
                if has_valid_seg:
                    with_segment += 1
    
    # Sort by date
    valid_entries.sort(key=lambda x: x['date'], reverse=True)
    
    print(f'总计: {total} 条')
    print(f'实施日期在 2024-2025: {in_range} 条')
    print(f'其中有非N/A行业: {with_segment} 条')
    print(f'\n前 50 条 2024-2025 年条目:')
    print('=' * 100)
    
    for i, entry in enumerate(valid_entries[:50], 1):
        seg_mark = '✓' if entry['has_segment'] else '✗'
        print(f'{i:2d}. [{entry["date"]}] [{seg_mark}] {entry["topic"]}')
    
    print(f'\n\n有行业的条目详情 (前30条):')
    print('=' * 100)
    with_seg_entries = [e for e in valid_entries if e['has_segment']]
    for i, entry in enumerate(with_seg_entries[:30], 1):
        print(f'{i:2d}. [{entry["date"]}] {entry["topic"]}')
        print(f'    行业: {entry["segments"]}')
        print(f'    有效性: {entry["validity"]}')
        print()


if __name__ == '__main__':
    main()
