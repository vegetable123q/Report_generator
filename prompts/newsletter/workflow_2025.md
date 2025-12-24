# Newsletter Generation (2024-01 ~ 2025-12) — Codex Execution Prompt

## Goal
Generate the final Word deliverable for the regulation update newsletter, but **only** for entries whose implementation date is within **2024-01 to 2025-12**. Do not show any data source paths.

## Technical Context: Chart Visualization Fix
**Problem**: The workflow_2025 chart was only showing 2 records instead of all 89 records in the date range.

**Root Cause**: The `_render_chart()` function in `src/tiangong_ai_workspace/newsletter.py` (lines 744-902) was filtering entries to only include 5 predefined indicators (Material & Substance, Recirculation, Energy Efficiency, PKG & Operation, Life Extension). 87 out of 89 records had `indicator="Others"` and were being excluded.

**Solution Implemented**: Added an early branch for `chart_style == "single"` (used by workflow_2025) that:
1. Dynamically collects ALL indicator values from the data, including "Others"
2. Counts all entries (not just the 5 predefined indicators)
3. Generates a complete bar chart showing distribution across all indicators found in the data
4. Preserves the original multi-layer stacked bar logic for default mode

**Key Code Pattern** (lines 764-809 in `newsletter.py`):
```python
if chart_style == "single":
    # Count all entries by indicator, including Others
    indicator_counts = {}
    for item in entries:
        indicator = str(item.get("indicator") or "Others")
        indicator = indicator.replace("_", " ").replace("&", "&").strip()
        if indicator == "Others" or not indicator:
            indicator = "Others"
        indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
    
    # Build chart with ALL indicators found in data
    all_indicators = list(indicators)  # Start with known indicators
    for ind in indicator_counts.keys():
        if ind not in all_indicators:
            all_indicators.append(ind)
    # ... render chart with all_indicators
```

**Verification**: 
- ✅ Chart now shows all 89 records distributed across indicators
- ✅ Key Insights correctly states "时间范围内条目共 89 条"
- ✅ All 7 unit tests pass
- ✅ Both workflow_2025 and workflow_default generate correctly

## Available Tools (commands)
- `uv run tiangong-workspace newsletter generate` — generate chart + markdown + embedded Word (`.docx`)
- `uv run tiangong-workspace newsletter export-docx` — export Word (`.docx`) from an edited markdown
- `unzip -l <docx>` — verify the chart is embedded (`word/media/*.png`)
- `rg` / `sed` — quick validation of output text

## Inputs
- Use project defaults. Do not pass or mention any `--csv-path` argument.

## Constraints
- Output must be a single final Word document (`.docx`) with the chart embedded.
- Chart: **single-series bar chart** showing ALL indicators (including "Others") found in the date-filtered data — the chart_style="single" branch in `_render_chart()` ensures all 89 records are counted and displayed.
- Table rows: 12 policies; **do not apply other filters** (e.g., do not require non-`N/A` segment), except:
  - Implementation date must be within **2024-01 to 2025-12** (inclusive).
- Table summary ("内容概要/链接"): AI-generated one-sentence core meaning ≤50 Chinese characters; must summarize the core requirement/change (not just truncate).
- AI bold emphasis must appear only in "内容概要/链接" and "Key Insights" (no bold anywhere else).
- All Word text must be black; table should be three-line style; chart caption must be centered below the image and read exactly `Regulation Update Chart`.
- Do not include "Regulation Update Newsletter" anywhere.
- Policy table header must use `生态设计指标类别` (not `Indicator`).
- If "适用产品" contains more than 5 items, keep 5 most relevant items, then append `等电力装备` (avoid overly long lists).

## Steps
1. Run newsletter generation (skip DOCX for now):
   - `uv run tiangong-workspace newsletter generate --workflow 2025 --output-dir outputs --no-docx --json`
   - Expected: Chart will show all indicators (including "Others") with full 89 records counted
2. Confirm outputs exist:
   - `outputs/regulation_update_chart.png` (should be ~110 KB, 3243x1919 px)
   - `outputs/regulation_newsletter.md`
3. Validate chart data (optional verification):
   - Key Insights should state: "时间范围内条目共 89 条"
   - Chart should display bars for all indicators found in the data (typically: Material & Substance: 2, Recirculation: 2, Life Extension: 1, Others: 84)
4. Codex post-process (summary + emphasis + product list sanity):
   - Ensure every "实施日期" in the table is within `2024-01` to `2025-12` (inclusive).
   - Ensure "内容概要/链接" is a single complete Chinese sentence (≤50 chars, do not use `…` / `...` / `……`) and summarizes the core requirement/change.
   - If "适用产品" is too long (>5 items), keep 5 most relevant, then append `等电力装备`.
   - Add Markdown bold (**...**) to highlight key info:
     - Summary column: bold 1–2 key requirement/impact phrases and/or key numbers per row.
     - Key Insights: bold key numbers, proportions, and priority conclusions (≤2 bold segments per bullet).
     - Do not add bold anywhere else (policy name/products/segments/dates/impact rules must remain plain).
5. Export the final DOCX from the edited markdown:
   - `uv run tiangong-workspace newsletter export-docx --markdown-path outputs/regulation_newsletter.md --output-dir outputs --json`
5. Verify the chart is embedded into the Word file:
   - `unzip -l outputs/regulation_newsletter.docx | rg "word/media/.*\\.png"`
6. Sanity-check that forbidden phrase is absent:
   - `rg -n "Regulation Update Newsletter" outputs/regulation_newsletter.md outputs/regulation_newsletter.docx || true`

## Deliverable
- Return the path to the final Word file: `outputs/regulation_newsletter.docx`

## Troubleshooting: Chart Shows Incomplete Data

If the chart only shows 2-5 bars instead of all indicators (e.g., only showing 2 out of 89 records):

### Diagnosis Steps
1. Check Key Insights in the markdown output — it should state total record count (e.g., "时间范围内条目共 89 条")
2. Inspect the chart PNG — count the number of bars displayed
3. If counts don't match, the `_render_chart()` function needs fixing

### Root Cause
The function is filtering entries by a fixed list of 5 indicators, excluding "Others" which contains most records.

### Fix Required
Edit `src/tiangong_ai_workspace/newsletter.py` (lines 744-902) in the `_render_chart()` function:

1. **Add early branching** — Check `if chart_style == "single":` BEFORE the main counting loop
2. **Dynamic indicator collection** — For single-style charts, collect ALL indicator values:
   ```python
   if chart_style == "single":
       indicator_counts = {}
       for item in entries:
           indicator = str(item.get("indicator") or "Others")
           indicator = indicator.replace("_", " ").replace("&", "&").strip()
           if indicator == "Others" or not indicator:
               indicator = "Others"
           indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
       
       # Build chart with ALL indicators found in data
       all_indicators = list(indicators)  # Start with known indicators
       for ind in indicator_counts.keys():
           if ind not in all_indicators:
               all_indicators.append(ind)
       # ... render chart with all_indicators
   ```
3. **Preserve default mode** — Keep original logic for multi-layer stacked bars unchanged

### Verification Commands
```bash
# Run tests (should show 7/7 passing)
uv run pytest tests/test_newsletter.py -v

# Format and lint check
uv run black src/tiangong_ai_workspace/newsletter.py
uv run ruff check src/tiangong_ai_workspace/newsletter.py

# Regenerate and verify
uv run tiangong-workspace newsletter generate --workflow 2025 --output-dir outputs --first-run
```

### Expected Results After Fix
- **workflow_2025**: "时间范围内条目共 89 条" in Key Insights
- **Chart PNG**: ~110 KB, 3243x1919 px, showing bars for all indicators
- **workflow_default**: "现行有效条目共 7370 条" (should still work correctly)


