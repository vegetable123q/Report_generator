# Newsletter Generation — Codex Execution Prompt

## Goal
Generate the final Word deliverable for the regulation update newsletter based on the current project workflow, without showing any data source paths.

## Available Tools (commands)
- `uv run tiangong-workspace newsletter generate` — generate chart + markdown + embedded Word (`.docx`)
- `uv run tiangong-workspace newsletter export-docx` — export Word (`.docx`) from an edited markdown
- `unzip -l <docx>` — verify the chart is embedded (`word/media/*.png`)
- `rg` / `sed` — quick validation of output text

## Inputs
- Use project defaults. Do not pass or mention any `--csv-path` argument.

## Constraints
- Output must be a single final Word document (`.docx`) with the chart embedded.
- Table rows: 12 policies; policies must include both implementation date and non-`N/A` segment.
- Table summary (“内容概要/链接”): AI-generated one-sentence core meaning ≤50 Chinese characters; must summarize the core requirement/change (not just truncate).
- AI bold emphasis must appear only in “内容概要/链接” and “Key Insights” (no bold anywhere else).
- All Word text must be black; table should be three-line style; chart caption must be centered below the image and read exactly `Regulation Update Chart`.
- Do not include “Regulation Update Newsletter” anywhere.
- Policy table header must use `生态设计指标类别` (not `Indicator`).
- If “适用产品” contains more than 5 items, keep 5 most relevant items, then append `等电力装备` (avoid overly long lists).

## Steps
1. Run newsletter generation (first-run chart mode; skip DOCX for now):
   - `uv run tiangong-workspace newsletter generate --output-dir outputs --first-run --no-docx --json`
2. Confirm outputs exist:
   - `outputs/regulation_update_chart.png`
   - `outputs/regulation_newsletter.md`
3. Codex post-process (summary + emphasis):
   - Ensure “内容概要/链接” is a single complete Chinese sentence (≤50 chars, do not use `…` / `...` / `……`) and summarizes the core requirement/change.
   - Add Markdown bold (**...**) to highlight key info:
     - Summary column: bold 1–2 key requirement/impact phrases and/or key numbers per row.
     - Key Insights: bold key numbers, proportions, and priority conclusions (≤2 bold segments per bullet).
     - Do not add bold anywhere else (policy name/products/segments/dates/impact rules must remain plain).
   - If “适用产品” is too long (>5 items), keep 5 most relevant, then append `等电力装备`.
   - If needed, re-run step 1 to refresh content, then re-apply this step.
4. Export the final DOCX from the edited markdown:
   - `uv run tiangong-workspace newsletter export-docx --markdown-path outputs/regulation_newsletter.md --output-dir outputs --json`
5. Verify the chart is embedded into the Word file:
   - `unzip -l outputs/regulation_newsletter.docx | rg "word/media/.*\\.png"`
6. Sanity-check that forbidden phrase is absent:
   - `rg -n "Regulation Update Newsletter" outputs/regulation_newsletter.md outputs/regulation_newsletter.docx || true`

## Deliverable
- Return the path to the final Word file: `outputs/regulation_newsletter.docx`
