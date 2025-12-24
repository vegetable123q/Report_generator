from __future__ import annotations

from datetime import date

from tiangong_ai_workspace import newsletter


def test_policy_table_requires_concrete_implementation_date() -> None:
    entries = [
        {
            "indicator": "Material & Substance",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "Test policy A.",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2025-06-30",
            "issue_date": "",
        },
        {
            "indicator": "Recirculation",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "Test policy B.",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "",
            "issue_date": "2025-06",
        },
    ]

    table = newsletter._build_policy_table(entries, max_rows=10, ai_emphasis=False, implementation_cutoff=date(2025, 12, 31))
    data_lines = [line for line in table.splitlines()[2:] if line.startswith("| ")]
    assert len(data_lines) == 2
    dates = []
    for line in data_lines:
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        dates.append(fields[-1])
    assert any("2025-06-30" in item for item in dates)
    assert any(item == "2025-06" for item in dates)


def test_policy_table_prioritises_schneider_relevance() -> None:
    entries = [
        {
            "indicator": "Material & Substance",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "Schneider Electric",
            "abstract": "A1",
            "product_category": ["Schneider"],
            "segment": ["Industrial"],
            "validity_date": "2025-06-30",
            "issue_date": "",
        },
        {
            "indicator": "Material & Substance",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "A2",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2025-06-29",
            "issue_date": "",
        },
        {
            "indicator": "Recirculation",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "B1",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2025-06-28",
            "issue_date": "",
        },
    ]

    table = newsletter._build_policy_table(entries, max_rows=2, ai_emphasis=False, implementation_cutoff=date(2025, 12, 31))
    data_lines = [line for line in table.splitlines()[2:] if line.startswith("| ")]
    assert len(data_lines) == 2
    assert "Schneider" in data_lines[0]


def test_policy_table_summary_is_capped() -> None:
    entries = [
        {
            "indicator": "Material & Substance",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "这是一个很长很长的摘要，用来测试是否会被截断到三十字以内。",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2025-06-30",
            "issue_date": "",
        }
    ]

    table = newsletter._build_policy_table(entries, max_rows=8, ai_emphasis=False, implementation_cutoff=date(2025, 12, 31))
    data_lines = [line for line in table.splitlines()[2:] if line.startswith("| ")]
    assert len(data_lines) == 1
    fields = [field.strip() for field in data_lines[0].strip().strip("|").split("|")]
    assert len(fields) >= 6
    summary_cell = fields[2]
    summary_only = summary_cell.split("<br>", 1)[0].strip()
    assert len(summary_only) <= 50
    assert "…" not in summary_only
    assert "..." not in summary_only


def test_policy_table_excludes_dates_after_cutoff() -> None:
    entries = [
        {
            "indicator": "Material & Substance",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "A",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2026-01-01",
            "issue_date": "",
        },
        {
            "indicator": "Recirculation",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "B",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2025-12-31",
            "issue_date": "",
        },
    ]

    table = newsletter._build_policy_table(entries, max_rows=10, ai_emphasis=False, implementation_cutoff=date(2025, 12, 31))
    data_lines = [line for line in table.splitlines()[2:] if line.startswith("| ")]
    assert len(data_lines) == 1
    assert "2025-12-31" in data_lines[0]


def test_policy_table_excludes_na_segments() -> None:
    entries = [
        {
            "indicator": "Material & Substance",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "A",
            "product_category": [],
            "segment": [],
            "validity_date": "2025-06-30",
            "issue_date": "",
        },
        {
            "indicator": "Recirculation",
            "validity": "现行有效",
            "document_type": "policy",
            "topics": "",
            "abstract": "B",
            "product_category": [],
            "segment": ["Industrial"],
            "validity_date": "2025-06-29",
            "issue_date": "",
        },
    ]

    table = newsletter._build_policy_table(entries, max_rows=10, ai_emphasis=False, implementation_cutoff=date(2025, 12, 31))
    data_lines = [line for line in table.splitlines()[2:] if line.startswith("| ")]
    assert len(data_lines) == 1
    assert "Recirculation" in data_lines[0]
