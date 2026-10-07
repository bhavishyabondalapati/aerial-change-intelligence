import json

import numpy as np

from aci.rag import build_index
from aci.report import write_report
from test_rag import fake_embed


def _setup(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "seasonal").mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"f1": 0.898, "iou": 0.815}))
    (out / "seasonal" / "seasonal_summary.json").write_text(json.dumps({"area_ha": 3005.15, "target_year": 2025}))
    monkeypatch.setattr("aci.rag.read_pdf_pages", lambda p: [(4, "During drought, irrigate at critical stages and save water."),
                                                            (9, "Yellow leaves can mean nitrogen deficiency.")])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.pdf").write_bytes(b"")
    build_index(tmp_path / "docs", tmp_path / "index", fake_embed)
    return out


def test_report_is_grounded_in_facts_and_numbered_sources(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch)
    seen = {}

    def fake_generate(prompt, system):
        seen["prompt"], seen["system"] = prompt, system
        return "## Summary\nAbout 3005 ha showed an unusual drop [S1]."

    report = write_report(out, tmp_path / "index", tmp_path / "report", fake_embed, fake_generate, "fake-model")
    # The model was given our numbers and the guide passages, with ids
    assert "3005.15" in seen["prompt"] and "[S1]" in seen["prompt"] and "drought" in seen["prompt"].lower()
    assert "Do not calculate new numbers" in seen["system"]
    # Sources are listed by code, with file and page
    assert "## Sources" in report and "guide.pdf, page" in report
    context = json.loads((tmp_path / "report" / "context.json").read_text())
    assert context["facts"]["seasonal_anomaly"]["area_ha"] == 3005.15
    assert [s["id"] for s in context["sources"]] == ["S1", "S2"]  # duplicates across questions removed
    assert (tmp_path / "report" / "report.md").read_text() == report


def test_missing_result_files_are_skipped(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch)
    write_report(out, tmp_path / "index", tmp_path / "report", fake_embed, lambda p, s: "ok", "fake-model")
    facts = json.loads((tmp_path / "report" / "context.json").read_text())["facts"]
    assert set(facts) == {"change_detection_test_metrics", "seasonal_anomaly"}
