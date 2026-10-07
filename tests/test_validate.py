from aci.validate import validate

FACTS = {
    "change_detection_test_metrics": {"f1": 0.898020941966479, "precision": 0.9077, "n_images": 128},
    "seasonal_anomaly": {"area_ha": 3005.15, "flagged_fraction_of_comparable": 0.1464, "mean_confidence": 0.68,
                         "target_year": 2025, "window": "02-01..03-10",
                         "scenes": {"2025": {"date": "2025-02-28"}}},
    "building_change_example": {"area_m2": 27479.5, "mean_confidence": 0.97},
}
SOURCES = [{"id": "S1", "text": "Apply 25 kg nitrogen per hectare at panicle initiation."}]


def status_of(result, text):
    return next(i["status"] for c in result["claims"] for i in c["items"] if i["text"].startswith(text))


def test_numbers_checked_against_facts_with_honest_rounding():
    r = validate("The model reached an F1 of 0.898 on 128 test images. About 3,005 ha changed, roughly 3,000 ha. "
                 "That is 14.6% of the farmland. Buildings covered 27,480 m².", FACTS, SOURCES)
    for t in ["0.898", "128", "3,005", "3,000", "14.6", "27,480"]:
        assert status_of(r, t) == "verified", t


def test_wrong_and_invented_numbers_are_caught():
    r = validate("About 3,300 ha changed. F1 was 0.95. There are 7 hotspots.", FACTS, SOURCES)
    assert status_of(r, "3,300") == "mismatch"  # has a unit but is 10% off the real 3,005
    assert status_of(r, "0.95") == "unsupported"
    assert status_of(r, "7") == "unsupported"
    assert r["summary"]["flagged_for_review"] == 3


def test_number_from_a_cited_guide_counts_only_with_the_citation():
    r = validate("Apply 25 kg of nitrogen per hectare [S1]. Others use 25 kg without a source.", FACTS, SOURCES)
    assert [i["status"] for c in r["claims"] for i in c["items"]] == ["sourced", "unsupported"]


def test_dates_are_checked():
    r = validate("The 2025 image is from 28 February 2025. The window runs Feb 1 to March 10. "
                 "A second image is from 5 March 2025.", FACTS, SOURCES)
    statuses = {i["text"]: i["status"] for c in r["claims"] for i in c["items"]}
    assert statuses["28 February 2025"] == "verified"
    assert statuses["Feb 1"] == "verified" and statuses["March 10"] == "verified"
    assert statuses["5 March 2025"] == "unsupported"


def test_low_confidence_uncited_causes_and_overclaims_are_flagged():
    r = validate("Buildings covered 27,480 m². In 2025, 3,005 ha showed an unusual drop. "
                 "This drop was caused by drought. Satellite data confirmed crop stress.", FACTS, SOURCES)
    reasons = {c["sentence"][:12]: " ".join(c["reasons"]) for c in r["claims"]}
    assert reasons["Buildings co"] == ""  # building track confidence 0.97 is fine
    assert "low-confidence result (seasonal_anomaly confidence 0.68)" in reasons["In 2025, 3,0"]
    assert "causal claim without a cited source" in reasons["This drop wa"]
    assert "overclaiming word: confirmed" in reasons["Satellite da"]


def test_sources_section_names_and_citations_are_not_claims():
    report = ("## Summary\nSentinel-2 and the U-Net (ResNet18) on tile 44QPD found 3,005 ha [S1].\n\n"
              "## Sources\n\n- **[S1]** guide.pdf, page 12\n")
    r = validate(report, FACTS, SOURCES)
    assert [i["text"] for c in r["claims"] for i in c["items"]] == ["3,005 ha"]


def test_citation_to_missing_source_is_reported():
    r = validate("Fields need water [S9].", FACTS, SOURCES)
    assert r["summary"]["unknown_citations"] == ["S9"]
