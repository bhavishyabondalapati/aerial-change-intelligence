"""Check every number and date in the Gemini report against the JSON facts it was given, and flag claims for human review.

Each number gets a status:
  verified    - matches a fact (honest rounding allowed)
  sourced     - not one of our facts, but appears in a guide passage the same sentence cites
  mismatch    - has a measured unit (ha, m², %) yet matches none of our facts: probably wrong
  unsupported - matches nothing at all
Each sentence can also be flagged low-confidence: it uses numbers from a result whose own confidence is low,
makes a causal claim without citing a passage, or overclaims ("confirmed", "proves").
"""
import argparse
import json
import re
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}
MONTH_RE = "(" + "|".join(list(MONTHS) + [m[:3] for m in MONTHS]) + r")\.?"
DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),  # 2025-02-28
    re.compile(rf"\b(\d{{1,2}})\s+{MONTH_RE}(?:\s+(\d{{4}}))?\b", re.I),  # 28 February 2025 / 1 March
    re.compile(rf"\b{MONTH_RE}\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b", re.I),  # February 28, 2025 / Mar 10
]
# Names that contain digits but are not claims
IGNORE = re.compile(r"Sentinel-2[AB]?|\bS2\b|L2A|U-Net|ResNet\d+|LEVIR-CD|EPSG:\d+|gemini-[\w.-]+|\b\d{2}[A-Z]{3}\b", re.I)
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\w)\s*(%|percent|ha\b|hectares?|m²|m2\b|sq\.? ?m)?", re.I)
CITATION = re.compile(r"\[(S\d+)\]")
CAUSAL = re.compile(r"\b(caused by|due to|because of|result of|attributable to|leads? to)\b", re.I)
OVERCLAIM = re.compile(r"\b(confirm(?:s|ed)?|prov(?:es|en)|definitely|certainly|clearly shows)\b", re.I)
UNIT_FAMILY = {"%": "%", "percent": "%", "ha": "ha", "hectare": "ha", "hectares": "ha",
               "m²": "m2", "m2": "m2", "sq m": "m2", "sq. m": "m2", "sqm": "m2", "sq.m": "m2"}


def flatten(facts, prefix=""):
    """Nested JSON -> ([(path, number)], [(path, text)])."""
    nums, texts = [], []
    items = facts.items() if isinstance(facts, dict) else enumerate(facts)
    for k, v in items:
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            nums.append((path, float(v)))
        elif isinstance(v, str):
            texts.append((path, v))
        elif isinstance(v, (dict, list)):
            n, t = flatten(v, path)
            nums += n
            texts += t
    return nums, texts


def rounding_tolerance(digits, decimals):
    """How far off a printed number may be and still be an honest rounding: half its last significant place,
    but never more than 5% (so "3,000" can stand for 3,005 but not for 3,300)."""
    if decimals:
        return 0.5 * 10 ** -len(decimals.lstrip("."))
    stripped = digits.replace(",", "")
    trailing_zeros = len(stripped) - len(stripped.rstrip("0")) if stripped.strip("0") else 0
    return 0.5 * 10 ** trailing_zeros


def matches(value, tol, fact):
    diff = abs(value - fact)
    if tol == 0.5 and abs(fact) < 10:  # small whole numbers ("3 regions") must be exact, not roundings of 0.8 etc.
        return diff < 1e-9
    return diff <= tol and (tol < 1 or diff <= 0.05 * abs(fact))


def check_number(value, tol, unit, facts_num, facts_text):
    """Fact paths this number matches (a % may also match a 0-1 fraction times 100)."""
    hits = [p for p, f in facts_num if matches(value, tol, f)]
    if unit == "%":
        hits += [p for p, f in facts_num if 0 <= f <= 1 and matches(value, tol, f * 100)]
    if not hits and float(value).is_integer() and 1900 <= value <= 2100:  # a year inside a date string
        hits = [p for p, t in facts_text if str(int(value)) in t]
    return hits


def extract_dates(sentence):
    """Find written dates; returns ([(phrase, [candidate strings like '2025-02-28' or '02-28'])], sentence_without_them)."""
    found = []
    for i, pat in enumerate(DATE_PATTERNS):
        for m in pat.finditer(sentence):
            g = m.groups()
            if i == 0:
                year, month, day = g[0], int(g[1]), int(g[2])
            elif i == 1:
                day, month, year = int(g[0]), MONTHS.get(g[1].lower().rstrip("."), None) or _short(g[1]), g[2]
            else:
                month, day, year = MONTHS.get(g[0].lower().rstrip("."), None) or _short(g[0]), int(g[1]), g[2]
            md = f"{month:02d}-{day:02d}"
            found.append((m.group(0), [f"{year}-{md}"] if year else [md]))
        sentence = pat.sub(" ", sentence)
    return found, sentence


def _short(name):
    return next(v for k, v in MONTHS.items() if k.startswith(name.lower().rstrip(".")[:3]))


def sentences_of(report):
    """Report text -> sentences, skipping headings and the code-written Sources section."""
    body = report.split("## Sources")[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^([-*]|\d+\.)\s+", "", line).replace("**", "")
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", line) if s.strip()]
    return out


def track_confidence(facts):
    """Each result's own confidence (mean_confidence where it has one). Model test metrics count as well-measured."""
    return {k: float(v.get("mean_confidence", 1.0)) if isinstance(v, dict) else 1.0 for k, v in facts.items()}


def validate(report, facts, sources, min_confidence=0.7):
    facts_num, facts_text = flatten(facts)
    all_text = " ".join(t for _, t in facts_text)
    by_id = {s["id"]: s["text"].replace(",", "") for s in sources}
    conf = track_confidence(facts)
    claims, flagged = [], []
    for sentence in sentences_of(report):
        cited = CITATION.findall(sentence)
        cited_text = " ".join(by_id.get(c, "") for c in cited)
        dates, rest = extract_dates(CITATION.sub(" ", sentence))
        rest = IGNORE.sub(" ", rest)
        items = []
        for phrase, candidates in dates:
            ok = any(c in all_text for c in candidates)
            items.append({"text": phrase, "status": "verified" if ok else "unsupported", "matched": candidates if ok else []})
        for m in NUMBER.finditer(rest):
            digits, decimals, unit_raw = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
            value = float(digits.replace(",", "") + decimals)
            unit = UNIT_FAMILY.get(unit_raw.replace("  ", " "), unit_raw)
            hits = check_number(value, rounding_tolerance(digits, decimals), unit, facts_num, facts_text)
            if hits:
                status = "verified"
            elif cited and (digits.replace(",", "") + decimals) in cited_text:
                status = "sourced"
            elif unit in ("ha", "m2", "%"):
                status = "mismatch"
            else:
                status = "unsupported"
            items.append({"text": m.group(0).strip(), "status": status, "matched": hits})
        reasons = []
        bad = [i["text"] for i in items if i["status"] in ("mismatch", "unsupported")]
        if bad:
            reasons.append("numbers not found in the facts or cited sources: " + ", ".join(bad))
        tracks = {h.split(".")[0] for i in items for h in i["matched"] if i["status"] == "verified"}
        weak = sorted(t for t in tracks if conf.get(t, 1.0) < min_confidence)
        if weak:
            reasons.append("low-confidence result (" + ", ".join(f"{t} confidence {conf[t]:.2f}" for t in weak) + ")")
        if CAUSAL.search(sentence) and not cited:
            reasons.append("causal claim without a cited source")
        if OVERCLAIM.search(sentence):
            reasons.append("overclaiming word: " + OVERCLAIM.search(sentence).group(0))
        claims.append({"sentence": sentence, "items": items, "citations": cited, "reasons": reasons})
        if reasons:
            flagged.append(claims[-1])
    statuses = [i["status"] for c in claims for i in c["items"]]
    summary = {s: statuses.count(s) for s in ("verified", "sourced", "mismatch", "unsupported")}
    summary.update(sentences=len(claims), flagged_for_review=len(flagged),
                   unknown_citations=sorted({c for cl in claims for c in cl["citations"] if c not in by_id}))
    return {"summary": summary, "flagged": flagged, "claims": claims}


def review_markdown(result):
    s = result["summary"]
    lines = ["# Report validation", "",
             f"Numbers and dates checked: {s['verified']} verified against the facts, {s['sourced']} found in cited guides, "
             f"**{s['mismatch']} mismatched**, **{s['unsupported']} unsupported**.",
             f"Sentences flagged for human review: **{s['flagged_for_review']}** of {s['sentences']}."]
    if s["unknown_citations"]:
        lines.append(f"Citations to sources that do not exist: **{', '.join(s['unknown_citations'])}**")
    lines += ["", "| # | Sentence | Why it needs review |", "|---|---|---|"]
    for i, c in enumerate(result["flagged"], 1):
        lines.append(f"| {i} | {c['sentence'].replace('|', '/')} | {'; '.join(c['reasons'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/report", help="folder with report.md and context.json")
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--strict", action="store_true", help="exit with an error if any number is mismatched or unsupported")
    args = ap.parse_args()
    d = Path(args.dir)
    context = json.loads((d / "context.json").read_text())
    result = validate((d / "report.md").read_text(), context["facts"], context["sources"], args.min_confidence)
    (d / "validation.json").write_text(json.dumps(result, indent=1))
    (d / "review.md").write_text(review_markdown(result))
    print(review_markdown(result))
    if args.strict and (result["summary"]["mismatch"] or result["summary"]["unsupported"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
