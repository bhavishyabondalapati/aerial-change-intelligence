"""Write a plain-language report with Gemini, grounded in our JSON results and in passages from the agriculture guides.

Grounding rules: numbers may only come from FACTS (the JSON outputs), advice must cite a numbered guide passage,
and the Sources list is written by this code, not by the model. Everything the model saw is saved to context.json
so the validation step (aci.validate) can check the report against exactly that.
"""
import argparse
import json
import os
from pathlib import Path

from aci.rag import gemini_embedder, load_index, retrieve

GEN_MODEL = "gemini-3.8-flash"  # gemini-2.5-flash is no longer offered to new API keys

# Where each result comes from; missing files are skipped so the report works with whatever has been run
FACT_FILES = {
    "change_detection_test_metrics": "metrics.json",
    "building_change_example": "analysis/area_summary.json",
    "vegetation_loss_two_dates": "crop_stress/crop_stress_summary.json",
    "seasonal_anomaly": "seasonal/seasonal_summary.json",
}

# Questions used to pull guide passages; they follow what the results could mean and what a farmer could do next
QUESTIONS = [
    "causes of a drop in rice crop greenness or canopy during the rabi dry season in February and March",
    "managing water shortage, delayed canal release or drought for rabi rice in East Godavari district",
    "rice diseases and nutrient deficiencies that cause yellowing, drying or thin canopy",
    "how to scout and confirm crop problems in the field before taking action",
    "conversion of paddy fields to aquaculture ponds and salinity in the Godavari delta",
]

SYSTEM = """You write short, careful reports for agricultural officers in the Godavari delta, India.
Rules:
1. Every number you write must be copied from FACTS (you may round it). Do not calculate new numbers:
   no sums, differences, ratios or unit conversions that are not already in FACTS.
2. Any agronomic explanation or advice must cite the passage it comes from as [S1], [S2] ... using SOURCES only.
   If a number comes from a passage (e.g. a fertiliser dose), it must appear in that passage.
3. Satellite NDVI drops are signals, not proof of crop stress. Say "possible", never "confirmed".
   The crop tracks have no ground truth; their confidence values are scaled NDVI drops, not probabilities.
4. If something is not covered by FACTS or SOURCES, say it is unknown. Do not use outside knowledge.
5. Do not write a Sources or References section; it is added automatically.
Format: Markdown with these sections: ## Summary, ## Building change detection, ## Vegetation change in the
Godavari delta, ## Possible causes, ## Recommended field checks, ## Limitations. Keep it under 600 words."""


def gather_facts(outputs_dir):
    facts = {}
    for name, rel in FACT_FILES.items():
        path = Path(outputs_dir) / rel
        if path.exists():
            facts[name] = json.loads(path.read_text())
    if not facts:
        raise FileNotFoundError(f"No result JSON files found in {outputs_dir}; run the pipelines first")
    return facts


def gather_sources(chunks, vectors, embed, questions=QUESTIONS, per_question=4):
    """Top passages for each question, duplicates removed, numbered S1, S2, ..."""
    seen, sources = set(), []
    for q in questions:
        for hit in retrieve(q, chunks, vectors, embed, k=per_question):
            key = (hit["source"], hit["page"], hit["text"][:80])
            if key not in seen:
                seen.add(key)
                sources.append(dict(hit, id=f"S{len(sources) + 1}", question=q))
    return sources


def build_prompt(facts, sources):
    passages = "\n\n".join(f"[{s['id']}] ({s['source']}, page {s['page']})\n{s['text']}" for s in sources)
    return f"FACTS (JSON results of our analysis):\n{json.dumps(facts, indent=1)}\n\nSOURCES:\n{passages}\n\nWrite the report."


def sources_section(sources):
    lines = [f"- **[{s['id']}]** {s['source']}, page {s['page']}" for s in sources]
    return "## Sources\n\n" + "\n".join(lines) + "\n"


def gemini_generator(model=GEN_MODEL, api_key=None):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def generate(prompt, system):
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.2)
        return client.models.generate_content(model=model, contents=prompt, config=config).text

    return generate


def write_report(outputs_dir, index_dir, out_dir, embed, generate, model_name=GEN_MODEL):
    facts = gather_facts(outputs_dir)
    chunks, vectors = load_index(index_dir)
    sources = gather_sources(chunks, vectors, embed)
    prompt = build_prompt(facts, sources)
    body = generate(prompt, SYSTEM).strip()
    report = f"# Change intelligence report: Godavari delta\n\n{body}\n\n{sources_section(sources)}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report)
    context = {"model": model_name, "system": SYSTEM, "facts": facts, "sources": sources}
    (out / "context.json").write_text(json.dumps(context, indent=1))
    return report


def main():
    from dotenv import load_dotenv

    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="outputs/report")
    ap.add_argument("--model", default=GEN_MODEL)
    args = ap.parse_args()
    report = write_report(args.outputs, args.index, args.out, gemini_embedder("RETRIEVAL_QUERY"),
                          gemini_generator(args.model), args.model)
    print(report)


if __name__ == "__main__":
    main()
