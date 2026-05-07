# compare_runs.py
# Usage:
#   python compare_runs.py run_baseline.json run_with_kg.json
# Output:
#   Prints a comparison report + saves compare_report.json

import json
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

# ---------------------------
# Config: dictionaries / lexicons
# ---------------------------

IR_RANKING_TERMS = {
    # IR / ranking core
    "information retrieval", "retrieval", "ranking", "learning to rank", "ltr",
    "rerank", "reranker", "cross-encoder", "dense retriever", "dense retrieval",
    "lexical retriever", "bm25", "embedding", "vector", "ann", "faiss",
    "query intent", "intent classification", "clickthrough", "ctr", "log",
    "propensity", "ips", "snips", "doubly robust", "counterfactual",
    "position bias", "exposure bias", "unbiased", "debias", "bandit", "ucb",
    "thompson", "cold start", "cache", "latency", "p95", "p99"
}

# "Off-topic" / drift detection vocabulary (expand as needed)
OFFTOPIC_TERMS = {
    # medical / biotech
    "diagnosis", "disease", "patient", "clinical", "drug", "gene", "genetic",
    "tumor", "cancer", "therapy", "radiology", "pharma",
    # finance / trading
    "stock", "bond", "derivative", "portfolio", "interest rate", "yield",
    "credit", "loan", "mortgage", "hedge", "trading", "exchange rate",
    # law / gov
    "election", "voting", "parliament", "constitution",
}

# words that indicate "mechanism/formula/constraints" (novelty density proxy)
MECHANISM_MARKERS = {
    "propensity", "ips", "snips", "doubly robust", "dr", "counterfactual",
    "loss", "objective", "optimize", "gradient", "regularization", "constraint",
    "slot", "schema", "ontology", "knowledge graph", "graph", "embedding",
    "ucb", "thompson", "bandit", "cache", "latency", "scheduling",
    "estimat", "calibrat", "weight", "reweight"
}

GENERIC_BENEFIT_MARKERS = {
    # signals of "just claims benefit" without mechanism
    "improve", "improved", "increase", "enhance", "better", "more accurate",
    "efficient", "robust", "stable", "reduce", "lower", "optimize performance",
    "user satisfaction"
}

# ---------------------------
# Helpers
# ---------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(x: str) -> str:
    x = x.lower()
    x = re.sub(r"[\u2010-\u2015]", "-", x)  # normalize dashes
    x = re.sub(r"\s+", " ", x).strip()
    return x

def extract_keywords(doc: Dict[str, Any]) -> List[str]:
    # Prefer parsed_details.keywords if available
    kws = []
    parsed = doc.get("parsed_info", {}).get("parsed_details", {})
    if isinstance(parsed, dict):
        kws = parsed.get("keywords") or []
    if not kws:
        # fallback: innovation JSON block keywords (string embedded)
        inv = doc.get("innovation", "")
        m = re.search(r'"keywords"\s*:\s*\[(.*?)\]', inv, flags=re.S | re.I)
        if m:
            inside = m.group(1)
            kws = re.findall(r'"([^"]+)"', inside)
    return [str(k) for k in kws if str(k).strip()]

def extract_claims_text(doc: Dict[str, Any]) -> str:
    # claims are inside draft/reviewed sections
    # We try to pull from "draft" first; if missing use "reviewed"
    text = doc.get("draft") or doc.get("reviewed") or ""
    return str(text)

def extract_description_text(doc: Dict[str, Any]) -> str:
    # Use the "draft" detailed description + background if present
    text = doc.get("draft") or ""
    return str(text)

def sentence_split(text: str) -> List[str]:
    text = text.replace("\n", " ")
    # split on sentence-ish boundaries
    parts = re.split(r"(?<=[.!?。；;:])\s+", text)
    # cleanup
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def tokenize_terms(text: str) -> List[str]:
    # very light tokenization for counting
    text = normalize_text(text)
    # keep hyphenated / slash words
    tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", text)
    return tokens

def contains_any(text: str, termset: set) -> bool:
    t = normalize_text(text)
    return any(term in t for term in termset)

def count_term_hits(text: str, termset: set) -> int:
    t = normalize_text(text)
    return sum(1 for term in termset if term in t)

def safe_get_solution_modules(doc: Dict[str, Any]) -> List[str]:
    parsed = doc.get("parsed_info", {}).get("parsed_details", {})
    if not isinstance(parsed, dict):
        return []
    sol = parsed.get("proposed_solution")
    if isinstance(sol, list):
        return [str(x).strip() for x in sol if str(x).strip()]
    if isinstance(sol, str):
        # split by ; , newline
        chunks = re.split(r"[;\n,]+", sol)
        return [c.strip() for c in chunks if c.strip()]
    return []

def module_in_text(module: str, text: str) -> bool:
    # allow fuzzy matching by core words
    m = normalize_text(module)
    t = normalize_text(text)
    if m in t:
        return True
    # fuzzy: require at least 2 meaningful tokens present
    toks = [w for w in tokenize_terms(m) if w not in {"with", "and", "or", "module", "system", "framework"}]
    toks = [w for w in toks if len(w) >= 4]
    if len(toks) == 0:
        return False
    hits = sum(1 for w in toks if w in t)
    return hits >= min(2, len(toks))  # at least 2 hits, or all if <2

def extract_defined_terms_from_description(desc_text: str) -> set:
    """
    Recognize patent-style definitions:
      In this invention, the term "X" refers to ...
      the term "X" refers to ...
    """

    t = desc_text
    defined = set()
    # 1) capture quoted term definitions
    for m in re.finditer(r'the term\s+"([^"]{2,80})"\s+refers\s+to', t, flags=re.I):
        defined.add(normalize_text(m.group(1).strip()))

    # 2) also support single quotes (optional)
    for m in re.finditer(r"the term\s+'([^']{2,80})'\s+refers\s+to", t, flags=re.I):
        defined.add(normalize_text(m.group(1).strip()))

    # 3) keep your old patterns too (optional)
    patterns = [
        r"\b([A-Z][A-Za-z0-9\- ]{2,60})\s+is\s+",
        r"\b([A-Z][A-Za-z0-9\- ]{2,60})\s+refers\s+to\s+",
        r"\b([A-Z][A-Za-z0-9\- ]{2,60})\s+is\s+configured\s+to\s+",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            phrase = m.group(1).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            if 3 <= len(phrase) <= 60:
                defined.add(normalize_text(phrase))

    return defined

def extract_candidate_terms_from_claims(claims_text: str) -> set:
    """
    Extract capitalized technical phrases (heuristic) and key tokens.
    """
    cands = set()
    # capitalized phrases
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})\b", claims_text):
        phrase = m.group(1).strip()
        phrase_n = normalize_text(phrase)
        if len(phrase_n) >= 4 and phrase_n not in {"the present invention"}:
            cands.add(phrase_n)

    # also include important lower-case terms by dictionary markers
    lt = normalize_text(claims_text)
    for term in IR_RANKING_TERMS.union(MECHANISM_MARKERS):
        if term in lt:
            cands.add(term)
    return cands

def novelty_density(desc_text: str) -> Dict[str, Any]:
    """
    Proxy metric:
      - mechanism sentences: contain any MECHANISM_MARKERS
      - generic benefit sentences: contain any GENERIC_BENEFIT_MARKERS but no mechanism markers
    density = mechanism_sentences / total_sentences
    """
    sents = sentence_split(desc_text)
    if not sents:
        return {"total_sentences": 0, "mechanism_sentences": 0, "generic_benefit_only": 0, "density": 0.0}

    mech = 0
    generic_only = 0
    for s in sents:
        has_mech = contains_any(s, MECHANISM_MARKERS)
        has_gen = contains_any(s, GENERIC_BENEFIT_MARKERS)
        if has_mech:
            mech += 1
        elif has_gen:
            generic_only += 1

    return {
        "total_sentences": len(sents),
        "mechanism_sentences": mech,
        "generic_benefit_only": generic_only,
        "density": round(mech / max(1, len(sents)), 4),
    }

def anchoring_rate(keywords: List[str]) -> Dict[str, Any]:
    if not keywords:
        return {"keyword_count": 0, "hit_count": 0, "rate": 0.0, "hits": [], "misses": []}

    hits, misses = [], []
    for kw in keywords:
        nkw = normalize_text(kw)
        # hit if keyword itself or its core tokens overlap with IR terms
        if nkw in IR_RANKING_TERMS:
            hits.append(kw)
        else:
            # token overlap
            toks = set(tokenize_terms(nkw))
            ir_toks = set()
            for term in IR_RANKING_TERMS:
                ir_toks |= set(tokenize_terms(term))
            if len(toks & ir_toks) >= 1:
                hits.append(kw)
            else:
                misses.append(kw)

    rate = len(hits) / max(1, len(keywords))
    return {
        "keyword_count": len(keywords),
        "hit_count": len(hits),
        "rate": round(rate, 4),
        "hits": hits,
        "misses": misses,
    }

def module_coverage(mods: List[str], claims_text: str, desc_text: str) -> Dict[str, Any]:
    covered_claims = []
    covered_desc = []
    missing = []

    for m in mods:
        in_c = module_in_text(m, claims_text)
        in_d = module_in_text(m, desc_text)
        if in_c:
            covered_claims.append(m)
        if in_d:
            covered_desc.append(m)
        if not (in_c or in_d):
            missing.append(m)

    return {
        "module_count": len(mods),
        "covered_in_claims": covered_claims,
        "covered_in_description": covered_desc,
        "missing": missing,
        "coverage_rate_anywhere": round((len(mods) - len(missing)) / max(1, len(mods)), 4),
        "coverage_rate_claims": round(len(covered_claims) / max(1, len(mods)), 4),
        "coverage_rate_description": round(len(covered_desc) / max(1, len(mods)), 4),
    }

def term_consistency(claims_text: str, desc_text: str) -> Dict[str, Any]:
    defined = extract_defined_terms_from_description(desc_text)
    cands = extract_candidate_terms_from_claims(claims_text)

    # term is "undefined" if it appears in claims but not in defined set and not a very common filler
    stop = {
        "system", "method", "module", "framework", "platform", "network", "data", "model",
        "ranking", "retrieval",
        # add these to kill noise:
        "background", "technical field", "definitions", "detailed description", "claims",
        "additionally", "current"
    }
    undefined = []
    for term in sorted(cands):
        toks = set(tokenize_terms(term))
        if toks <= stop:
            continue
        if term not in defined:
            undefined.append(term)

    return {
        "defined_terms_count": len(defined),
        "candidate_terms_in_claims": len(cands),
        "undefined_terms_count": len(undefined),
        "undefined_terms_sample": undefined[:30],  # cap for readability
    }

def drift_detection(text: str) -> Dict[str, Any]:
    t = normalize_text(text)
    hits = []
    for term in OFFTOPIC_TERMS:
        if re.search(rf"\\b{re.escape(term)}\\b", t):
            hits.append(term)
    return {
        "offtopic_hit_count": len(hits),
        "offtopic_hits": sorted(hits),
    }

def compare(base: Dict[str, Any], kg: Dict[str, Any]) -> Dict[str, Any]:
    base_kws = extract_keywords(base)
    kg_kws = extract_keywords(kg)

    base_claims = extract_claims_text(base)
    kg_claims = extract_claims_text(kg)

    base_desc = extract_description_text(base)
    kg_desc = extract_description_text(kg)

    base_mods = safe_get_solution_modules(base)
    kg_mods = safe_get_solution_modules(kg)

    report = {
        "inputs": {
            "baseline": {"keywords": base_kws, "module_list": base_mods},
            "with_kg": {"keywords": kg_kws, "module_list": kg_mods},
        },
        "metrics": {
            "anchoring_rate": {
                "baseline": anchoring_rate(base_kws),
                "with_kg": anchoring_rate(kg_kws),
            },
            "term_consistency": {
                "baseline": term_consistency(base_claims, base_desc),
                "with_kg": term_consistency(kg_claims, kg_desc),
            },
            "module_coverage": {
                "baseline": module_coverage(base_mods, base_claims, base_desc),
                "with_kg": module_coverage(kg_mods, kg_claims, kg_desc),
            },
            "drift_detection": {
                "baseline": drift_detection(base_claims + "\n" + base_desc),
                "with_kg": drift_detection(kg_claims + "\n" + kg_desc),
            },
            "novelty_density": {
                "baseline": novelty_density(base_desc),
                "with_kg": novelty_density(kg_desc),
            }
        }
    }
    return report

def winner_hint(report: Dict[str, Any]) -> List[str]:
    m = report["metrics"]
    hints = []

    ar_b = m["anchoring_rate"]["baseline"]["rate"]
    ar_k = m["anchoring_rate"]["with_kg"]["rate"]
    if ar_k > ar_b:
        hints.append(f"✅ KG wins on domain anchoring rate: {ar_k:.3f} > {ar_b:.3f}")
    elif ar_k < ar_b:
        hints.append(f"✅ Baseline wins on domain anchoring rate: {ar_b:.3f} > {ar_k:.3f}")
    else:
        hints.append(f"🤝 Domain anchoring rate tie: {ar_b:.3f}")

    tc_b = m["term_consistency"]["baseline"]["undefined_terms_count"]
    tc_k = m["term_consistency"]["with_kg"]["undefined_terms_count"]
    if tc_k < tc_b:
        hints.append(f"✅ KG wins on term consistency (fewer undefined terms): {tc_k} < {tc_b}")
    elif tc_k > tc_b:
        hints.append(f"✅ Baseline wins on term consistency: {tc_b} < {tc_k}")
    else:
        hints.append(f"🤝 Term consistency tie (undefined terms): {tc_b}")

    mc_b = m["module_coverage"]["baseline"]["coverage_rate_anywhere"]
    mc_k = m["module_coverage"]["with_kg"]["coverage_rate_anywhere"]
    if mc_k > mc_b:
        hints.append(f"✅ KG wins on module coverage: {mc_k:.3f} > {mc_b:.3f}")
    elif mc_k < mc_b:
        hints.append(f"✅ Baseline wins on module coverage: {mc_b:.3f} > {mc_k:.3f}")
    else:
        hints.append(f"🤝 Module coverage tie: {mc_b:.3f}")

    dr_b = m["drift_detection"]["baseline"]["offtopic_hit_count"]
    dr_k = m["drift_detection"]["with_kg"]["offtopic_hit_count"]
    if dr_k < dr_b:
        hints.append(f"✅ KG has less off-topic drift: {dr_k} < {dr_b}")
    elif dr_k > dr_b:
        hints.append(f"✅ Baseline has less off-topic drift: {dr_b} < {dr_k}")
    else:
        hints.append(f"🤝 Off-topic drift tie: {dr_b}")

    nd_b = m["novelty_density"]["baseline"]["density"]
    nd_k = m["novelty_density"]["with_kg"]["density"]
    if nd_k > nd_b:
        hints.append(f"✅ KG has higher novelty density: {nd_k:.3f} > {nd_b:.3f}")
    elif nd_k < nd_b:
        hints.append(f"✅ Baseline has higher novelty density: {nd_b:.3f} > {nd_k:.3f}")
    else:
        hints.append(f"🤝 Novelty density tie: {nd_b:.3f}")

    return hints

def pretty_print(report: Dict[str, Any]) -> None:
    m = report["metrics"]

    print("\n================= Patent A/B Compare Report =================")
    print("\n[1] Domain Anchoring Rate (keywords hit IR/Ranking lexicon)")
    b = m["anchoring_rate"]["baseline"]
    k = m["anchoring_rate"]["with_kg"]
    print(f"  Baseline: {b['hit_count']}/{b['keyword_count']} = {b['rate']:.3f}")
    print(f"  With KG : {k['hit_count']}/{k['keyword_count']} = {k['rate']:.3f}")
    if k["misses"]:
        print(f"  With KG misses: {k['misses']}")

    print("\n[2] Term Consistency (claims terms not defined in description) [heuristic]")
    b = m["term_consistency"]["baseline"]
    k = m["term_consistency"]["with_kg"]
    print(f"  Baseline undefined terms: {b['undefined_terms_count']}")
    if b["undefined_terms_sample"]:
        print(f"    sample: {b['undefined_terms_sample'][:12]}")
    print(f"  With KG undefined terms : {k['undefined_terms_count']}")
    if k["undefined_terms_sample"]:
        print(f"    sample: {k['undefined_terms_sample'][:12]}")

    print("\n[3] Module Coverage (proposed_solution modules appear in claims/description)")
    b = m["module_coverage"]["baseline"]
    k = m["module_coverage"]["with_kg"]
    print(f"  Baseline coverage(anywhere): {b['coverage_rate_anywhere']:.3f}  (claims {b['coverage_rate_claims']:.3f}, desc {b['coverage_rate_description']:.3f})")
    if b["missing"]:
        print(f"    missing: {b['missing']}")
    print(f"  With KG  coverage(anywhere): {k['coverage_rate_anywhere']:.3f}  (claims {k['coverage_rate_claims']:.3f}, desc {k['coverage_rate_description']:.3f})")
    if k["missing"]:
        print(f"    missing: {k['missing']}")

    print("\n[4] Drift Detection (off-topic vocabulary hits)")
    b = m["drift_detection"]["baseline"]
    k = m["drift_detection"]["with_kg"]
    print(f"  Baseline off-topic hits: {b['offtopic_hit_count']}  {b['offtopic_hits']}")
    print(f"  With KG  off-topic hits: {k['offtopic_hit_count']}  {k['offtopic_hits']}")

    print("\n[5] Novelty Density (mechanism/formula/constraint sentences / total) [proxy]")
    b = m["novelty_density"]["baseline"]
    k = m["novelty_density"]["with_kg"]
    print(f"  Baseline: {b['mechanism_sentences']}/{b['total_sentences']} = {b['density']:.3f}  (generic-only {b['generic_benefit_only']})")
    print(f"  With KG : {k['mechanism_sentences']}/{k['total_sentences']} = {k['density']:.3f}  (generic-only {k['generic_benefit_only']})")

    print("\n[Winner hints]")
    for h in winner_hint(report):
        print(" ", h)

    print("============================================================\n")

def main():
    # 默认文件名（和 ab_runner.py 里保存的一致）
    base_path = "run_baseline.json"
    kg_path = "run_with_kg.json"

    try:
        base = load_json(base_path)
        kg = load_json(kg_path)
    except FileNotFoundError:
        print("❌ 找不到 run_baseline.json 或 run_with_kg.json")
        print("请确认你已经先运行 ab_runner.py")
        return

    report = compare(base, kg)
    pretty_print(report)

    with open("compare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved: compare_report.json")

if __name__ == "__main__":
    main()