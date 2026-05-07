# ab_metric_checker.py
import json
import re
from pathlib import Path


REQUIRED_SECTIONS = [
    "Technical Field",
    "Background",
    "Definitions",
    "Summary of the Invention",
    "Detailed Description",
    "Claims",
]

MECHANISM_TERMS = [
    "inverse propensity scoring",
    "IPS",
    "propensity",
    "position bias",
    "UCB",
    "Upper Confidence Bound",
    "exploration budget",
    "TTL",
    "LRU",
    "p95",
    "cache",
    "knowledge graph",
    "embedding",
    "cross-encoder",
    "reranker",
]

FORBIDDEN_DOMAIN_TERMS = [
    "gene", "patient", "clinical", "stock", "bond", "medical", "diagnosis"
]

CLAIM_MODULE_HINTS = [
    "module", "system", "architecture", "retriever", "reranker",
    "cache", "mechanism", "model", "algorithm", "embedding"
]


def load_text(path: str, field: str | None = None) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if field:
            return str(data.get(field, ""))
        return json.dumps(data, ensure_ascii=False, indent=2)

    return p.read_text(encoding="utf-8")


def section_completeness(text: str) -> dict:
    found = {}
    lower = text.lower()
    for sec in REQUIRED_SECTIONS:
        found[sec] = sec.lower() in lower
    score = sum(found.values()) / len(REQUIRED_SECTIONS)
    return {"score": round(score, 3), "found": found}


def extract_definitions(text: str) -> set:
    definitions = set()

    # Pattern 1: In this invention, the term "X" refers to ...
    for m in re.finditer(r'term\s+"([^"]+)"\s+refers to', text, re.IGNORECASE):
        definitions.add(m.group(1).strip().lower())

    # Pattern 2: Markdown bullet style: **X:** ...
    for m in re.finditer(r'\*\*([^:*]{2,80})\*\*\s*:', text):
        definitions.add(m.group(1).strip().lower())

    return definitions


def extract_claims(text: str) -> list[str]:
    claims_section_match = re.search(
        r'(?:^|\n)\s*(?:#{1,6}\s*)?(?:\d+\.\s*)?Claims\s*(.*)$',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if not claims_section_match:
        return []

    claims_text = claims_section_match.group(1)

    claim_blocks = re.split(
        r'\n\s*(?:Claim\s*)?\d+\s*[\.:]',
        claims_text,
        flags=re.IGNORECASE
    )

    claims = [c.strip() for c in claim_blocks if len(c.strip()) > 30]
    return claims


def extract_claim_terms(claims: list[str]) -> set:
    terms = set()

    for claim in claims:
        # 抓 “xxx module/system/mechanism/model/cache/retriever/reranker”
        pattern = r'\b([A-Za-z][A-Za-z0-9\- ]{1,60}\s+(?:' + "|".join(CLAIM_MODULE_HINTS) + r'))\b'
        for m in re.finditer(pattern, claim, re.IGNORECASE):
            term = re.sub(r'\s+', ' ', m.group(1).strip().lower())
            if len(term.split()) <= 8:
                terms.add(term)

    return terms


def definition_coverage(text: str) -> dict:
    definitions = extract_definitions(text)
    claims = extract_claims(text)
    claim_terms = extract_claim_terms(claims)

    if not claim_terms:
        return {
            "score": 0.0,
            "claim_terms": [],
            "defined_terms": sorted(definitions),
            "undefined_terms": []
        }

    covered = []
    undefined = []

    for term in claim_terms:
        matched = False
        for d in definitions:
            if term in d or d in term:
                matched = True
                break
        if matched:
            covered.append(term)
        else:
            undefined.append(term)

    score = len(covered) / len(claim_terms) if claim_terms else 0

    return {
        "score": round(score, 3),
        "claim_terms": sorted(claim_terms),
        "defined_terms": sorted(definitions),
        "undefined_terms": sorted(undefined)
    }


def mechanism_coverage(text: str) -> dict:
    lower = text.lower()
    found = []

    for term in MECHANISM_TERMS:
        if term.lower() in lower:
            found.append(term)

    score = len(found) / len(MECHANISM_TERMS)

    return {
        "score": round(score, 3),
        "found_terms": found,
        "missing_terms": [t for t in MECHANISM_TERMS if t not in found]
    }


def domain_drift(text: str) -> dict:
    lower = text.lower()
    found = [t for t in FORBIDDEN_DOMAIN_TERMS if re.search(rf'\b{re.escape(t)}\b', lower)]

    return {
        "score": 1.0 if not found else 0.0,
        "forbidden_terms_found": found
    }


def claim_count(text: str) -> dict:
    claims = extract_claims(text)
    n = len(claims)
    score = 1.0 if 5 <= n <= 8 else max(0.0, 1 - abs(n - 6.5) / 6.5)

    return {
        "score": round(score, 3),
        "claim_count": n
    }


def claim_support_rate(text: str) -> dict:
    claims = extract_claims(text)
    claim_terms = extract_claim_terms(claims)

    desc_match = re.search(
        r'(?:Detailed Description)(.*?)(?:Claims|$)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    desc_text = desc_match.group(1).lower() if desc_match else text.lower()

    if not claim_terms:
        return {
            "score": 0.0,
            "supported_terms": [],
            "unsupported_terms": []
        }

    supported = []
    unsupported = []

    for term in claim_terms:
        # 粗略支持：claim 里的模块名在 detailed description 出现
        core = term.lower()
        if core in desc_text:
            supported.append(term)
        else:
            # 去掉泛化尾词再查
            simplified = re.sub(
                r'\b(module|system|architecture|mechanism|model|algorithm)\b',
                '',
                core
            ).strip()
            if simplified and simplified in desc_text:
                supported.append(term)
            else:
                unsupported.append(term)

    score = len(supported) / len(claim_terms)

    return {
        "score": round(score, 3),
        "supported_terms": sorted(supported),
        "unsupported_terms": sorted(unsupported)
    }


def evaluate_text(name: str, text: str) -> dict:
    result = {
        "name": name,
        "section_completeness": section_completeness(text),
        "definition_coverage": definition_coverage(text),
        "mechanism_coverage": mechanism_coverage(text),
        "domain_drift": domain_drift(text),
        "claim_count": claim_count(text),
        "claim_support_rate": claim_support_rate(text),
    }

    weighted_score = (
        result["section_completeness"]["score"] * 0.15 +
        result["definition_coverage"]["score"] * 0.20 +
        result["mechanism_coverage"]["score"] * 0.25 +
        result["domain_drift"]["score"] * 0.10 +
        result["claim_count"]["score"] * 0.10 +
        result["claim_support_rate"]["score"] * 0.20
    )

    result["overall_rule_score"] = round(weighted_score, 3)
    return result


def print_summary(results: list[dict]):
    print("\n===== Ablation Metric Summary =====")
    print(
        f"{'Method':<18} "
        f"{'Section':>8} "
        f"{'DefCov':>8} "
        f"{'Mechanism':>10} "
        f"{'NoDrift':>8} "
        f"{'Claims':>8} "
        f"{'Support':>8} "
        f"{'Overall':>8}"
    )

    for r in results:
        print(
            f"{r['name']:<18} "
            f"{r['section_completeness']['score']:>8.3f} "
            f"{r['definition_coverage']['score']:>8.3f} "
            f"{r['mechanism_coverage']['score']:>10.3f} "
            f"{r['domain_drift']['score']:>8.3f} "
            f"{r['claim_count']['score']:>8.3f} "
            f"{r['claim_support_rate']['score']:>8.3f} "
            f"{r['overall_rule_score']:>8.3f}"
        )


def main():
    # 按你的实际文件名修改这里
    single_text = load_text("single_agent_output.txt")
    multi_text = load_text("multi_agent_output.txt")

    results = [
        evaluate_text("Single-Agent", single_text),
        evaluate_text("Multi-Agent", multi_text),
    ]

    print_summary(results)

    with open("ab_metric_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved detailed results to ab_metric_results.json")


if __name__ == "__main__":
    main()