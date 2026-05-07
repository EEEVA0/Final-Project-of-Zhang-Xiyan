from openai import OpenAI
import os
import json

client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://one.ocoolai.com/v1"
)


def drafting_agent(innovation_data, cpc_label=None, kg_constraints: str = ""):
    innovation_text = (
        innovation_data if isinstance(innovation_data, str)
        else json.dumps(innovation_data, ensure_ascii=False)
    )

    cpc_info = (
        f"""
CPC Domain Constraint:
The invention has been classified by the local CPC classifier as: {cpc_label}.

You SHOULD:
- Keep the patent draft consistent with this CPC technical domain.
- Use terminology consistent with this CPC domain where relevant.
- Avoid unrelated domains or technologies not supported by the input.
"""
        if cpc_label
        else ""
    )

    prompt = f"""
You are a patent drafting expert for computer systems, information retrieval, machine learning, and software systems.

INPUT (innovation JSON/text):
{innovation_text}

{cpc_info}

{kg_constraints}

=====================
HARD CONSTRAINTS
=====================
You MUST output a single patent draft in plain text with the following headings exactly and in this order:
1. Technical Field
2. Background
3. Definitions
4. Summary of the Invention
5. Detailed Description
6. Claims

Do NOT output explanations outside the patent draft.

A) Definitions
- Provide 12–24 definitions.
- Each definition MUST follow exactly:
  In this invention, the term "X" refers to ...
- Definitions MUST cover:
  (1) all modules/features in the input,
  (2) all technical terms that will appear in the Claims,
  (3) key domain terms from the CPC or KG constraints only when they are relevant to the invention.
- Do NOT define generic words like "system", "data", or "model" unless you specify a concrete technical meaning.
- Do NOT introduce unrelated domains such as gene, patient, clinical, stock, bond, portfolio, or trading.

B) Detailed Description
Write 4–8 paragraphs describing the concrete mechanisms of the invention.

CRITICAL INPUT-FAITHFULNESS RULE:
- Only describe mechanisms that are explicitly present in the input or clearly implied by the Innovation Agent output.
- Do NOT add debiasing, IPS, SNIPS, Doubly Robust, UCB, Thompson sampling, exploration budget, TTL, LRU, LFU, p95/p99 latency, QPS, or caching mechanisms unless they are explicitly mentioned or clearly required by the invention.
- If a mechanism is not grounded in the input, do not add it to the Detailed Description or Claims.

Mechanism-specific guidance:
- If the input explicitly mentions debiasing, counterfactual learning, propensity score, position bias, exposure bias, IPS, SNIPS, or Doubly Robust learning, then describe:
  (a) how the relevant bias or propensity is estimated; and
  (b) one debiased objective used by the system.
- If the input explicitly mentions exploration, multi-armed bandit, UCB, Thompson sampling, uncertainty-based selection, or exploration budget, then describe:
  (a) the chosen exploration method;
  (b) what parameter is updated online; and
  (c) the exploration trigger or budget.
- If the input explicitly mentions caching, cache replacement, cache admission, TTL, LRU, LFU, latency, p95, p99, QPS, or delay-aware scheduling, then describe:
  (a) the cache policy, cache admission rule, or cache replacement rule; and
  (b) the latency-related or workload-related trigger.
- If the input mentions knowledge graph embeddings, specify how embeddings are computed, initialized, and used.
- If the input mentions prompt tuning, sentence embeddings, phrase alignment, or contrastive learning, focus on encoder structure, prompt tokens, alignment scoring, loss functions, and negative sampling.

C) Claims
- Provide 5–8 claims.
- Claim 1 must be an independent SYSTEM claim listing only essential modules grounded in the input.
- Claims 2–8 must be dependent claims, each adding one concrete implementation detail.
- Every term/module in Claims MUST be defined in Definitions or Detailed Description.
- Do NOT add a claim for a mechanism that is not grounded in the input.
- Keep terminology consistent across Definitions, Detailed Description, and Claims.

Now produce the full patent draft.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.30,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()