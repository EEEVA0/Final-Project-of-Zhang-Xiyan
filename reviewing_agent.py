from openai import OpenAI
import os

client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://one.ocoolai.com/v1"
)


def reviewing_agent(draft_text, kg_constraints: str = ""):
    prompt = f"""
You are a Patent Examiner and Patent Lint Tool.

TASK:
Perform a strict consistency audit and output a revised version.

The goal is NOT to add more mechanisms. The goal is to make the draft faithful to the original invention, internally consistent, and patent-style.

A) Term consistency lint:
- Extract technical terms and modules appearing in the Claims.
- Verify each term is defined in Definitions or Detailed Description.
- Output UndefinedTerms list with actions:
  ADD_DEFINITION(term) / REPLACE(term -> defined_term) / REMOVE(term).

B) Domain drift check:
- Remove unrelated domain words or mechanisms not supported by the invention.
- Forbidden unrelated examples: gene, patient, clinical, tumor, cancer, stock, bond, portfolio, trading.
- Also remove technical mechanisms that are not grounded in the draft's stated invention field.

C) Input-faithfulness and mechanism audit:
- Review only the mechanisms that already appear in the draft.
- Do NOT require missing mechanism types to be added.
- Do NOT add debiasing, IPS, SNIPS, Doubly Robust, UCB, Thompson sampling, exploration budget, TTL, LRU, LFU, p95/p99 latency, QPS, caching, or latency mechanisms unless they are clearly grounded in the invention described by the draft.
- If the draft introduces an ungrounded mechanism, mark it as REMOVE or revise it out of the patent text.

Mechanism-specific checks:
- If debiasing, counterfactual learning, propensity score, position bias, exposure bias, IPS, SNIPS, or Doubly Robust learning appears and is relevant to the invention, check whether the draft specifies the bias/propensity estimation and the debiased objective.
- If exploration, multi-armed bandit, UCB, Thompson sampling, uncertainty-based selection, or exploration budget appears and is relevant to the invention, check whether the draft specifies the algorithm, online update parameter, and trigger or budget.
- If caching, cache replacement, cache admission, TTL, LRU, LFU, latency, p95, p99, QPS, or delay-aware scheduling appears and is relevant to the invention, check whether the draft specifies the cache policy and latency/workload trigger.
- If any of these mechanisms appear without being relevant to the invention, remove them rather than strengthening them.

D) Section structure:
- Keep the following headings:
  1. Technical Field
  2. Background
  3. Definitions
  4. Summary of the Invention
  5. Detailed Description
  6. Claims

OUTPUT FORMAT:
1) UndefinedTerms: [...]
2) RemovedOrRevisedUngroundedMechanisms: [...]
3) RevisedDraft:
(full revised patent text, keeping the required headings and a Definitions section)

{kg_constraints}

--- DRAFT ---
{draft_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()