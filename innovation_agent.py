from openai import OpenAI
import os

client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://one.ocoolai.com/v1"
)


def innovation_agent(title, abstract, description, cpc_label=None, kg_constraints: str = ""):
    cpc_info = (
        f"""
CPC Domain Constraint:
The invention has been classified by the local CPC classifier as: {cpc_label}.

You MUST:
- Keep the innovation analysis within this CPC technical domain.
- Prefer terminology consistent with this CPC domain.
- Avoid introducing unrelated technical fields.
"""
        if cpc_label
        else ""
    )

    prompt = f"""
You are a Senior Patent Analyst with 10+ years of experience.

TASK:
Analyze the following patent and extract:
1. Core technical innovations
2. Key technical features
3. Problem solved and technical effect
4. 3–8 most relevant technical keywords

Respond in JSON format with fields:
{{
    "innovation_summary": "...",
    "technical_problem": "...",
    "technical_solution": "...",
    "technical_effect": "...",
    "keywords": ["...", "...", "..."]
}}

{cpc_info}

{kg_constraints}

--- PATENT INPUT ---
Title: {title}
Abstract: {abstract}
Description: {description}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()