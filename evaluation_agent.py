import re
import json
from openai import OpenAI

client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://one.ocoolai.com/v1"
)

def safe_parse_json(text):
    """安全提取 JSON 内容"""
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            fixed = (
                match.group(0)
                .replace("'", '"')
                .replace("\n", " ")
                .replace(", }", "}")
                .replace(",]", "]")
            )
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    # 如果完全不是 JSON，就返回一个空结构
    return {"raw_text": text.strip()}


def evaluation_agent(patent_text):
    """
    评估生成的专利文档质量，输出结构化结果（dict）
    """
    prompt = f"""
    You are a professional patent examiner.

    Evaluate the following patent draft based on the following criteria.
    Give each a score (1–10) and a one-sentence reason.
    Respond **only** in pure JSON format, no explanations, no markdown.

    JSON schema:
    {{
      "clarity": {{"score": int, "reason": "..." }},
      "novelty": {{"score": int, "reason": "..." }},
      "technical_depth": {{"score": int, "reason": "..." }},
      "practical_value": {{"score": int, "reason": "..." }},
      "language_quality": {{"score": int, "reason": "..." }}
    }}

    --- PATENT DRAFT ---
    {patent_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    output_text = response.choices[0].message.content.strip()
    parsed = safe_parse_json(output_text)

    # ✅ 确保始终返回 dict 而不是字符串
    if not isinstance(parsed, dict):
        parsed = {"raw_text": output_text}

    return parsed
