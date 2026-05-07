import re
import json
from openai import OpenAI

client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://one.ocoolai.com/v1"
)


def safe_parse_json(text):
    """
    从模型返回的文本中安全提取 JSON。
    支持带前后说明文字的情况，例如：
    'Here is the structured info: { ... }'
    """
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            json_str = match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = (
                json_str
                .replace("'", '"')  # 单引号转双引号
                .replace("\n", " ")
                .replace(", }", "}")
                .replace(",]", "]")
            )
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return {"raw_text": text.strip()}


def input_parser_agent(project_text):
    """
    将自然语言项目描述解析成结构化信息。
    自动生成 title、abstract、description，以适配主程序。
    """
    prompt = f"""
    You are an expert technical analyst.

    TASK:
    Analyze the following project description and extract structured technical information.

    Respond ONLY in pure JSON format (no markdown, no explanations).

    JSON Schema:
    {{
        "technical_field": "field of technology",
        "core_device_or_system": "main system or product",
        "technical_problem": "key problem addressed",
        "proposed_solution": "core technical solution",
        "expected_effect": "benefit or effect",
        "target_users": ["..."],
        "keywords": ["...", "...", "..."]
    }}

    --- PROJECT DESCRIPTION ---
    {project_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    output_text = response.choices[0].message.content.strip()
    parsed = safe_parse_json(output_text)

    # 如果模型没生成字段，就创建默认值
    parsed.setdefault("technical_field", "")
    parsed.setdefault("core_device_or_system", "")
    parsed.setdefault("technical_problem", "")
    parsed.setdefault("proposed_solution", "")
    parsed.setdefault("expected_effect", "")
    parsed.setdefault("target_users", [])
    parsed.setdefault("keywords", [])

    # ✅ 自动生成 title / abstract / description 给后续 agent 使用
    title = parsed["core_device_or_system"] or parsed["technical_field"] or "Unnamed Invention"
    abstract = (
        f"A {parsed['core_device_or_system']} designed for {parsed['technical_field']} "
        f"to solve {parsed['technical_problem']} through {parsed['proposed_solution']}."
    )
    description = (
        f"The invention relates to {parsed['technical_field']}. "
        f"It focuses on {parsed['core_device_or_system']}, "
        f"which aims to address {parsed['technical_problem']}. "
        f"The proposed solution involves {parsed['proposed_solution']}, "
        f"resulting in {parsed['expected_effect']}. "
        f"Target users include {', '.join(parsed['target_users']) or 'general users'}. "
        f"Keywords: {', '.join(parsed['keywords']) or 'N/A'}."
    )

    # ✅ 返回兼容主程序结构
    return {
        "title": title.strip(),
        "abstract": abstract.strip(),
        "description": description.strip(),
        "parsed_details": parsed
    }
