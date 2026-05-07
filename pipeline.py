from input_parser_agent import input_parser_agent
from innovation_agent import innovation_agent
from drafting_agent import drafting_agent
from reviewing_agent import reviewing_agent
from summarization_agent import summarization_agent
from evaluation_agent import evaluation_agent
from kg_context_builder import KGContextBuilder
from typing import Optional
from cpc_predictor import predict_cpc
import json

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def patent_pipeline(project_text, use_kg: bool = False, root_hint: Optional[str] = None):
    print("Step 0: Parsing Input Information...")
    parsed_info = input_parser_agent(project_text)
    print("\n Parsed Structured Info:\n", parsed_info)

    # 提取结构化字段
    try:
        import json
        if isinstance(parsed_info, dict):
            parsed_json = parsed_info
        else:
            parsed_json = json.loads(parsed_info)
        title = parsed_json.get("title") or parsed_json.get("core_device_or_system", "Unnamed Invention")
        abstract = parsed_json.get("abstract", "")
        description = parsed_json.get("description", project_text)
    except Exception:
        print("⚠️ Warning: Failed to parse structured info as JSON, using raw text instead.")
        title, abstract, description = "Unnamed Project", project_text, project_text

    # === KG constraints (optional) ===
    kg_constraints = ""
    if use_kg:
        kg = KGContextBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        try:
            # 用 project_text 选 root；你也可以传 root_hint="information retrieval" 固定做 A/B 实验
            kg_constraints = kg.build_prompt_block(project_text, root_hint=root_hint, depth=2, term_limit=350)
        finally:
            kg.close()

    print("\nStep 0.5: Predicting Technical Domain (CPC)...")
    cpc_label = predict_cpc(project_text)
    print(" Predicted CPC:", cpc_label)

    print("\nStep 1: Extracting Innovation Points...")
    innovation = innovation_agent(
        title,
        abstract,
        description,
        cpc_label=cpc_label,
        kg_constraints=kg_constraints
    )
    print("\n Innovation JSON:\n", innovation)

    print("\nStep 2: Drafting Structured Patent...")
    draft = drafting_agent(
        innovation,
        cpc_label=cpc_label,
        kg_constraints=kg_constraints
    )
    print("\n Draft Preview:\n", draft[:800], "...")

    print("\nStep 3: Reviewing & Scoring...")
    reviewed = reviewing_agent(draft, kg_constraints=kg_constraints)
    print("\n Reviewed Version:\n", reviewed[:800], "...")

    print("\nStep 4: Generating Summary & Highlights...")
    summary = summarization_agent(reviewed)
    print("\n Final Summary:\n", summary)

    print("\nStep 5: Evaluating Final Patent Quality...")
    evaluation = evaluation_agent(reviewed)
    print("\n Evaluation Report:\n", evaluation)

    print("\n===== ✅ Patent Generation Completed =====")

    return {
        "parsed_info": parsed_info,
        "innovation": innovation,
        "draft": draft,
        "reviewed": reviewed,
        "summary": summary,
        "evaluation": evaluation,
        "kg_used": use_kg,
        "kg_root_hint": root_hint,
    }