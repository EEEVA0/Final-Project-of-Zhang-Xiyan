from input_parser_agent import input_parser_agent
from innovation_agent import innovation_agent
from drafting_agent import drafting_agent
from reviewing_agent import reviewing_agent
from summarization_agent import summarization_agent
from evaluation_agent import evaluation_agent
from kg_context_builder import KGContextBuilder
from cpc_predictor import predict_cpc

from typing import Optional, Any, Dict
import json


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def safe_json_loads(data: Any) -> Dict[str, Any]:
    """
    Convert agent output into a dictionary when possible.
    If parsing fails, return an empty dictionary.
    """
    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        try:
            return json.loads(data)
        except Exception:
            return {}

    return {}


def shorten_text(text: Any, max_len: int = 1500) -> str:
    """
    Shorten long outputs for thesis examples.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, indent=2)

    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."


def save_full_pipeline_result(result: Dict[str, Any], filename: str = "pipeline_full_output.json") -> None:
    """
    Save complete pipeline outputs, including draft and reviewed draft.
    This file may be long.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nFull pipeline result saved to: {filename}")


def save_chapter3_example(result: Dict[str, Any], filename: str = "chapter3_example_output.json") -> None:
    """
    Save a shortened version suitable for Chapter 3 examples.
    This is better for sending to ChatGPT or inserting into the thesis.
    """
    example = {
        "input_text": shorten_text(result.get("input_text", ""), 1200),
        "parsed_info": result.get("parsed_info"),
        "cpc_label": result.get("cpc_label"),
        "context_T_excerpt": shorten_text(result.get("kg_constraints", ""), 1200),
        "innovation": result.get("innovation"),
        "draft_excerpt": shorten_text(result.get("draft", ""), 1800),
        "reviewed_excerpt": shorten_text(result.get("reviewed", ""), 1800),
        "summary": result.get("summary"),
        "evaluation": result.get("evaluation"),
        "kg_used": result.get("kg_used"),
        "kg_root_hint": result.get("kg_root_hint"),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(example, f, ensure_ascii=False, indent=2)

    print(f"Chapter 3 example saved to: {filename}")


def patent_pipeline(
    project_text: str,
    use_kg: bool = False,
    root_hint: Optional[str] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Full patent generation pipeline.

    Input:
        project_text:
            Raw technical description submitted by the user.
        use_kg:
            Whether to enable CSO-based knowledge enhancement.
        root_hint:
            Optional fixed CSO root topic for controlled experiments.
        save_outputs:
            Whether to save JSON files locally.

    Output:
        A dictionary containing all intermediate and final outputs.
    """

    print("Step 0: Parsing Input Information...")
    parsed_info = input_parser_agent(project_text)
    print("\nParsed Structured Info:\n", parsed_info)

    parsed_json = safe_json_loads(parsed_info)

    if parsed_json:
        title = parsed_json.get("title") or parsed_json.get("core_device_or_system", "Unnamed Invention")
        abstract = parsed_json.get("abstract", "")
        description = parsed_json.get("description", project_text)
    else:
        print("Warning: Failed to parse structured info as JSON, using raw text instead.")
        title = "Unnamed Project"
        abstract = project_text
        description = project_text

    print("\nStep 0.5: Building KG Context T...")
    kg_constraints = ""

    if use_kg:
        kg = KGContextBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        try:
            kg_constraints = kg.build_prompt_block(
                project_text,
                root_hint=root_hint,
                depth=2,
                term_limit=350,
            )
            print("\nContext T Preview:\n", shorten_text(kg_constraints, 800))
        finally:
            kg.close()
    else:
        print("KG enhancement disabled.")

    print("\nStep 0.6: Predicting Technical Domain with CPC Classifier...")
    cpc_label = predict_cpc(project_text)
    print("Predicted CPC:", cpc_label)

    print("\nStep 1: Extracting Innovation Points...")
    innovation = innovation_agent(
        title,
        abstract,
        description,
        cpc_label=cpc_label,
        kg_constraints=kg_constraints,
    )
    print("\nInnovation JSON:\n", innovation)

    print("\nStep 2: Drafting Structured Patent...")
    draft = drafting_agent(
        innovation,
        cpc_label=cpc_label,
        kg_constraints=kg_constraints,
    )
    print("\nDraft Preview:\n", shorten_text(draft, 800))

    print("\nStep 3: Reviewing and Revising Draft...")
    reviewed = reviewing_agent(
        draft,
        kg_constraints=kg_constraints,
    )
    print("\nReviewed Version Preview:\n", shorten_text(reviewed, 800))

    print("\nStep 4: Generating Summary and Highlights...")
    summary = summarization_agent(reviewed)
    print("\nFinal Summary:\n", summary)

    print("\nStep 5: Evaluating Final Patent Quality...")
    evaluation = evaluation_agent(reviewed)
    print("\nEvaluation Report:\n", evaluation)

    print("\n===== Patent Generation Completed =====")
    result = {
        "input_text": project_text,
        "parsed_info": parsed_info,
        "title": title,
        "abstract": abstract,
        "description": description,
        "cpc_label": cpc_label,
        "kg_constraints": kg_constraints,
        "innovation": innovation,
        "draft": draft,
        "reviewed": reviewed,
        "summary": summary,
        "evaluation": evaluation,
        "kg_used": use_kg,
        "kg_root_hint": root_hint,
    }

    if save_outputs:
        save_full_pipeline_result(result, "pipeline_full_output4.json")
        save_chapter3_example(result, "chapter3_example_output40.json")

    return result


if __name__ == "__main__":
    project_text = """
    We have developed a low-resource cross-lingual sentence embedding system for multilingual information retrieval and semantic matching. The system is designed for scenarios where a high-resource language has sufficient training data, while the target low-resource language has limited parallel data and weak lexical coverage. Existing sentence-level alignment methods often fail when two sentences share a general topic but differ in key phrases or domain-specific expressions.
    
    The proposed system introduces a phrase-aware prompt tuning framework. A shared multilingual text encoder is used to encode both high-resource and low-resource sentences. Most encoder parameters are frozen, while a small set of learnable soft prompt tokens guides the encoder toward cross-lingual semantic alignment. The system first extracts word-level alignment signals from a teacher model, then constructs phrase candidates using n-gram mining, dependency-based extraction, and contextual similarity filtering.
    
    After phrase construction, the system computes phrase-level alignment scores between candidate phrases across languages. These scores are combined with a sentence-level contrastive learning objective, so that semantically equivalent sentence pairs are close in the embedding space while mismatched pairs are separated. A dynamic hard negative sampling module selects sentence pairs that are topically similar but semantically different at the phrase level, improving the model’s ability to distinguish subtle cross-lingual meaning differences.
    
    At inference time, the system encodes multilingual sentences into a shared vector space. The generated embeddings can be used for multilingual search, cross-lingual retrieval, semantic clustering, and low-resource question matching. In testing, the phrase-aware prompt tuning method improved retrieval accuracy and reduced false matches caused by phrase-level semantic drift.    """

    result = patent_pipeline(
        project_text=project_text,
        use_kg=True,
        root_hint="information retrieval",
        save_outputs=True,
    )