# kg_context_builder.py
from __future__ import annotations
from typing import List, Dict, Optional
from models.root_predictor import predict_root
from neo4j import GraphDatabase

DEFAULT_ROOT_CANDIDATES = [
    # 你做专利撰写助手很常见的计算机根领域
    "information retrieval",
    "machine learning",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "distributed systems",
    "computer networks",
    "recommendation systems",
    "data mining",
    "computer science",
]

class KGContextBuilder:
    """
    从 Neo4j(CSO) 里取一个 root 子树的术语清单，作为 LLM 写作约束。
    - root: CSO 中的一个节点 label
    - allowed_terms: root 下 0..depth 层的术语（用于限定领域、减少跑偏）
    - synonyms: 相关等价术语（用于术语一致性）
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _node_exists(self, label: str) -> bool:
        q = "MATCH (t:CSOTopic {label:$label}) RETURN count(t) AS c"
        with self.driver.session() as s:
            return (s.run(q, label=label).single()["c"] or 0) > 0

    def choose_root(self, text: str, root_hint: Optional[str] = None) -> str:
        """
        优先 root_hint；否则用训练好的 root classifier；再用简单包含匹配；最后回退 computer science。
        """
        text_l = (text or "").lower()

        # 1) 用户/调用方显式指定 root_hint
        if root_hint and self._node_exists(root_hint):
            return root_hint

        # 2) ✅ 模型预测 root（你刚训练出来的）
        try:
            pred = predict_root(text)
            if pred and self._node_exists(pred):
                return pred
        except Exception:
            pass

        # 3) 原来的 substring 兜底（仍保留）
        for r in DEFAULT_ROOT_CANDIDATES:
            if r in text_l and self._node_exists(r):
                return r

        # 4) 最终兜底
        
        if self._node_exists("computer science"):
            return "computer science"

        for r in DEFAULT_ROOT_CANDIDATES:
            if self._node_exists(r):
                return r
        return "computer science"


    def get_allowed_terms(self, root: str, depth: int = 2, limit: int = 350) -> List[str]:
        depth = int(depth)
        if depth < 0:
            depth = 0
        if depth > 6:  # 防止一次拉太大
            depth = 6

        q = f"""
        MATCH (root:CSOTopic {{label:$root}})-[:SUPER_TOPIC_OF*0..{depth}]->(t)
        RETURN collect(DISTINCT t.label) AS terms
        """
        with self.driver.session() as s:
            rec = s.run(q, root=root).single()
            terms = (rec["terms"] if rec else []) or []
        terms = sorted({t for t in terms if isinstance(t, str) and t.strip()})
        return terms[:limit]

    def get_synonyms(self, allowed_terms: List[str], limit_pairs: int = 120, each_limit: int = 4) -> Dict[str, List[str]]:
        """
        从 RELATED_EQUIVALENT 关系里取同义术语映射（如果你导入了该关系）。
        如果没导入 RELATED_EQUIVALENT，返回空 dict，不影响流程。
        """
        q = """
        MATCH (a:CSOTopic)-[:RELATED_EQUIVALENT]->(b:CSOTopic)
        WHERE a.label IN $terms
        RETURN a.label AS a, collect(DISTINCT b.label) AS bs
        LIMIT $limit_pairs
        """
        syn: Dict[str, List[str]] = {}
        with self.driver.session() as s:
            rows = list(s.run(q, terms=allowed_terms, limit_pairs=limit_pairs))
        for r in rows:
            a = r["a"]
            bs = [x for x in (r["bs"] or []) if isinstance(x, str) and x.strip()]
            if a and bs:
                syn[a] = bs[:each_limit]
        return syn

    def build_prompt_block(
            self,
            text: str,
            root_hint: Optional[str] = None,
            depth: int = 2,
            term_limit: int = 350,
            include_synonyms: bool = True,
    ) -> str:
        root = self.choose_root(text, root_hint=root_hint)
        allowed = self.get_allowed_terms(root, depth=depth, limit=term_limit)
        syn = self.get_synonyms(allowed) if include_synonyms else {}

        # 术语表不要太长：用“允许术语 + 必须优先使用”表达即可
        allowed_preview = ", ".join(allowed[:180])  # prompt 不要爆

        lines = [
            "=== Knowledge Graph Constraints (CSO) ===",
            f"Root domain: {root}",
            "Use the root domain as the main technical scope and prefer the vocabulary below when it is relevant.",
            "Do not introduce unrelated technical domains unless they are explicitly required by the invention.",
            f"Preferred/Allowed terms (partial list): {allowed_preview}",
            "Terminology consistency rule: use ONE preferred term per concept; avoid mixing synonyms.",
        ]

        # 同义词映射只放部分，避免 prompt 过长
        if syn:
            syn_items = list(syn.items())[:40]
            syn_text = "; ".join([f"{k} ~ {', '.join(v)}" for k, v in syn_items])
            lines.append(f"Synonym hints (partial): {syn_text}")

        # Writing constraints: keep the draft patent-style, but do not force unrelated mechanisms.
        lines.extend([
            "",
            "=== Writing Constraints (STRICT, must follow) ===",
            "Your output must remain a patent-style document and must satisfy these constraints:",
            "",
            "1) Definitions coverage:",
            '- If you include a Definitions section, each definition should use the form: In this invention, the term "X" refers to ...',
            "- Every technical term or module appearing in the Claims MUST be defined either in the Definitions section or in the Detailed Description.",
            "- Prefer one canonical term per concept using the CSO vocabulary where appropriate.",
            "- Do not mix synonyms for the same concept across sections unless the relationship between them is explicitly defined.",
            "",
            "2) Input-faithfulness rule:",
            "- Do NOT introduce a technical mechanism merely because it appears in this constraint block.",
            "- Only elaborate mechanisms that are explicitly present in the user input, the parsed invention information, or the Innovation Agent output.",
            "- If a mechanism is not mentioned or clearly implied by the invention, do not add it to the draft, the claims, the summary, or the review.",
            "- The CSO vocabulary should be used as a terminology guide, not as a mandatory list of concepts to include.",
            "",
            "3) Mechanism specificity rule:",
            "- If debiasing, counterfactual learning, propensity score, position bias, exposure bias, IPS, SNIPS, or Doubly Robust learning is explicitly mentioned, then specify:",
            "  (a) how the relevant bias or propensity is estimated; and",
            "  (b) one debiased objective, such as IPS, SNIPS, or Doubly Robust.",
            "  If these terms are not present in the invention, do not introduce a debiasing mechanism.",
            "- If exploration, multi-armed bandit, UCB, Thompson sampling, uncertainty-based selection, or exploration budget is explicitly mentioned, then specify:",
            "  (a) the chosen exploration algorithm;",
            "  (b) what parameter is updated online, such as a confidence bound or posterior; and",
            "  (c) the exploration trigger or budget.",
            "  If these terms are not present in the invention, do not introduce an exploration mechanism.",
            "- If caching, cache replacement, cache admission, TTL, LRU, LFU, latency, p95, p99, QPS, or delay-aware scheduling is explicitly mentioned, then specify:",
            "  (a) the cache policy, cache admission rule, or cache replacement rule; and",
            "  (b) the latency-related or workload-related trigger.",
            "  If these terms are not present in the invention, do not introduce a caching or latency mechanism.",
            "",
            "4) Domain alignment:",
            "- Keep the draft aligned with the invention's stated technical field and the selected CSO root domain.",
            "- Do not reframe the invention as a different type of system.",
            "- For example, do not turn a cross-lingual sentence embedding system into a generic recommendation system, and do not turn a cache replacement system into a ranking or advertising system.",
            "",
            "5) Forbidden drift domains:",
            "- Do NOT introduce biomedical, clinical, financial trading, or unrelated industrial domains unless explicitly required by the user input.",
            "- Forbidden unrelated examples include: gene, patient, clinical, tumor, cancer, stock, bond, portfolio, trading.",
            "",
            "=== End of Writing Constraints ===",
            ""
        ])

        lines.append("=== End of KG Constraints ===")
        return "\n".join(lines)