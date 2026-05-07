# ab_runner.py
import json
from pipeline import patent_pipeline

TEST_INPUT = """
We have developed an online ranking optimization system for the mixed display of e-commerce advertisements and search results. This system adopts a counterfactual ranking learning framework, models the exposure probability based on historical click logs, and weights and corrects the ranking loss function based on propensity score, thereby eliminating training biases caused by position bias and exposure bias. 
During the training phase, we estimate the display probability of each exposure log and construct an inverse probability weighted loss function (IPS loss) to optimize the parameters of the ranking model. This method achieves unbiased ranking learning without changing the original log collection mechanism. 
To alleviate the cold start problem of new products, we have built a product knowledge graph embedding module, which encodes the category, brand, attributes and upstream-downstream association relationships of products as graph structure vectors and uses them to initialize the product embedding parameters in the ranking model. Through this structured initialization, new products can still obtain reasonable ranking positions even without historical click data. 
In addition, we have implemented a dynamic negative sample generation mechanism. In each training batch, high-similarity but un-clicked items are selected as hard negative samples based on the distance in the embedding space, which is used to enhance the model's discrimination ability. 
During the online phase, the system introduces a lightweight multi-armed bandit exploration module to randomly expose a limited proportion of highly uncertain products while maintaining the overall click-through rate stable. The exploration strategy is based on the Upper Confidence Bound (UCB) algorithm and updates the confidence interval in real time based on click feedback. 
To meet the real-time response requirements, the system adopts a two-level cache structure: the first-level cache stores the sorting results of frequently queried items, while the second-level cache stores the embedding results of highly reliable products. Additionally, a delay-aware scheduling strategy is employed to dynamically adjust the cache update frequency. 
This system has been deployed on an actual e-commerce advertising platform. The results of online A/B testing show that: 
The click-through rate (CTR) of cold-start products has increased by 12.4%. 
The sensitivity of the overall sorting model to positional deviations has decreased by 35%. 
The average response delay is consistently within 43 milliseconds. 
The system is mainly applied to the mixed sorting scenarios of advertisements and search results on large-scale e-commerce platforms.
""".strip()

def save(out, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    print("\n=== Baseline (no KG) ===")
    out_base = patent_pipeline(TEST_INPUT, use_kg=False)
    save(out_base, "run_baseline.json")

    print("\n=== With KG (CSO constraints) ===")
    # 为了让 A/B 更明显，建议先固定 root_hint="information retrieval"
    out_kg = patent_pipeline(TEST_INPUT, use_kg=True, root_hint="information retrieval")
    save(out_kg, "run_with_kg.json")

    print("\nSaved: run_baseline.json, run_with_kg.json")
    print("\nCompare these fields:")
    print("- innovation.keywords 是否更集中在 IR/Ranking/Embedding")
    print("- Claims 是否更少出现未定义模块")
    print("- Technical Field / Background 是否更少跑偏到无关领域")

if __name__ == "__main__":
    main()