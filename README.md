# Multi-Agent Patent Drafting System with CSO Knowledge Enhancement

This repository contains the implementation of a graduation thesis project on automated patent drafting for computer science inventions. The system combines a multi-agent LLM workflow, a Neo4j-based Computer Science Ontology (CSO) knowledge graph, and a local CPC/domain classifier to generate, review, summarize, and evaluate structured patent documents.

The prototype includes a FastAPI backend and a browser-based frontend. Users submit a technical invention description, and the system returns a patent-style draft, a reviewed version, a summary, and quality scores.

## Project Background

Patent drafting is a professional long-text generation task that requires technical accuracy, structural completeness, terminology consistency, and compliance with patent-writing conventions. A single end-to-end LLM may lose consistency across long sections or introduce unrelated terminology. To address this, the project decomposes patent drafting into a sequential multi-agent workflow and injects domain knowledge from CSO.

The system follows six main stages:

1. Input parsing
2. Innovation extraction
3. Patent drafting
4. Draft review and revision
5. Summary generation
6. Quality evaluation

Two additional mechanisms support the pipeline:

- CSO knowledge enhancement: CSO triples are imported into Neo4j and retrieved as terminology constraints.
- CPC/domain classification: a fine-tuned transformer classifier predicts the technical domain label and guides generation.

## Main Features

- Multi-agent patent generation pipeline
- FastAPI backend
- Static HTML frontend with Bootstrap 5 and Chart.js
- Neo4j knowledge graph retrieval based on `CSO_clean.csv`
- Local transformer-based CPC/domain classifier
- A/B experiment scripts for comparing KG-enhanced and non-KG generation
- Rule-based metric checker for patent section completeness, definition coverage, and domain drift

## Repository Structure

```text
.
|-- backend/
|   `-- app.py
|-- frontend/
|   `-- index.html
|-- models/
|   |-- __init__.py
|   |-- root_predictor.py
|   `-- root_classifier/              # optional local model directory
|-- training/
|   `-- train_root_classifier_hupd.py
|-- ab_metric_checker.py
|-- ab_runner.py
|-- compare_AB.py
|-- cpc_predictor.py
|-- CSO_clean.csv
|-- drafting_agent.py
|-- evaluation_agent.py
|-- innovation_agent.py
|-- input_parser_agent.py
|-- kg_context_builder.py
|-- pipeline.py
|-- reviewing_agent.py
|-- run_pipeline_example.py
|-- summarization_agent.py
`-- README.md
```

## File Description

| File | Description |
|---|---|
| `backend/app.py` | FastAPI backend service. |
| `frontend/index.html` | Browser frontend for submitting invention descriptions and viewing generated outputs. |
| `pipeline.py` | Main multi-agent orchestration pipeline. |
| `run_pipeline_example.py` | Local example for running the full pipeline without the frontend. |
| `input_parser_agent.py` | Converts raw invention descriptions into structured information. |
| `innovation_agent.py` | Extracts the core technical innovation points. |
| `drafting_agent.py` | Generates the structured patent draft. |
| `reviewing_agent.py` | Reviews and revises the generated draft. |
| `summarization_agent.py` | Generates the final summary and highlights. |
| `evaluation_agent.py` | Scores the final draft across quality dimensions. |
| `kg_context_builder.py` | Retrieves CSO topics from Neo4j and formats them as prompt constraints. |
| `cpc_predictor.py` | Loads the local classifier and predicts the CPC/domain label. |
| `models/root_predictor.py` | Predicts the CSO root topic used for knowledge retrieval. |
| `training/train_root_classifier_hupd.py` | Training script for the HUPD-based classifier. |
| `ab_runner.py` | Runs baseline and KG-enhanced outputs for ablation experiments. |
| `ab_metric_checker.py` | Computes rule-based evaluation metrics. |
| `compare_AB.py` | Compares baseline and KG-enhanced outputs. |
| `CSO_clean.csv` | Cleaned CSO triples for rebuilding the Neo4j graph. |

## Environment Requirements

Recommended environment:

- Python 3.10+
- Neo4j 5.x or Neo4j Desktop
- A modern browser

Install Python dependencies:

```bash
pip install fastapi uvicorn pydantic openai neo4j torch transformers datasets scikit-learn matplotlib
```

Optional virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn pydantic openai neo4j torch transformers datasets scikit-learn matplotlib
```

Optional virtual environment on macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic openai neo4j torch transformers datasets scikit-learn matplotlib
```

## Environment Variables

The project reads API and database credentials from environment variables. Do not commit real keys, passwords, or `.env` files to GitHub.

Required variables:

```text
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-api-base-url/v1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

Windows PowerShell example:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://your-api-base-url/v1"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="your_neo4j_password"
```

macOS/Linux example:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-api-base-url/v1"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_neo4j_password"
```

You can also create a local `.env` file for your own use, but it should not be committed. A safe example format is:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-api-base-url/v1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

## Model Files

The classifier model files are not always suitable for normal Git upload because `model.safetensors` can be large. If the weights are not included in the repository, download or train them and place them in the following paths.

Runtime CPC/domain classifier expected by `cpc_predictor.py`:

```text
experiments/
|-- config.json
|-- id2label.json
|-- label2id.json
|-- model.safetensors
|-- tokenizer.json
|-- tokenizer_config.json
|-- model_card.json
|-- train_metrics.json
`-- training_config.json
```

Root topic classifier expected by `models/root_predictor.py`:

```text
models/root_classifier/
|-- config.json
|-- merges.txt
|-- model.safetensors
|-- special_tokens_map.json
|-- tokenizer.json
|-- tokenizer_config.json
`-- vocab.json
```

Recommended handling for large model weights:

- Upload `model.safetensors` through Git LFS;
- or publish it in GitHub Releases;
- or host it on Hugging Face;
- or retrain the classifier with `training/train_root_classifier_hupd.py`.

Do not upload training checkpoints such as:

```text
models/root_classifier/checkpoint-100/
models/root_classifier/checkpoint-*/optimizer.pt
```

## Neo4j Knowledge Graph Setup

The system uses Neo4j to store CSO topic nodes and topic relations. The file `CSO_clean.csv` contains cleaned CSO triples:

```text
<source_topic_uri>,<relation_uri>,<target_topic_uri>
```

Example:

```text
<https://cso.kmi.open.ac.uk/topics/computer_science>,<http://cso.kmi.open.ac.uk/schema/cso#superTopicOf>,<https://cso.kmi.open.ac.uk/topics/artificial_intelligence>
```

The code expects:

- node label: `CSOTopic`
- node property: `label`
- relation type: `SUPER_TOPIC_OF`
- optional synonym relation type: `RELATED_EQUIVALENT`

### Option 1: Neo4j Desktop

1. Install Neo4j Desktop.
2. Create and start a local DBMS.
3. Set the password for user `neo4j`.
4. Copy `CSO_clean.csv` into the Neo4j import directory.
5. Open Neo4j Browser.
6. Run the Cypher import commands below.

### Option 2: Docker

Windows PowerShell:

```powershell
docker run --name neo4j-cso `
  -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/your_password `
  -v ${PWD}/neo4j-data:/data `
  -v ${PWD}:/import `
  neo4j:5
```

macOS/Linux:

```bash
docker run --name neo4j-cso \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -v "$PWD/neo4j-data:/data" \
  -v "$PWD:/import" \
  neo4j:5
```

Open Neo4j Browser:

```text
http://localhost:7474
```

### Import CSO_clean.csv

Create a uniqueness constraint:

```cypher
CREATE CONSTRAINT cso_topic_label IF NOT EXISTS
FOR (t:CSOTopic)
REQUIRE t.label IS UNIQUE;
```

Import topics and relations:

```cypher
LOAD CSV FROM 'file:///CSO_clean.csv' AS row
WITH
  row,
  toLower(replace(split(replace(row[0], '>', ''), '/topics/')[1], '_', ' ')) AS source_label,
  row[1] AS relation_uri,
  toLower(replace(split(replace(row[2], '>', ''), '/topics/')[1], '_', ' ')) AS target_label
MERGE (source:CSOTopic {label: source_label})
MERGE (target:CSOTopic {label: target_label})
FOREACH (_ IN CASE WHEN relation_uri CONTAINS 'superTopicOf' THEN [1] ELSE [] END |
  MERGE (source)-[:SUPER_TOPIC_OF]->(target)
)
FOREACH (_ IN CASE WHEN relation_uri CONTAINS 'relatedEquivalent' THEN [1] ELSE [] END |
  MERGE (source)-[:RELATED_EQUIVALENT]->(target)
);
```

Check the imported graph:

```cypher
MATCH (t:CSOTopic) RETURN count(t) AS topic_count;
```

```cypher
MATCH ()-[r:SUPER_TOPIC_OF]->() RETURN count(r) AS super_topic_edges;
```

## Backend Deployment

Start the backend from the project root:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Test the backend:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "backend is running"
}
```

Main endpoint:

```text
POST /api/patent/run
```

Example request body:

```json
{
  "text": "We propose an intelligent retrieval and ranking system for e-commerce search...",
  "root_hint": "information retrieval"
}
```

The response includes:

- parsed invention information
- innovation analysis
- patent draft
- reviewed draft
- summary
- evaluation result
- KG usage information

## Frontend Deployment

The frontend is a static HTML page.

Option 1: open directly:

```text
frontend/index.html
```

Option 2: serve locally:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

Make sure the backend is running at:

```text
http://127.0.0.1:8000
```

## Running Without the Frontend

Run the example script:

```bash
python run_pipeline_example.py
```

Or call the pipeline directly:

```python
from pipeline import patent_pipeline

result = patent_pipeline(
    project_text="Your invention description here...",
    use_kg=True,
    root_hint="information retrieval"
)

print(result["reviewed"])
```

## Running Ablation Experiments

Run baseline and KG-enhanced generation:

```bash
python ab_runner.py
```

This produces:

```text
run_baseline.json
run_with_kg.json
```

Compare the outputs:

```bash
python compare_AB.py run_baseline.json run_with_kg.json
```

Run rule-based metric checks:

```bash
python ab_metric_checker.py
```

These scripts are used to support the thesis evaluation of:

- CSO knowledge enhancement;
- multi-agent architecture;
- terminology consistency;
- definition coverage;
- patent section completeness;
- domain drift reduction.

## Training the Classifier

The classifier training script is:

```text
training/train_root_classifier_hupd.py
```

It supports the following environment variables:

```text
ROOT_CLS_BASE_MODEL
ROOT_CLS_OUT_DIR
LOCAL_JSONL
TRAIN_SAMPLES
EVAL_SAMPLES
MAX_LEN
SEED
```

Example:

```powershell
$env:ROOT_CLS_BASE_MODEL="distilroberta-base"
$env:ROOT_CLS_OUT_DIR="models/root_classifier"
$env:LOCAL_JSONL="D:\patent1.jsonl"
python training/train_root_classifier_hupd.py
```

The output directory should contain Hugging Face-compatible files such as `config.json`, tokenizer files, and `model.safetensors`.

## Recommended .gitignore

The following files should not be committed:

```text
.env
__pycache__/
*.pyc
.idea/
.venv/
neo4j-data/
models/root_classifier/checkpoint-*/
models/root_classifier/checkpoint-*/optimizer.pt
*.pt
```

Handle large model weights separately if necessary:

```text
experiments/model.safetensors
models/root_classifier/model.safetensors
```

## Notes

- `CSO_clean.csv` is enough to rebuild the Neo4j knowledge graph. The original raw CSO file is not required for normal use.
- Do not upload the local Neo4j database directory.
- If KG enhancement is disabled, the pipeline can still run as a multi-agent patent drafting system, but CSO terminology constraints will not be injected.
- If classifier model files are missing, `cpc_predictor.py` and `models/root_predictor.py` cannot load the local models. Download, provide, or retrain the model files before running the complete system.
- If you publish this repository, keep real API keys and passwords outside Git history.

## Thesis Context

This project corresponds to a thesis on knowledge-enhanced multi-agent patent drafting for computer science inventions. The implementation supports:

- Chapter 2: system architecture, CSO knowledge enhancement, multi-agent framework, and CPC/domain classifier;
- Chapter 3: backend, frontend, prompt-based agents, data processing, and end-to-end pipeline implementation;
- Chapter 4: classifier evaluation and ablation study comparing KG-enhanced and non-KG settings.

The main conclusion is that CSO knowledge enhancement improves terminology consistency and definition coverage, while the multi-agent architecture improves patent-style structure and section completeness compared with a single-agent baseline.
