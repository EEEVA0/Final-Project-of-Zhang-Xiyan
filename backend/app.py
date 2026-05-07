import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from pipeline import patent_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Original request model (unchanged) ──
class RunReq(BaseModel):
    text: str
    root_hint: str | None = "information retrieval"


# ── NEW: Revision request model ──
class ReviseReq(BaseModel):
    original_input: str
    current_draft: str
    instructions: str


@app.get("/")
def root():
    return {"message": "backend is running"}


# ── Original endpoint (unchanged) ──
@app.post("/api/patent/run")
def run_patent(req: RunReq):
    out = patent_pipeline(
        req.text,
        use_kg=True,
        root_hint=req.root_hint
    )
    return out


# ── NEW: Revision endpoint ──
# Calls the Drafting Agent again with the current draft + user instructions appended.
# Your pipeline.py needs to expose a `revision_pipeline` function (see below),
# OR you can inline the agent call here if preferred.
@app.post("/api/patent/revise")
def revise_patent(req: ReviseReq):
    """
    Re-invokes the Drafting Agent with:
      - the original inventive points (re-parsed from original_input)
      - the current draft as context
      - the user's revision instructions

    Returns { revised_draft: str, scores: dict } in the same shape as /run.

    ── What you need in pipeline.py ──
    Add a function like:

        def revision_pipeline(original_input: str, current_draft: str, instructions: str) -> dict:
            import anthropic
            client = anthropic.Anthropic()

            prompt = f\"\"\"You are a patent drafting expert performing a targeted revision.

    ORIGINAL TECHNICAL DESCRIPTION:
    {original_input}

    CURRENT DRAFT:
    {current_draft}

    REVISION INSTRUCTIONS FROM USER:
    {instructions}

    Please rewrite the draft incorporating the revision instructions.
    Keep all sections intact unless the instruction explicitly targets a section.
    Output the full revised draft only.\"\"\")

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            revised = message.content[0].text

            # Optionally re-run the Evaluation Agent on the new draft
            scores = evaluate_draft(revised)   # reuse your existing eval function
            return {"revised_draft": revised, "scores": scores}
    """
    try:
        from pipeline import revision_pipeline
        result = revision_pipeline(
            original_input=req.original_input,
            current_draft=req.current_draft,
            instructions=req.instructions
        )
        return result
    except ImportError:
        # Fallback: revision_pipeline not yet implemented in pipeline.py
        # Return the current draft unchanged so the frontend doesn't break
        return {
            "revised_draft": req.current_draft,
            "scores": None,
            "warning": "revision_pipeline not found in pipeline.py – please implement it."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)