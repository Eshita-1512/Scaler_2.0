import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException

# ── Resolve imports relative to package root ─────────────────────────────────
# Ensure the package root (parent of drug_interaction_env/) is on sys.path
_pkg_root = Path(__file__).resolve().parent.parent          # drug_interaction_env/
_repo_root = _pkg_root.parent                               # repo root
for p in (_pkg_root, _repo_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import (
    DrugInteractionAction,
    ResetRequest,
    StepResponse,
)

from server.drug_interaction_environment import DrugInteractionEnvironment

env = DrugInteractionEnvironment()
app = FastAPI(
    title="Drug Interaction Environment",
    description="OpenEnv-compliant step-reset environment for pharmaceutical drug interaction checking.",
    version="0.1.0",
)

@app.post("/reset")
def reset(request: ResetRequest):
    obs = env.reset(request.task_level)
    state = env.state
    return {"observation": obs, "reward": 0.0, "done": False, "state": state}

@app.post("/step")
def step(action: DrugInteractionAction):
    action_dict = action.model_dump()
    obs, reward, done, state = env.step(action_dict)
    return StepResponse(observation=obs, reward=reward, done=done, state=state)

@app.get("/state")
def get_state():
    return env.state

@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
