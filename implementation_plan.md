# Implementation Plan — Pharmaceutical Drug Interaction Checker
# 4-Hour Split Across 3 People

---

## Split Philosophy

Each person owns one vertical layer. No person blocks another.
The only coordination point is the **interface contract** defined in Hour 0 (15 min together).

```
Person A — Core Logic (Pure Python, no web framework)
    drug_database.py, patients.py, validate(), calculate_reward(), environment class

Person B — Server + OpenEnv Spec (FastAPI + Docker + OpenEnv)
    models.py, app.py, openenv.yaml, Dockerfile, HF deployment

Person C — Inference + Grader (Client side, pure Python)
    inference.py, grader.py, episode scoring, stdout log format
```

Why these are non-correlated:
- A writes pure functions, no imports from B or C
- B wraps A's functions — imports only after Hour 3 handoff, uses stubs before that
- C works against a mock/stub server until B's server is live in Hour 3

---

## Hour 0 — Contract Definition (All 3 Together, 15 min)

Before splitting, agree on these interfaces so everyone can work independently:

```python
# Agreed action shape
action = {
    "action_type": "flag_interaction" | "DONE",
    "drug_a": str | None,
    "drug_b": str | None,
    "severity": "mild" | "moderate" | "severe" | None,
    "suggested_action": "monitor" | "reduce_dose" | "replace_drug" | None
}

# Agreed step() return shape
{
    "observation": { ... },
    "reward": float,
    "done": bool,
    "state": { ... }
}

# Agreed API endpoints
POST /reset  → { "task_level": "easy" | "medium" | "hard" }
POST /step   → action JSON
GET  /state  → current state JSON
```

---

## 4-Hour Timeline

```
TIME        PERSON A                    PERSON B                    PERSON C
────────────────────────────────────────────────────────────────────────────
0:00-0:15   [ALL THREE] Hour 0 — agree on interface contract
────────────────────────────────────────────────────────────────────────────
0:15-1:00   drug_database.py            models.py                   mock_env.py
            patients.py                 (all Pydantic models)        (local stub server)
────────────────────────────────────────────────────────────────────────────
1:00-2:00   validate()                  app.py stubs                inference.py
            calculate_reward()          (/reset /step /state        (LLM loop +
                                        with placeholder logic)      [START][STEP][END])
────────────────────────────────────────────────────────────────────────────
2:00-3:00   DrugInteraction             Wire A's environment        grader.py
            Environment class           into app.py                 (per task scoring +
            (reset/step/state/          openenv.yaml                normalization)
            termination)               Dockerfile
────────────────────────────────────────────────────────────────────────────
3:00-4:00   Unit tests for             Docker build test            End-to-end test
            validate() and             HF Space deploy              against live server
            calculate_reward()         Validator script run         Log format verify
────────────────────────────────────────────────────────────────────────────

HANDOFFS:
  A → B  Hour 1: B can import drug_database.py to build models
  A → B  Hour 3: B wires DrugInteractionEnvironment into app.py
  B → C  Hour 2: C switches from mock to B's stub server
  B → C  Hour 4: C runs full end-to-end against live Docker server
```

---

---

# PERSON A — Prompt for Antigravity

```
You are implementing the CORE LOGIC layer of a Pharmaceutical Drug Interaction Checker
built on the OpenEnv framework. You write pure Python only — no FastAPI, no web server.
Your code will be imported by Person B (server layer).

---

## Your Files to Build

drug_interaction_env/server/drug_database.py
drug_interaction_env/server/patients.py
drug_interaction_env/server/drug_interaction_environment.py

---

## File 1: drug_database.py

Build a Python dict called DRUG_INTERACTIONS with 30-40 real, publicly documented
drug interaction pairs. Each key is a sorted tuple of two lowercase drug names.

Structure:
    DRUG_INTERACTIONS = {
        ("aspirin", "warfarin"): {
            "severity": "severe",
            "action": "replace_drug",
            "explanation": "Increased bleeding risk"
        },
        ...
    }

Rules:
- Keys MUST be tuple(sorted([drug_a.lower(), drug_b.lower()])) — always sorted alphabetically
- severity must be exactly one of: "mild" | "moderate" | "severe"
- action must be exactly one of: "monitor" | "reduce_dose" | "replace_drug"
- Include at least: 10 severe, 15 moderate, 10 mild interactions
- Use only real, well-known drug pairs (warfarin+aspirin, simvastatin+amiodarone, ssri+tramadol etc.)

Also add a lookup helper:
    def lookup_pair(drug_a: str, drug_b: str) -> dict | None:
        key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        return DRUG_INTERACTIONS.get(key, None)

---

## File 2: patients.py

Build a dict called PATIENTS with exactly 3 patient scenarios.

Structure:
    PATIENTS = {
        "easy": {
            "patient_id": "P001",
            "age": int,
            "conditions": [str, ...],
            "medications": [str, ...]    # 6 drugs, exactly 1 interaction in DRUG_INTERACTIONS
        },
        "medium": {
            "patient_id": "P002",
            ...                          # 10 drugs, exactly 3 interactions in DRUG_INTERACTIONS
                                         # mix of 1 severe, 1 moderate, 1 mild
        },
        "hard": {
            "patient_id": "P003",
            ...                          # 15 drugs, exactly 5 interactions in DRUG_INTERACTIONS
                                         # mix across severity levels
        }
    }

Rules:
- Ground truth interactions are NOT stored in PATIENTS — they are derived at runtime
  by scanning medications against DRUG_INTERACTIONS
- Verify your patient scenarios actually produce the correct number of interactions
  by running: [lookup_pair(a,b) for a,b in combinations(medications, 2) if lookup_pair(a,b)]
- All drug names in medications must match keys in DRUG_INTERACTIONS exactly

---

## File 3: drug_interaction_environment.py

Implement the class DrugInteractionEnvironment with these methods:

### __init__:
    self.task_level = None
    self.patient = None
    self.ground_truth_keys = set()
    self.attempted_keys = set()
    self.identified_pairs = set()
    self.perfectly_completed_pairs = set()
    self.predictions = {}
    self.step_count = 0
    self.max_steps = 0
    self.episode_reward = 0.0
    self.done = False

### reset(task_level: str) -> dict:
    Load patient from PATIENTS[task_level]
    Compute ground_truth_keys by scanning all combinations of medications against DRUG_INTERACTIONS
    Set max_steps = len(ground_truth_keys) * 3
    Reset all tracking sets to empty
    Return initial observation dict

### validate(action: dict) -> True | False | None:
    Returns one of: True | False | None

    Steps (in order):
    1. Normalize key: key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
    2. If drug_a not in patient medications OR drug_b not in patient medications:
           return False   [invalid drug name, not in patient list]
    3. If key in self.attempted_keys:
           return None    [duplicate]
    4. self.attempted_keys.add(key)
    5. If key not in DRUG_INTERACTIONS:
           return False   [phantom pair]
    6. return True

### calculate_reward(key: tuple, severity: str, action: str) -> float:
    gt = DRUG_INTERACTIONS[key]

    base_score = 0.4

    if severity == gt["severity"]:
        severity_score = 0.2
    elif gt["severity"] == "severe":
        severity_score = -0.2
    elif gt["severity"] == "moderate":
        severity_score = -0.1
    else:  # mild
        severity_score = -0.05

    if action == gt["action"]:
        action_score = 0.2
    else:
        action_score = -0.1

    step_reward = base_score + severity_score + action_score

    # Update tracking
    self.identified_pairs.add(key)
    self.predictions[key] = {
        "predicted_severity": severity,
        "predicted_action": action,
        "ground_truth_severity": gt["severity"],
        "ground_truth_action": gt["action"],
        "reward_received": step_reward,
        "perfectly_completed": (severity == gt["severity"] and action == gt["action"])
    }
    if severity == gt["severity"] and action == gt["action"]:
        self.perfectly_completed_pairs.add(key)

    return step_reward

### step(action: dict) -> tuple[dict, float, bool, dict]:
    if self.done:
        raise RuntimeError("Episode already done. Call reset() first.")

    if action["action_type"] == "DONE":
        reward = self._apply_termination_penalty()
        self.done = True
        return self._get_observation(), reward, True, self._get_state()

    # Run validate
    key = tuple(sorted([action["drug_a"].lower(), action["drug_b"].lower()]))
    result = self.validate(action)

    if result is None:
        step_reward = -0.05    # duplicate
    elif result is False:
        step_reward = -0.3     # phantom pair or invalid drug
    else:
        step_reward = self.calculate_reward(key, action["severity"], action["suggested_action"])

    self.episode_reward += step_reward
    self.step_count += 1

    # Check perfect completion
    if self.perfectly_completed_pairs == self.ground_truth_keys:
        self.done = True
        return self._get_observation(), step_reward, True, self._get_state()

    # Check step budget
    if self.step_count >= self.max_steps:
        penalty = self._apply_termination_penalty()
        self.done = True
        return self._get_observation(), penalty, True, self._get_state()

    return self._get_observation(), step_reward, False, self._get_state()

### _apply_termination_penalty(self) -> float:
    unidentified = self.ground_truth_keys - self.identified_pairs
    penalty = 0.0
    for key in unidentified:
        gt_severity = DRUG_INTERACTIONS[key]["severity"]
        if gt_severity == "severe":
            penalty -= 0.4
        elif gt_severity == "moderate":
            penalty -= 0.3
        else:
            penalty -= 0.2
    self.episode_reward += penalty
    return penalty

### _get_observation(self) -> dict:
    Return dict with:
        patient_id, age, conditions, medications,
        flags_raised_so_far (list of prediction entries),
        steps_remaining (max_steps - step_count)

### _get_state(self) -> dict:
    Return dict with:
        patient_id, step_count, task_level,
        attempted_keys (list), identified_pairs (list),
        perfectly_completed_pairs (list), predictions (dict), done

### state(self) -> dict:
    return self._get_state()

### get_episode_score(self) -> float:
    max_possible = len(self.ground_truth_keys) * 0.8
    if max_possible == 0:
        return 1.0
    return max(0.0, min(1.0, self.episode_reward / max_possible))

---

## Verification

After writing, verify with a quick script:
    from drug_database import DRUG_INTERACTIONS, lookup_pair
    from patients import PATIENTS
    from itertools import combinations

    for level, patient in PATIENTS.items():
        meds = patient["medications"]
        interactions = [(a,b) for a,b in combinations(meds,2) if lookup_pair(a,b)]
        print(f"{level}: {len(interactions)} interactions found")
        # Should print: easy: 1, medium: 3, hard: 5

Expected counts: easy=1, medium=3, hard=5
```

---

---

# PERSON B — Prompt for Antigravity

```
You are implementing the SERVER + OPENENV SPEC layer of a Pharmaceutical Drug Interaction
Checker. You build the FastAPI server, Pydantic models, OpenEnv manifest, and Docker setup.
Person A is building the core environment logic in parallel — you will import it in Hour 3.
Until then, use stubs.

---

## Your Files to Build

drug_interaction_env/models.py
drug_interaction_env/server/app.py
drug_interaction_env/openenv.yaml
drug_interaction_env/server/Dockerfile
drug_interaction_env/server/requirements.txt
drug_interaction_env/__init__.py

---

## File 1: models.py (Build First — Hour 1)

from pydantic import BaseModel
from typing import Literal, Optional

class DrugInteractionAction(BaseModel):
    action_type: Literal["flag_interaction", "DONE"]
    drug_a: Optional[str] = None
    drug_b: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    suggested_action: Optional[Literal["monitor", "reduce_dose", "replace_drug"]] = None

class FlagEntry(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    suggested_action: str
    step: int
    reward_received: float

class DrugInteractionObservation(BaseModel):
    patient_id: str
    age: int
    conditions: list[str]
    medications: list[str]
    flags_raised_so_far: list[FlagEntry]
    steps_remaining: int

class PredictionEntry(BaseModel):
    predicted_severity: str
    predicted_action: str
    ground_truth_severity: str
    ground_truth_action: str
    reward_received: float
    perfectly_completed: bool

class DrugInteractionState(BaseModel):
    patient_id: str
    step_count: int
    task_level: str
    attempted_keys: list[str]
    identified_pairs: list[str]
    perfectly_completed_pairs: list[str]
    predictions: dict
    done: bool

class ResetRequest(BaseModel):
    task_level: Literal["easy", "medium", "hard"] = "easy"

class StepResponse(BaseModel):
    observation: DrugInteractionObservation
    reward: float
    done: bool
    state: DrugInteractionState

---

## File 2: app.py (Hours 1-3)

### Hour 1-2: Build with stubs (do NOT wait for Person A)

from fastapi import FastAPI, HTTPException
from models import *  # import all models above

app = FastAPI(title="Drug Interaction Environment")

# STUB — replace with real environment in Hour 3
_env_state = {}

@app.post("/reset")
def reset(request: ResetRequest):
    # STUB: return a hardcoded observation shape for now
    return {
        "observation": {
            "patient_id": "STUB",
            "age": 0,
            "conditions": [],
            "medications": [],
            "flags_raised_so_far": [],
            "steps_remaining": 9
        },
        "reward": 0.0,
        "done": False,
        "state": {}
    }

@app.post("/step")
def step(action: DrugInteractionAction):
    # STUB
    return {"observation": {}, "reward": 0.0, "done": False, "state": {}}

@app.get("/state")
def state():
    # STUB
    return {}

### Hour 3: Wire up Person A's environment (after they finish DrugInteractionEnvironment)

Replace stubs with:

from server.drug_interaction_environment import DrugInteractionEnvironment

env = DrugInteractionEnvironment()

@app.post("/reset")
def reset(request: ResetRequest):
    obs = env.reset(request.task_level)
    state = env.state()
    return {"observation": obs, "reward": 0.0, "done": False, "state": state}

@app.post("/step")
def step(action: DrugInteractionAction):
    action_dict = action.model_dump()
    obs, reward, done, state = env.step(action_dict)
    return StepResponse(observation=obs, reward=reward, done=done, state=state)

@app.get("/state")
def get_state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok"}

---

## File 3: openenv.yaml

spec_version: 1
name: drug-interaction-env
type: step-reset
runtime: docker
app: server/app.py
port: 8000

---

## File 4: Dockerfile

FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

---

## File 5: server/requirements.txt

openenv-core
fastapi
uvicorn
pydantic>=2.0
openai

---

## File 6: __init__.py

from .models import DrugInteractionAction, DrugInteractionObservation, DrugInteractionState

---

## Hour 4: Deployment Checklist

1. Build Docker image locally:
       docker build -t drug-interaction-env .
       docker run -p 8000:8000 drug-interaction-env

2. Test all endpoints:
       curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_level":"easy"}'
       curl http://localhost:8000/state
       curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
            -d '{"action_type":"flag_interaction","drug_a":"warfarin","drug_b":"aspirin","severity":"severe","suggested_action":"replace_drug"}'

3. Run pre-submission validator script

4. Push to Hugging Face Spaces
       huggingface-cli login
       openenv push --enable-interface

5. Verify Space URL returns 200 and responds to reset()
```

---

---

# PERSON C — Prompt for Antigravity

```
You are implementing the INFERENCE SCRIPT + GRADER layer of a Pharmaceutical Drug
Interaction Checker. You build the LLM inference loop, episode logging, and per-task
grading logic. You work client-side only — no FastAPI, no environment internals.

In Hours 1-2, test against a local mock server you build yourself.
In Hours 3-4, switch to Person B's live server.

---

## Your Files to Build

inference.py                         (root directory — mandatory)
drug_interaction_env/grader.py

---

## File 1: mock_env.py (Hour 1 — local stub for testing, NOT submitted)

Build a minimal Flask or raw HTTP mock that mimics /reset and /step responses
so you can develop inference.py without waiting for Person B's server.

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

MOCK_RESET_RESPONSE = {
    "observation": {
        "patient_id": "P001",
        "age": 67,
        "conditions": ["hypertension"],
        "medications": ["warfarin", "aspirin", "metformin", "lisinopril", "vitamin_d", "omeprazole"],
        "flags_raised_so_far": [],
        "steps_remaining": 3
    },
    "reward": 0.0,
    "done": False,
    "state": {"task_level": "easy", "step_count": 0}
}

MOCK_STEP_RESPONSE = {
    "observation": { ... },   # same shape, steps_remaining -= 1
    "reward": 0.8,
    "done": True,
    "state": { ... }
}

Run this mock on localhost:8001 during Hours 1-2.
Switch BASE_URL to Person B's server (localhost:8000) in Hour 3.

---

## File 2: inference.py (Root Directory — Hours 1-3)

### Required environment variables:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face / API key
    ENV_BASE_URL  — environment server URL (default: http://localhost:8000)

### Required imports:
    import os, json, requests
    from openai import OpenAI

### LLM client setup:
    llm_client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"]
    )

### System prompt (pass this every call):
    SYSTEM_PROMPT = """
    You are a clinical pharmacist reviewing a patient medication list for dangerous drug interactions.
    
    For each step, output ONLY a valid JSON object in one of these two formats:
    1. {"action_type": "flag_interaction", "drug_a": "...", "drug_b": "...",
        "severity": "mild|moderate|severe", "suggested_action": "monitor|reduce_dose|replace_drug"}
    2. {"action_type": "DONE"}

    Rules:
    - Only flag drugs that appear in the patient medications list
    - Never flag the same pair twice (check flags_raised_so_far)
    - A pair flagged with wrong severity or action is LOCKED IN — cannot be changed
    - Missing a severe interaction costs -0.4 at episode end
    - Wrong severity deducts: severe=-0.2, moderate=-0.1, mild=-0.05
    - Wrong action deducts: -0.1
    - Phantom pairs (not real interactions) cost -0.3
    - Duplicates cost -0.05
    - Send DONE only when confident all interactions are found
    - Output ONLY the JSON object, no explanation
    """

### Main inference loop — run for all 3 tasks:

    TASKS = ["easy", "medium", "hard"]
    ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

    for task in TASKS:
        # Reset environment
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset",
                                   json={"task_level": task}).json()
        observation = reset_resp["observation"]
        patient_id = observation["patient_id"]

        # MANDATORY log format
        print(f"[START] task={task} patient_id={patient_id}")

        step_num = 0
        done = False
        episode_score = 0.0

        while not done:
            step_num += 1

            # Build user message from observation
            user_message = f"Patient profile and current state:\n{json.dumps(observation, indent=2)}"

            # Call LLM
            response = llm_client.chat.completions.create(
                model=os.environ["MODEL_NAME"],
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )

            raw_output = response.choices[0].message.content.strip()

            # Parse action JSON
            try:
                action = json.loads(raw_output)
            except json.JSONDecodeError:
                # Malformed output — send DONE as fallback
                action = {"action_type": "DONE"}

            # Send to environment
            step_resp = requests.post(f"{ENV_BASE_URL}/step", json=action).json()
            reward = step_resp["reward"]
            done = step_resp["done"]
            observation = step_resp["observation"]

            # MANDATORY log format
            if action["action_type"] == "DONE":
                print(f"[STEP] step={step_num} action=DONE reward={reward}")
            else:
                print(f"[STEP] step={step_num} action=flag_interaction "
                      f"drug_a={action.get('drug_a','')} drug_b={action.get('drug_b','')} "
                      f"severity={action.get('severity','')} "
                      f"suggested_action={action.get('suggested_action','')} "
                      f"reward={reward}")

        # Compute episode score from grader
        episode_score = grade_episode(task, step_resp["state"])

        # MANDATORY log format
        print(f"[END] task={task} patient_id={patient_id} episode_score={episode_score}")

---

## File 3: grader.py (Hours 2-3)

    from drug_interaction_env.server.drug_database import DRUG_INTERACTIONS

    TASK_TRUE_PAIRS = {
        "easy":   1,
        "medium": 3,
        "hard":   5
    }

    def grade_episode(task_level: str, final_state: dict) -> float:
        """
        Takes the final state dict from the environment at episode end.
        Computes normalized episode score in [0.0, 1.0].
        """
        n_true_pairs = TASK_TRUE_PAIRS[task_level]
        max_possible = n_true_pairs * 0.8

        predictions = final_state.get("predictions", {})
        identified = set(final_state.get("identified_pairs", []))
        ground_truth_keys = set(final_state.get("ground_truth_keys", []))

        # Sum rewards from all predictions
        total_reward = sum(p["reward_received"] for p in predictions.values())

        # Apply severity-weighted termination penalties for unidentified pairs
        unidentified = ground_truth_keys - identified
        for key_str in unidentified:
            key = tuple(key_str.split("|"))    # adjust based on serialization format
            gt = DRUG_INTERACTIONS.get(key, {})
            severity = gt.get("severity", "mild")
            if severity == "severe":
                total_reward -= 0.4
            elif severity == "moderate":
                total_reward -= 0.3
            else:
                total_reward -= 0.2

        # Normalize
        if max_possible == 0:
            return 1.0
        return max(0.0, min(1.0, total_reward / max_possible))

    def verify_score_range(score: float) -> bool:
        return 0.0 <= score <= 1.0

---

## Hour 4: End-to-End Verification Checklist

1. Switch ENV_BASE_URL to Person B's live Docker server (localhost:8000)
2. Run: python inference.py
3. Verify stdout contains exactly:
       - One [START] line per task
       - One [STEP] line per step
       - One [END] line per task with episode_score in [0.0, 1.0]
4. Verify all 3 tasks complete without error
5. Verify total runtime < 20 minutes
6. Verify episode_score values are valid floats between 0.0 and 1.0

If LLM is hallucinating drug names not in medications list:
    → The environment returns -0.3 (invalid drug name)
    → This is expected behavior, grader handles it

If inference.py crashes on malformed LLM JSON:
    → The try/except in action parsing sends DONE as fallback
    → Episode ends cleanly with termination penalties applied
```
