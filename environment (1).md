# environment.md — Pharmaceutical Drug Interaction Checker

## Environment Overview

A **clinical pharmacist simulator** built on the OpenEnv framework. The environment loads a patient profile, exposes the medication list to an LLM agent, and evaluates whether the agent can correctly identify dangerous drug-drug interactions, classify their severity, and recommend an appropriate clinical action.

This is a real-world task — hospital pharmacists perform this review daily and incorrect flagging (missing a severe interaction or hallucinating a non-existent one) directly maps to patient harm.

---

## OpenEnv Spec Compliance

The environment implements the full OpenEnv interface:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Initialize a new episode, load patient profile |
| `/step` | POST | Submit one agent action, receive reward + observation |
| `/state` | GET | Return current episode metadata |

All action and observation types are **typed Pydantic models** defined in `models.py`.

---

## File Structure

```
drug_interaction_env/
├── __init__.py                  # Export Action, Observation, DrugInteractionEnv
├── models.py                    # Typed Action, Observation, State dataclasses
├── client.py                    # DrugInteractionEnv(EnvClient)
├── openenv.yaml                 # Environment manifest
├── pyproject.toml               # Dependencies
├── inference.py                 # Baseline inference script (root level, mandatory)
├── README.md
├── outputs/
│   ├── logs/
│   └── evals/
└── server/
    ├── drug_interaction_environment.py   # Core environment logic
    ├── drug_database.py                  # Hardcoded DRUG_INTERACTIONS dict
    ├── patients.py                       # Patient scenario definitions
    ├── app.py                            # FastAPI app
    ├── requirements.txt
    └── Dockerfile
```

---

## openenv.yaml

```yaml
spec_version: 1
name: drug-interaction-env
type: step-reset
runtime: docker
app: server/app.py
port: 8000
```

---

## Typed Models (models.py)

### Action
```python
from pydantic import BaseModel
from typing import Literal, Optional

class DrugInteractionAction(BaseModel):
    action_type: Literal["flag_interaction", "DONE"]
    drug_a: Optional[str] = None
    drug_b: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    suggested_action: Optional[Literal["monitor", "reduce_dose", "replace_drug"]] = None
```

### Observation
```python
class FlagEntry(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    suggested_action: str
    step: int
    reward_received: float    # agent can see how well this flag scored, helps it calibrate next action

class DrugInteractionObservation(BaseModel):
    patient_id: str
    age: int
    conditions: list[str]
    medications: list[str]
    flags_raised_so_far: list[FlagEntry]
    steps_remaining: int
```

### State
```python
class PredictionEntry(BaseModel):
    key: tuple                      # normalized sorted (drug_a, drug_b)
    predicted_severity: str
    predicted_action: str
    ground_truth_severity: str
    ground_truth_action: str
    reward_received: float
    perfectly_completed: bool

class DrugInteractionState(BaseModel):
    patient_id: str
    step_count: int
    task_level: Literal["easy", "medium", "hard"]
    attempted_keys: list[str]            # ALL keys submitted, used only for duplicate detection
    identified_pairs: list[str]          # valid pairs found (validate() == True), severity/action may be wrong
    perfectly_completed_pairs: list[str] # subset of identified_pairs — all three correct
    predictions: dict                    # key → PredictionEntry, updated only on validate() == True
    done: bool
```

---

## Drug Interaction Database (drug_database.py)

Hardcoded Python dict of ~30-40 publicly documented interactions. No external API needed.

```python
DRUG_INTERACTIONS = {
    ("warfarin", "aspirin"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Increased bleeding risk"
    },
    ("warfarin", "ibuprofen"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Increased bleeding risk, GI complications"
    },
    ("metformin", "ibuprofen"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Reduced renal clearance of metformin"
    },
    ("simvastatin", "amiodarone"): {
        "severity": "severe",
        "action": "reduce_dose",
        "explanation": "Risk of myopathy and rhabdomyolysis"
    },
    ("ssri", "tramadol"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Serotonin syndrome risk"
    },
    # ... ~30-40 pairs total
}
```

**Lookup is always normalized** — `(drug_a, drug_b)` and `(drug_b, drug_a)` treated as the same pair:
```python
def lookup_pair(drug_a, drug_b):
    key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
    return DRUG_INTERACTIONS.get(key, None)
```

---

## Patient Scenarios (patients.py)

```python
PATIENTS = {
    "task_easy_01": {
        "patient_id": "P001",
        "age": 67,
        "conditions": ["hypertension", "type2_diabetes"],
        "medications": [
            "warfarin", "aspirin", "metformin",
            "lisinopril", "vitamin_d", "omeprazole"
        ]
        # ground truth derived at runtime from DRUG_INTERACTIONS lookup
    },
    "task_medium_01": {
        "patient_id": "P002",
        "age": 54,
        "conditions": ["atrial_fibrillation", "chronic_pain"],
        "medications": [
            "warfarin", "ibuprofen", "amiodarone",
            "simvastatin", "metformin", "lisinopril",
            "omeprazole", "vitamin_d", "aspirin", "atorvastatin"
        ]
        # 3 interactions of mixed severity
    },
    "task_hard_01": {
        "patient_id": "P003",
        "age": 71,
        "conditions": ["heart_failure", "depression", "type2_diabetes"],
        "medications": [
            "warfarin", "aspirin", "amiodarone", "simvastatin",
            "ssri", "tramadol", "metformin", "ibuprofen",
            "lisinopril", "digoxin", "furosemide", "spironolactone",
            "omeprazole", "vitamin_d", "atorvastatin"
        ]
        # 5 interactions across severity levels
    }
}
```

Ground truth interactions are **derived at runtime** by scanning the patient's medication list against `DRUG_INTERACTIONS` — not hardcoded per patient. This keeps data consistent and single-source-of-truth.

---

## step() / reset() / state() Signatures

### reset()
```python
def reset(task_level: Literal["easy", "medium", "hard"] = "easy") -> DrugInteractionObservation:
    # Load patient for task_level
    # Initialize step_count = 0
    # Initialize flags_raised_so_far = []
    # Compute ground_truth_interactions from patient medications × DRUG_INTERACTIONS
    # Set max_steps = len(ground_truth_interactions) × 3
    # Return initial observation
```

### step()
```python
def step(action: DrugInteractionAction) -> tuple[DrugInteractionObservation, float, bool, DrugInteractionState]:
    # 1. If action_type == "DONE" → run termination, return penalties
    # 2. Run validate(action, identified_pairs, flagged_keys)
    #       → returns False  : step_reward = -0.3
    #       → returns None   : step_reward = 0.0  (duplicate)
    #       → returns True   : pass to calculate_reward()
    # 3. If True → calculate_reward() → update identified_pairs / perfectly_completed_pairs
    # 4. Append PredictionEntry to predictions tracker
    # 5. Increment step_count, update flagged_keys
    # 6. Check termination conditions
    # 7. Return (observation, step_reward, done, state)
```

### state()
```python
def state() -> DrugInteractionState:
    # Return current patient_id, step_count, task_level, remaining_unflagged_pairs, done
```

---

## Task Definitions

### EASY
- **Medications:** 6
- **True Interactions:** 1 (severe, well-known — e.g. warfarin + aspirin)
- **max_steps:** 3
- **Success:** Agent flags the 1 correct pair with correct severity + action

### MEDIUM
- **Medications:** 10
- **True Interactions:** 3 (mixed severity — 1 severe, 1 moderate, 1 mild)
- **max_steps:** 9
- **Success:** Agent flags all 3 with correct severity + action

### HARD
- **Medications:** 15
- **True Interactions:** 5 (across all severity levels, more pair combinations to reason over)
- **max_steps:** 15
- **Success:** Agent flags all 5 with correct severity + action

---

## Termination Logic

```
Condition 1 — Perfect Completion:
    IF len(perfectly_completed_pairs) == len(ground_truth_keys):
        done = True
        no termination penalty
    NOTE: pair in identified_pairs but NOT perfectly_completed_pairs → episode continues

Condition 2 — DONE Signal:
    IF action_type == "DONE":
        unidentified_pairs = ground_truth_keys - identified_pairs
        for each pair in unidentified_pairs:
            if severity == "severe":   episode_reward -= 0.4
            if severity == "moderate": episode_reward -= 0.3
            if severity == "mild":     episode_reward -= 0.2
        partially_flagged_pairs → NO additional penalty
        done = True

Condition 3 — Step Budget Exhausted:
    IF step_count >= max_steps:
        same severity-weighted penalty logic as Condition 2
        done = True
```

**Duplicate Handling:**
```
key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
(warfarin, aspirin) == (aspirin, warfarin) → same key
IF key in attempted_keys → return None → reward = -0.05
attempted_keys tracks ALL submitted keys including phantom pairs
identified_pairs tracks ONLY valid pairs (validate() == True)
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
