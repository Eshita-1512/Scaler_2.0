# agents.md — Pharmaceutical Drug Interaction Checker

## Agent Overview

The agent simulates a **clinical pharmacist**. It receives a patient profile containing age, conditions, and a list of medications, and must systematically identify dangerous drug-drug interactions, classify their severity, and suggest a clinical action.

The agent operates in a **single-patient episode** — one patient per `reset()` call. It iterates through the medication list step-by-step, flagging one drug pair per action until it believes all interactions have been found, or until the step budget is exhausted.

---

## What the Agent Receives (Observation)

Each step, the agent receives the following observation as a JSON object:

```json
{
  "patient_id": "P001",
  "age": 67,
  "conditions": ["hypertension", "type2_diabetes"],
  "medications": ["warfarin", "aspirin", "metformin", "lisinopril", "vitamin_d", "omeprazole"],
  "flags_raised_so_far": [
    {
      "drug_a": "warfarin",
      "drug_b": "aspirin",
      "severity": "severe",
      "suggested_action": "replace_drug",
      "step": 1,
      "reward_received": 0.8
    }
  ],
  "steps_remaining": 17
}
```

**Note on duplicates:** `(drug_a, drug_b)` and `(drug_b, drug_a)` are treated as the same pair. If the agent flags a pair whose normalized key already exists in `flags_raised_so_far`, the step returns `reward = 0.0` with no penalty — the flag is silently ignored.

---

## What the Agent Outputs (Action)

The agent outputs **one of two action types per step**:

### Action Type 1 — Flag an Interaction
```json
{
  "action_type": "flag_interaction",
  "drug_a": "warfarin",
  "drug_b": "aspirin",
  "severity": "severe",
  "suggested_action": "replace_drug"
}
```

**Valid severity values:** `mild` | `moderate` | `severe`

**Valid suggested_action values:** `monitor` | `reduce_dose` | `replace_drug`

### Action Type 2 — Declare Done
```json
{
  "action_type": "DONE"
}
```

Sent when the agent believes all interactions have been identified. Triggers termination logic.

---

## Agent Loop Flow

```
reset() called
    → agent receives initial observation (patient profile, full medication list, empty flags)

LOOP:
    agent reasons over observation
        → scans medication list for potential interacting pairs
        → checks flags_raised_so_far to avoid duplicates
    agent outputs action JSON
        → action sent to step()
    step() returns (observation, step_reward, done, state)
        → agent receives updated observation with new flag appended
        → agent receives step_reward signal
    if done == True:
        episode ends
```

---

## Termination Conditions

Episode ends when **any one** of the following fires:

1. **Perfect Completion** — All ground truth pairs in `perfectly_completed_pairs` → `done=True`, no penalty. Pairs in `identified_pairs` but not `perfectly_completed_pairs` keep the episode running.
2. **Agent DONE Signal** — Sends `{"action_type": "DONE"}` → severity-weighted penalty per pair in `unidentified_pairs` (`-0.4 / -0.3 / -0.2`). Partially flagged pairs receive no extra penalty.
3. **Step Budget Exhausted** — `step_count >= max_steps` → same penalty logic as DONE signal.

**Duplicate rule:** If agent re-flags a key already in `attempted_keys` → `reward = -0.05`. Small penalty to discourage looping without being harsh.

---

## LLM Prompting Strategy

The system prompt should instruct the model to:

- Reason over all possible pairs from the medication list before flagging
- Only flag drugs that appear in the patient's `medications` list — flagging unknown drug names returns `-0.3`
- Never re-flag a pair already in `flags_raised_so_far` — duplicates return `-0.05`
- Be conservative — phantom pairs (not real interactions) return `-0.3`
- Pay close attention to severity — wrong severity actively deducts (`-0.2` severe, `-0.1` moderate, `-0.05` mild)
- Pay close attention to action — wrong action deducts `-0.1` even if pair and severity correct
- A flagged pair cannot be re-flagged — locked in regardless of severity/action correctness
- Send `DONE` only when confident all interactions are found — missing severe pairs costs `-0.4` each

**Recommended prompt structure:**
```
You are a clinical pharmacist reviewing a patient's medication list for dangerous drug interactions.

Patient Profile: {observation}

For each step, output ONLY a valid JSON object in one of these two formats:
1. Flag an interaction: {"action_type": "flag_interaction", "drug_a": "...", "drug_b": "...", "severity": "mild|moderate|severe", "suggested_action": "monitor|reduce_dose|replace_drug"}
2. Declare done: {"action_type": "DONE"}

Rules:
- Flag only ONE pair per step
- Do not repeat pairs already in flags_raised_so_far
- Only flag pairs you are confident interact dangerously
- Send DONE when you have found all interactions
```

---

## Inference Script Requirements (Mandatory)

The `inference.py` file must be placed in the **root directory** and must emit structured stdout logs in the following format:

```
[START] task=easy patient_id=P001
[STEP] step=1 action=flag_interaction drug_a=warfarin drug_b=aspirin severity=severe suggested_action=replace_drug reward=0.8
[STEP] step=2 action=DONE reward=-0.0
[END] task=easy patient_id=P001 episode_score=1.0
```

**Required environment variables for inference script:**
- `API_BASE_URL` — LLM API endpoint
- `MODEL_NAME` — model identifier
- `HF_TOKEN` — Hugging Face / API key

**All LLM calls must use the OpenAI client:**
```python
from openai import OpenAI
client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["HF_TOKEN"])
response = client.chat.completions.create(model=os.environ["MODEL_NAME"], ...)
```

**Infra constraints:**
- Total inference runtime must be **< 20 minutes**
- Must run on **2 vCPU, 8GB RAM**
