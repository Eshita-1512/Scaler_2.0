# requirements.md — Pharmaceutical Drug Interaction Checker

---

## Reward Function — One-Line Summary

```
# Per Step
step_reward = 0.4(pair_correct) + 0.2(severity_correct) + 0.2(action_correct)
            - 0.2/0.1/0.05(severity_wrong by severe/moderate/mild)
            - 0.1(action_wrong)

# Step Penalties (inside step())
phantom_pair   = -0.3       [validate() returned False]
duplicate_pair = -0.05      [pair already attempted]

# Termination Penalties (applied at episode end only)
severe_missed   = -0.4 × n    [ground truth severe, never identified]
moderate_missed = -0.3 × n    [ground truth moderate, never identified]
mild_missed     = -0.2 × n    [ground truth mild, never identified]

severity_score = NIL if validate() == False
action_score   = NIL if validate() == False

perfect_completion = pair_correct AND severity_correct AND action_correct for ALL pairs
episode_score = clip(Σstep_rewards + Σtermination_penalties) / (n_true_pairs × 0.8), 0.0, 1.0)
```

---

## Tracking State — Definitions (Single Source of Truth)

```
attempted_keys: set
    ALL keys the agent has ever submitted, regardless of outcome.
    Includes phantom pairs, valid pairs, duplicates.
    Used ONLY for duplicate detection in validate().
    NOT the same as identified_pairs.

identified_pairs: set
    Pairs where validate() returned True.
    Correct (drug_a, drug_b) key found in DRUG_INTERACTIONS.
    Severity and action may or may not be correct.
    Used to determine unidentified_pairs at termination.

perfectly_completed_pairs: set
    Subset of identified_pairs where severity AND action were also correct.
    Triggers perfect completion termination when == ground_truth_keys.

partially_flagged_pairs: set (derived)
    = identified_pairs - perfectly_completed_pairs
    Agent found the pair but got severity or action wrong.
    No termination penalty. Partial reward already received.

unidentified_pairs: set (derived at termination)
    = ground_truth_keys - identified_pairs
    Pairs agent never found at all.
    These incur severity-weighted termination penalty.

predictions: dict
    Maps normalized pair key → {predicted_severity, predicted_action,
                                  ground_truth_severity, ground_truth_action,
                                  reward_received, perfectly_completed}
    Updated ONLY when validate() == True.
    Used for reward calculation and grading.
```

---

## `validate(action, identified_pairs, attempted_keys)` — Runs First Inside step()

Returns one of: `True` | `False` | `None`

```
IF action_type == "DONE":
    → trigger termination (not handled in validate)

IF action_type == "flag_interaction":

    STEP 1 — Normalize key (handles (a,b) == (b,a)):
        key = tuple(sorted([drug_a.lower(), drug_b.lower()]))

    STEP 2 — Validate drug names are in patient medication list:
        IF drug_a not in medications OR drug_b not in medications:
            → return False                 [invalid drug name → -0.3]

    STEP 3 — Check for duplicate:
        IF key in attempted_keys:
            → return None                  [duplicate → -0.05]

    STEP 4 — Add to attempted_keys:
        attempted_keys.add(key)

    STEP 5 — Check DRUG_INTERACTIONS:
        IF key NOT in DRUG_INTERACTIONS:
            → return False                 [phantom pair → -0.3]
        IF key IN DRUG_INTERACTIONS:
            → return True                  [pass to calculate_reward()]

Reward assignment:
    True  → step_reward = calculate_reward(key, severity, action)
    False → step_reward = -0.3
    None  → step_reward = -0.05  (duplicate discouragement)
```

---

## `calculate_reward(key, severity, action)` — Runs Only if validate() == True

```
base_score     = +0.4    [pair correctly identified]

severity_score = +0.2    if predicted severity == ground_truth severity
               = -0.2    if ground_truth severity is "severe"   and prediction wrong
               = -0.1    if ground_truth severity is "moderate" and prediction wrong
               = -0.05   if ground_truth severity is "mild"     and prediction wrong

action_score   = +0.2    if predicted action == ground_truth action
               = -0.1    if predicted action != ground_truth action

step_reward = base_score + severity_score + action_score
            → range: [0.1, 0.8]

After calculate_reward():
    identified_pairs.add(key)
    predictions[key] = {
        predicted_severity, predicted_action,
        ground_truth_severity, ground_truth_action,
        reward_received, perfectly_completed
    }
    IF severity_correct AND action_correct:
        perfectly_completed_pairs.add(key)
```

**Example — correct pair, severe ground truth, both wrong:**
`0.4 - 0.2 - 0.1 = 0.1` — still positive, pair found but clinical details wrong.

---

## Perfect Completion Definition

```
perfect_completion = True
    IF len(perfectly_completed_pairs) == len(ground_truth_keys)
    → episode terminates immediately, no termination penalty
```

A pair in `identified_pairs` but NOT `perfectly_completed_pairs` → episode continues.

---

## Termination Penalties

Applied ONLY at episode end on DONE signal or step budget exhaustion.
NOT applied on perfect completion.

```
unidentified_pairs = ground_truth_keys - identified_pairs

for each pair in unidentified_pairs:
    IF ground_truth severity == "severe":
        episode_reward -= 0.4
    IF ground_truth severity == "moderate":
        episode_reward -= 0.3
    IF ground_truth severity == "mild":
        episode_reward -= 0.2

partially_flagged_pairs = identified_pairs - perfectly_completed_pairs
→ NO additional termination penalty (partial reward already received at step time)
```

**Why severity-weighted termination matters:**
Missing a severe interaction in a real clinical setting is more dangerous than missing a mild one.
The penalty gradient reflects this: `-0.4 / -0.3 / -0.2` per missed severity tier.

---

## Episode Score Normalization

```python
max_possible = num_true_pairs × 0.8
raw_score = sum(all step_rewards) + sum(termination_penalties)
episode_score = clip(raw_score / max_possible, 0.0, 1.0)
```

---

## Grader Logic (Per Task)

```python
def grade(episode_log: list[StepLog]) -> float:
    # Sum all step rewards from episode log
    # Apply severity-weighted termination penalties for unidentified pairs
    # Normalize to [0.0, 1.0]
    # Return final episode score
```

| Task   | n_true_pairs | max_possible_raw | max_steps |
|--------|-------------|-----------------|-----------|
| Easy   | 1           | 0.8             | 3         |
| Medium | 3           | 2.4             | 9         |
| Hard   | 5           | 4.0             | 15        |

---

## Environment Variables

```bash
API_BASE_URL=<LLM API endpoint>
MODEL_NAME=<model identifier>
HF_TOKEN=<Hugging Face / API key>
```

---

## Dependencies

```txt
# server/requirements.txt
openenv-core
fastapi
uvicorn
pydantic>=2.0
openai
```

---

## Mandatory Inference Script Requirements

- Named `inference.py`, placed in **root directory**
- Must use OpenAI client for all LLM calls
- Must emit structured stdout logs:

```
[START] task=easy patient_id=P001
[STEP] step=1 action=flag_interaction drug_a=warfarin drug_b=aspirin severity=severe suggested_action=replace_drug reward=0.8
[STEP] step=2 action=DONE reward=0.0
[END] task=easy patient_id=P001 episode_score=1.0
```

- Must run all 3 tasks without error
- Must complete in **under 20 minutes**
- Must run on **2 vCPU, 8GB RAM**

---

## Pre-Submission Checklist

| Check | Requirement |
|---|---|
| HF Space deploys | Space URL returns 200, responds to `reset()` |
| OpenEnv spec compliance | `openenv.yaml` valid, typed models, `step()`/`reset()`/`state()` live |
| Dockerfile builds | `docker build` completes without error |
| Baseline reproduces | `inference.py` runs end-to-end and produces scores |
| 3 tasks with graders | All graders run, all scores in `[0.0, 1.0]` |
| Stdout log format | `[START]`, `[STEP]`, `[END]` strictly followed |
| Env variables set | `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` defined |
| Runtime < 20 min | Full inference script within limit |

---

## Evaluation Criteria Mapping

| Judging Criterion | How This Project Satisfies It |
|---|---|
| Runtime correctness | FastAPI server, Dockerfile, all endpoints tested |
| Interface compliance | Full OpenEnv spec — typed models, `step()`/`reset()`/`state()` |
| Task design | 3 tasks, realistic clinical scenario, clear ground truth |
| Grading logic | Two-function reward, severity-weighted penalties, normalized `[0.0, 1.0]` |
| Meaningful reward | Partial progress at pair, severity, and action level |
| Real-world task | Drug interaction checking is a genuine daily pharmacist workflow |
