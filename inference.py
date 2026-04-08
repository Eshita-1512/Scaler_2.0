import os
import json
import requests
from openai import OpenAI
from drug_interaction_env.grader import grade_episode

def main():
    api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000") # Replace with valid default if available
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo") # Update with appropriate model name
    hf_token = os.environ.get("HF_TOKEN", "dummy_token")
    env_base_url = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

    llm_client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token
    )

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

    TASKS = ["easy", "medium", "hard"]

    for task in TASKS:
        # Reset environment
        reset_resp = requests.post(f"{env_base_url}/reset",
                                   json={"task_level": task}).json()
        observation = reset_resp["observation"]
        patient_id = observation["patient_id"]

        print(f"[START] task={task} patient_id={patient_id}")

        step_num = 0
        done = False
        episode_score = 0.0
        final_state = reset_resp["state"]

        while not done:
            step_num += 1
            user_message = f"Patient profile and current state:\n{json.dumps(observation, indent=2)}"

            # Call LLM
            response = llm_client.chat.completions.create(
                model=model_name,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )

            raw_output = response.choices[0].message.content.strip()

            # Parse action JSON
            # if the output is markdown code block we extract JSON
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()

            try:
                action = json.loads(raw_output)
            except json.JSONDecodeError:
                # Malformed output — send DONE as fallback
                action = {"action_type": "DONE"}

            # Send to environment
            step_resp = requests.post(f"{env_base_url}/step", json=action).json()
            reward = step_resp["reward"]
            done = step_resp["done"]
            observation = step_resp["observation"]
            final_state = step_resp.get("state", final_state)

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
        episode_score = grade_episode(task, final_state)

        # MANDATORY log format
        print(f"[END] task={task} patient_id={patient_id} episode_score={episode_score}")

if __name__ == "__main__":
    main()
