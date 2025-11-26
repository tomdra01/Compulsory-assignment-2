import os
import json
import autogen
from dotenv import load_dotenv
from research_tools import search_research_papers

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    print("Error: MISTRAL_API_KEY not found. Please check your .env file.")
    exit(1)

# --- 2. SCENARIO CONFIGURATION (The "Placeholders") ---
# Change these values to test different scenarios for your assignment video!
TOPIC = "Large Language Models"
YEAR = "2023"
MIN_CITATIONS = 10

# Automatically build the task string based on variables above
TASK_PROMPT = (
    f"Find a research paper on '{TOPIC}' "
    f"that was published in {YEAR} "
    f"and has at least {MIN_CITATIONS} citations."
)

# --- 3. AGENT CONFIGURATION ---
llm_config = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": api_key,
            "api_type": "mistral",
            "api_rate_limit": 0.25,
            "repeat_penalty": 1.1,
            "temperature": 0.5,
            "seed": 42,
        }
    ]
}

# User Proxy (Manager)
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# Researcher (Solver)
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="""You are a helpful academic researcher. 
    1. Use 'search_research_papers' to find papers matching the user's specific topic, year, and citation requirements.
    2. Analyze the results from the tool.
    3. If a paper meets ALL criteria, present the Title, Year, and Citation count.
    4. You must listen to the 'Critic'. If they reject your answer, search again or fix the error.
    5. Only say 'TERMINATE' when the Critic approves.""",
    llm_config=llm_config,
)

# Critic (Internal Evaluator)
critic = autogen.AssistantAgent(
    name="Critic",
    system_message=f"""You are a strict reviewer. 
    Check if the Researcher's found paper matches these EXACT requirements:
    - Topic: {TOPIC}
    - Year: {YEAR}
    - Min Citations: {MIN_CITATIONS}

    If it matches, reply EXACTLY: "TERMINATE".
    If not, explain what is wrong (e.g., "Wrong year", "Not enough citations").""",
    llm_config=llm_config,
)

# Judge (External Evaluator)
judge = autogen.AssistantAgent(
    name="Judge",
    llm_config=llm_config,
    system_message="""You are an external evaluator.
    Score the final answer based on:
    - Correctness (0-5)
    - Constraints Met (0-5)
    Return ONLY JSON: {"score": int, "explanation": "string"}"""
)

# Register Tool
autogen.agentchat.register_function(
    search_research_papers,
    caller=researcher,
    executor=user_proxy,
    name="search_research_papers",
    description="Searches for research papers based on topic, year, and citation count."
)

# --- 4. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    if not os.path.exists("coding"):
        os.makedirs("coding")

    print(f"Starting Agent with Task: \n'{TASK_PROMPT}'\n" + "-" * 50)

    # Use round_robin to prevent Mistral API crashing
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, critic],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin"
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    scores = []

    # Run the evaluation 2 times (Batch)
    for i in range(2):
        print(f"\n--- RUN {i + 1} ---")
        groupchat.messages.clear()

        try:
            user_proxy.initiate_chat(manager, message=TASK_PROMPT)
        except Exception as e:
            # --- FIX FOR THE CEREBRAS ERROR ---
            error_msg = str(e)
            if "cerebras" in error_msg:
                # This is a known library bug that happens AFTER the work is done.
                # We can safely ignore it.
                pass
            else:
                print(f"Run {i + 1} Error: {e}")

        # Evaluation Phase
        if not groupchat.messages:
            print("No messages found.")
            continue

        last_msg = groupchat.messages[-1]['content']

        print(f"\n[Run {i + 1}] Judging...")
        eval_chat = user_proxy.initiate_chat(
            judge,
            message=f"Task: {TASK_PROMPT}\nResult: {last_msg}\nGrade this as JSON.",
            max_turns=1,
            summary_method="last_msg"
        )

        try:
            content = eval_chat.chat_history[-1]['content']
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            result_json = json.loads(json_str)
            scores.append(result_json['score'])
            print(f"Run {i + 1} Score: {result_json['score']}/5")
            print(f"Reasoning: {result_json.get('explanation', 'None')}")
        except:
            print(f"Run {i + 1} Score: Error parsing JSON")
            scores.append(0)

    if scores:
        print(f"\n{'-' * 30}\nAverage Score: {sum(scores) / len(scores)}")