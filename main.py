import os
from dotenv import load_dotenv
import autogen
from research_tools import search_research_papers

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path, verbose=True)
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    print("Error: MISTRAL_API_KEY not found. Please check your .env file.")
    exit(1)

# The specific fork you installed allows 'api_rate_limit' to work here.
llm_config = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": api_key,
            "api_type": "mistral",
            "api_rate_limit": 0.25,
            "repeat_penalty": 1.1,
            "temperature": 0.0,
            "seed": 42,
            "stream": False,
            "native_tool_calls": False,
            "cache_seed": None,
        }
    ]
}


# User Proxy: Acts as the "manager" who executes the tool code
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",  # The agent runs autonomously
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",  # Folder where code is saved/run
        "use_docker": False  # Run locally (simpler for this assignment)
    },
)

# Researcher: The AI that plans and calls the search tool
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="""You are a helpful academic researcher. 
    Your task is to find research papers that match specific criteria.
    1. Use the 'search_research_papers' tool to find information.
    2. Analyze the results.
    3. If you find a matching paper, present the Title, Year, and Citation count clearly.
    4. If you cannot find a match, try adjusting your search terms.
    5. When you are finished, reply with 'TERMINATE'.""",
    llm_config=llm_config,
)

# This connects your Python function to the agents so they can "see" and use it
autogen.agentchat.register_function(
    search_research_papers,
    caller=researcher,
    executor=user_proxy,
    name="search_research_papers",
    description="Searches for research papers based on topic, year, and citation count."
)

# "Find a research paper on [topic] that was published [year] and has [citations]"
task = (
    "Find a research paper on 'Multi-Agent Systems' "
    "that was published in 2023 "
    "and has at least 10 citations."
)

# Start the Agent
if __name__ == "__main__":
    print(f"Starting Task: {task}\n" + "-" * 50)

    # Check if 'coding' directory exists, create if not
    if not os.path.exists("coding"):
        os.makedirs("coding")

    user_proxy.initiate_chat(
        researcher,
        message=task
    )