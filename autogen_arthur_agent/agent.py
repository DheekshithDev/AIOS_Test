import os
from autogen import AssistantAgent, UserProxyAgent

# llm_config = {
#     "model": "claude-3-opus",                  
#     "api_key": "",
# }

local_llm_config = {
  "config_list": [{

    "model": "llama3",  # Use your downloaded model name
    "base_url": "http://localhost:11434/v1",  # Ollama's API endpoint
    "api_key": "ollama"  # Authentication (required but unused)

  }],
  "cache_seed": None  # Fresh responses every time
}

assistant = AssistantAgent(
    name="assistant",
    llm_config=local_llm_config,
    system_message="Be concise. End with 'TERMINATE' when done."
)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER", 
    code_execution_config=False
)

response = user.initiate_chat(
    assistant,
    message="Give me three creative names for a cat cafe. Keep it short."
)

print(response.last_message()["content"])

