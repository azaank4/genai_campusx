from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Define your JSON schema
json_schema = {
    "title": "FactExtractor",
    "description": "Extract 3 facts about a topic",
    "type": "object",
    "properties": {
        "fact_1": {
            "type": "string",
            "description": "Fact 1 about the topic"
        },
        "fact_2": {
            "type": "string",
            "description": "Fact 2 about the topic"
        },
        "fact_3": {
            "type": "string",
            "description": "Fact 3 about the topic"
        }
    },
    "required": ["fact_1", "fact_2", "fact_3"]
}

# Initialize your model
model = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

llm = ChatHuggingFace(llm=model)

# Attach structured output with JSON schema
structured_llm = llm.with_structured_output(json_schema)

# Invoke with your prompt
result = structured_llm.invoke("Give 3 facts about black holes")

# Result is already parsed as a dict/JSON
print(result)