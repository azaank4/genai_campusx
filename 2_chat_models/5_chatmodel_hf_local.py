from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# HuggingFacePipeline is used for local model inference
from dotenv import load_dotenv
import os
load_dotenv()

# Storing model in specific path
os.environ['HF_HOME'] = "D:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    pipeline_kwargs = dict(
        temperature = 0.6,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the capital of India")
print(result)