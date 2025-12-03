from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# HUGGINGFACE Endpoint is used for accessing models via API instead of downloading
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)