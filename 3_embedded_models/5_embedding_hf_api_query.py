from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()

text = "Today is a sunny day and I will get some ice cream."

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.feature_extraction(
    text,
    model="google/embeddinggemma-300m",
)

print(str(result))