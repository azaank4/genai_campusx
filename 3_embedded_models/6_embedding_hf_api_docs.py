from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()

client = InferenceClient(
    provider = 'hf-inference',
    api_key = os.getenv('HF_TOKEN')
)

docs = [
    "Delhi is the capital of India.",
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "Mount Everest is the highest mountain in the world.",
    "The Amazon rainforest is the largest tropical rainforest."
]

result = client.feature_extraction(
    docs, 
    model = 'google/embeddinggemma-300m'
)

print(str(result))