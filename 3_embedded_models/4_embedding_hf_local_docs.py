from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India.",
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "Mount Everest is the highest mountain in the world.",
    "The Amazon rainforest is the largest tropical rainforest."
]

vectors = embedding.embed_documents(documents)

print(str(vectors))