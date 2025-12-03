from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
load_dotenv()

embedding = InferenceClient(
    provider = 'hf-inference',
    api_key = os.getenv("HF_TOKEN")
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Who is Virat Kohli?"

doc_embeddings = embedding.feature_extraction(
    documents,
    model = 'google/embeddinggemma-300m'
)

query_embedding = embedding.feature_extraction(
    query,
    model = 'google/embeddinggemma-300m'
)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]
# enumerate is used for creating index value pairs

print(query)
print(documents[index])
print(f"Similarity score : {score}")
