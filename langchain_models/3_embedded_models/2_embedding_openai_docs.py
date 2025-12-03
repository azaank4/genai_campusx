from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 32)

documents = [
    "India is a country in South Asia.",
    "The capital of India is Delhi.",
    "Mumbai is the financial capital of India.",
    "Bangalore is known as the Silicon Valley of India.",
    "The Taj Mahal is located in Agra, India.",

]

result = embedding.embed_documents(documents)
print(str(result))