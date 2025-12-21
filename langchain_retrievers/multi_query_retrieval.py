from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# This is an older version of LangChain, hence we import using langchain_classic. 
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

# Define embedding model
model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create sample data
doc1 = Document(
    page_content = "Virat Kohli is an Indian cricketer and former captain of the Indian national team. Known for his exceptional batting skills and leadership, he has been a key player in India's cricketing success.",
    metadata = {"team" : "Royal Challengers Bangalore", "role": "Batsman"}
)

doc2 = Document(
    page_content = "Rohit Sharma is an Indian cricketer and current captain of the Indian national cricket team. He is known for his opening batting and has led India to multiple ICC tournament victories.",
    metadata = {"team" : "Mumbai Indians", "role": "Batsman"}
)

doc3 = Document(
    page_content = "Jasprit Bumrah is an Indian fast bowler known for his unique bowling action and death bowling expertise. He has been instrumental in India's recent World Cup campaigns and IPL success.",
    metadata = {"team" : "Mumbai Indians", "role": "Bowler"}
)

doc4 = Document(
    page_content = "Ravindra Jadeja is an Indian all-rounder who excels in both batting and left-arm orthodox spin bowling. He has been a crucial player in India's Test and ODI teams.",
    metadata = {"team" : "Chennai Super Kings", "role": "All-rounder"}
)

doc5 = Document(
    page_content = "MS Dhoni is a legendary Indian cricketer and former captain known for his finishing skills and wicket-keeping abilities. He led India to multiple World Cup victories.",
    metadata = {"team" : "Chennai Super Kings", "role": "Wicket-keeper"}
)

# Combine documents into a list
docs = [doc1, doc2, doc3, doc4, doc5]

# creating vector store using FAISS
vector_store = FAISS.from_documents(documents = docs, embedding = model)

# Using simple similarity retriever
similarity_retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 2})

# Using MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = {"k" : 2}),
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
)

# Query
query = "Who are the captains of Indian cricket team?"

results_1 = similarity_retriever.invoke(query)
results_2 = multiquery_retriever.invoke(query)

# Printing Similarity Retriever Results
for i, docs in enumerate(results_1):
    print(f"Similarity Retriever results {i + 1}")
    print(docs.page_content)

print("*" * 100)

# Printing MultiQuery Retriever Results
for i, docs in enumerate(results_2):
    print(f"MultiQuery Retriever results {i + 1}")
    print(docs.page_content)