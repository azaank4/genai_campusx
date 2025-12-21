from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
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

# Initialize Chroma vector store
vector_store = Chroma.from_documents(
    documents = docs,
    embedding = model,
    persist_directory = 'chroma_db',
    collection_name = 'sample'
)

retriever = vector_store.as_retriever(search_kwargs = {"k" : 2})

query = "Who is MS Dhoni?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n-- Result {i + 1}--")
    print(doc.page_content)


