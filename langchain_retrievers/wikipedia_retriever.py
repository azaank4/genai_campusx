from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever
retriever = WikipediaRetriever(top_k_results = 3, lang = "en")

# Define the query
query = "Who was Albert Einstein?"

docs = retriever.invoke(query) 
print(docs)