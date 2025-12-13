from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('heart-disease.csv')

docs = loader.load()
print(len(docs))    # Prints the number of rows
print(docs[0].page_content)