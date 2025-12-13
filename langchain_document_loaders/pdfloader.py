from langchain_pymupdf4llm import PyMuPDF4LLMLoader

loader = PyMuPDF4LLMLoader('sample.pdf')

docs = loader.load()
print(docs)
print(len(docs))    # Prints the number of pages
print(docs[0].page_content) # Prints the content from first page only
print(docs[1].metadata) # Prints the metadata only for the second page