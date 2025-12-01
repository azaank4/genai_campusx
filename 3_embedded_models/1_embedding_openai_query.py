from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 32)
# The 'dimensions' parameter refers to the size of the embedding vector produced by the model.
# A higher dimensional embedding can capture more nuanced relationships in the data,
# but it may also require more computational resources and can lead to overfitting in some cases

result = embedding.embed_query("Delhi is the Capital of India")
print(str(result))