from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

# video transcript
video_id = "jjeLzr1JR4o"
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    # the tutorial uses .get_transcript but it doesnt work anymore, hence in the latest update we use .fetch
    transcript_text = " ".join(chunk.text for chunk in transcript_list.snippets)  # Extract text from all snippets
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

# print(transcript_text)  # This will print just the text content

# Text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.create_documents([transcript_text])
# print(len(chunks))
# print(chunks[0])

# Create vector store
client = InferenceClient(
    provider = "hf-inference",
    api_key = os.getenv("HF_TOKEN")
)

embedded_chunks = client.feature_extraction(
    chunks.page_content,

)

vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embedding
)

# Retrieval
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 3}
)

query = "What is Harvey Specter's closing argument when dealing with Gerald and his bad faith actions regarding the Cooper deal?"
results = retriever.invoke(query)
print(results)