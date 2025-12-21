from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(
        page_content = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.",
        metadata = {"source" : "doc1"}
    ),
    Document(
        page_content = "The Great Wall of China is a series of fortifications that stretch across northern China, built to protect against invasions and raids from various nomadic groups. The wall's construction began as early as the 7th century BC, with several dynasties contributing to its expansion and reinforcement over the centuries. The most well-known sections of the wall were built during the Ming Dynasty (1368–1644). Today, the Great Wall is a UNESCO World Heritage site and a popular tourist destination.",
        metadata = {"source" : "doc2"}
    ),
    Document(
        page_content = "Machu Picchu is an ancient Incan city located high in the Andes Mountains of Peru. Built in the 15th century and later abandoned, it is renowned for its sophisticated dry-stone construction that fuses huge blocks without the use of mortar. The site includes agricultural terraces, plazas, and temples, and is believed to have served as a royal estate or religious retreat. Rediscovered in 1911 by American historian Hiram Bingham, Machu Picchu is now a UNESCO World Heritage site and one of the most visited tourist attractions in South America.",
        metadata = {"source" : "doc3"}
    ),
    Document(
        page_content = "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor, USA. A gift from France to the United States, it was designed by French sculptor Frédéric Auguste Bartholdi and dedicated on October 28, 1886. The statue represents Libertas, the Roman goddess of freedom, and is a symbol of democracy and freedom. It stands at 305 feet (93 meters) from the ground to the tip of the torch and has become an iconic symbol of the United States.",
        metadata = {"source" : "doc4"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
vector_store = FAISS.from_documents(documents = docs, embedding = embedding_model)

# This basic retriever will retrieve larger chunks of documents
base_retriever = vector_store.as_retriever(search_kwargs = {"k" : 2})

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# Content will be compressed using the specified LLM
compressor = LLMChainExtractor.from_llm(llm)

# This retriever will fetch documents using the base retriever and then compress them using the compressor
compression_retriever = ContextualCompressionRetriever(
    base_retriever = base_retriever,
    compressor = compressor
)

query = "Tell me about some famous landmarks and their history."
results = compression_retriever.invoke(query)

for i, docs in enumerate(results):
    print(f"--Results {i + 1}--")
    print(docs.page_content)
