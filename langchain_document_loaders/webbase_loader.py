from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

url = 'https://huggingface.co/openai/gpt-oss-20b'   # Model card of gpt oss from huggingface

loader = WebBaseLoader(url)
docs = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Based on the provided content, answer the question: {question} \n Content: {content}',
    input_variables = ['content', 'question']
)

chain = prompt | model | parser
result = chain.invoke({'content' : docs[0].page_content, 'question' : 'What is the provided webpage about?'})
print(result)