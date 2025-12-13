from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

loader = TextLoader('transcript.txt', encoding = 'utf-8')
docs = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Explain the following transcript in short: {content}',
    input_variables = ['content']
)

chain = prompt | model | parser
result = chain.invoke({'content' : docs[0].page_content})
print(result)