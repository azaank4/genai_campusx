from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant expert in {domain} domain"),
    ("human", "Explain me in simple terms, what is {topic}?")
])

prompt = chat_template.invoke({
    'domain' : 'finance',
    'topic' : 'quant finance'
})