from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

template_1 = PromptTemplate(
    template = 'Generate a detailed report on the topic: {topic}',
    input_variables = ['topic']
)

template_2 = PromptTemplate(
    template = 'Generate a 5 pointers summary based on the following text \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser

result = chain.invoke({'topic' : 'Bugatti'})
print(result)

chain.get_graph().print_ascii()