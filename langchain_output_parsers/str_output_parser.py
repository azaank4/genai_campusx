from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

model = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    huggingfacehub_api_token = os.getenv('HF_TOKEN')
)

llm = ChatHuggingFace(llm = model)

# 1st prompt -> Detailed report on a topic
template_1 = PromptTemplate(
    template = "Provide a detailed report on the topic: {topic}",
    input_variables = ["topic"]
)

# 2nd prompt
template_2 = PromptTemplate(
    template = "Provide a concise, 5 line summary on the following text: {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain = template_1 | llm | parser | template_2 | llm | parser

result = chain.invoke({"topic" : "black hole"})
print("Final Summary:\n", result)