# This is without using String Output Parser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

model = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V3.2',
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

prompt_1 = template_1.invoke({"topic": "black hole"})

detailed_report = llm.invoke(prompt_1)
print("Detailed Report:\n", detailed_report.content)

prompt_2 = template_2.invoke({"text": detailed_report.content})

summary = llm.invoke(prompt_2)
print("Summary:\n", summary.content)