from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

model = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    huggingfacehub_api_token = os.getenv("HF_TOKEN")
)

llm = ChatHuggingFace(llm = model)

parser = JsonOutputParser()

template_1 = PromptTemplate(
    template = "Give me the name, age and city of a fictional character. \n {format_instruction}, ",
    input_variables = [],
    partial_variables = {"format_instruction" : parser.get_format_instructions()}
)
# Partial variable is used to inject format instructions into the prompt before the runtime
# Using parser.get_format_instructions() gives the necessary instructions for the LLM to output in the desired structure. Here, we want a JSON output.

chain = template_1 | llm | parser

result = chain.invoke({})
print(result)