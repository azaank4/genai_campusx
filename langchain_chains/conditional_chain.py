from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda   # RunnableBranch is used for creating conditional chains
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()


# Models
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser_1 = StrOutputParser()

# To get only 'positive' or 'negative' in the output
class feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description = 'Give the sentiment of the feedback')

parser_2 = PydanticOutputParser(pydantic_object = feedback)

prompt_1 = PromptTemplate(
    template = "Classify the following feedback text into positive or negative \n {text} \n {format_instruction}",
    input_variables = ['text'],
    partial_variables = {'format_instruction' : parser_2.get_format_instructions()}
)

classifier_chain = prompt_1 | model | parser_2

prompt_2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {text}',
    input_variables = ['text']
)

prompt_3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {text}',
    input_variables = ['text']
)

# If the feedback is positive
positive_chain = prompt_2 | model | parser_1
# If the feedback is negative
negative_chain = prompt_3 | model | parser_1

branch_chain = RunnableBranch(
    # here, we pass two tuples. Each tuple has two values. (condition and which chain to execute if the condition is true). If no condition is true, we define a default chain
    (lambda x : x.sentiment == 'positive', positive_chain),
    (lambda x : x.sentiment == 'negative', negative_chain),
    RunnableLambda(lambda x : 'could not find sentiment')
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({'text' : "This is a terrible smart phone. The battery life is very less"})
print(result)

final_chain.get_graph().print_ascii()