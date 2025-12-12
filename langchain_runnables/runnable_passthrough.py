# Our goal: Printing the joke which is generated in an intermediate step
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
load_dotenv()


# Models
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# Prompt Template
prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the joke: \n {joke}',
    input_variables = ['joke']
)

# Parser
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'Artificial Intelligence'})
print(result)

final_chain.get_graph().print_ascii()