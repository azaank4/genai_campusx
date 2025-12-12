# Our goal: Provide a topic -> generate linkedin and twitter post
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

# Models
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt_linkedin = PromptTemplate(
    template = 'Generate content for a linkedin post for topic: {topic}',
    input_variables = ['topic']
)

prompt_twitter = PromptTemplate(
    template = 'Generate content for a twitter, under 200 words post for topic: {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'linkedin' : RunnableSequence(prompt_linkedin, model, parser),
    'twitter' : RunnableSequence(prompt_twitter, model, parser)
})

result = parallel_chain.invoke({'topic' : 'LangChain history'})
print(result)

parallel_chain.get_graph().print_ascii()