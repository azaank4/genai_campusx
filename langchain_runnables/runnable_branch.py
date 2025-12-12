# Our goal: Provide a topic -> generate a report -> if >500 words, again summarize ; else print
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
load_dotenv()

# Models
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# parser
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a detailed report about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the content: {content}',
    input_variables = ['content']
)

report_gen_chain = RunnableSequence(prompt1, model, parser)
summarizer_chain = RunnableSequence(prompt2, model, parser)

# Conditional runnable
branch_chain = RunnableBranch(
    # (condition, runnable),
    # (default condition, runnable)
    (lambda x : len(str(x).split()) > 500, summarizer_chain),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)
result = final_chain.invoke({'topic' : 'Kawasaki bikes'})
print(result)

final_chain.get_graph().print_ascii()