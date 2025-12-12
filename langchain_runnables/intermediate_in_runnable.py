from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
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

# Runnable
sequence_chain = (
    prompt1 |
    model |
    parser |
    RunnableParallel({
        'joke' : RunnablePassthrough(),
        'explanation' : prompt2 | model | parser
    })
)
# This is exactly same as above. It's just two different syntaxes

result = sequence_chain.invoke({'topic' : 'Artificial Intelligence'})
print(f"Joke : \n{result['joke']}")
print("\n")
print(f"Explanation: \n{result['explanation']}")

sequence_chain.get_graph().print_ascii()