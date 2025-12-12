# Our goal: Generate a joke -> get joke in intermediate phase using Passthrough -> get word count from the joke using Runnable Lambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

# word counting function
def word_counter(text):
    return len(text.split())

# Models
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'tell me a joke about {topic}',
    input_variables = ['topic']
)

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'Artificial Intelligence'})

final_result = f"The joke : \n {result['joke']} \n Word count : \n {result['word_count']}"
print(final_result)

final_chain.get_graph().print_ascii()