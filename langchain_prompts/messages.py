from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

messages = [
    SystemMessage(content = "You are a helpful assistant that provides concise and accurate information."),
    HumanMessage(content = "Tell me about LangChain.")
]

response = model.invoke(messages)
messages.append(AIMessage(content = response.content))
print(response.content)
print(messages)