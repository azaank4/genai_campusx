from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

chat_history = [
    SystemMessage(content = 'You are a friendly chatbot that helps users with their questions. Type "exit" to end the conversation.')
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content = response.content))
    print(f"Bot: {response.content}")

print(chat_history)