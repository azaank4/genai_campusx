from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatAnthropic(model = 'claude-3.5-sonnet-24102022')

result = chat_model.invoke("Explain the theory of relativity in simple terms.")
print(result.content)