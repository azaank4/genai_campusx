# This code will run properly only if the model supports structured outputs
# This code wont work with gemini-2.5-flash as it does not support structured outputs yet
# We can use openrouter to check which models support structured outputs
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional
# Annotated is used to add some extra info to the types in TypedDict if needed
# Optional can be used to make a field optional in TypedDict
from dotenv import load_dotenv
import os
load_dotenv()
    
llm = HuggingFaceEndpoint(
    repo_id = 'moonshotai/Kimi-K2-Thinking',
    huggingfacehub_api_token = os.getenv('HF_TOKEN')
)

model = ChatHuggingFace(llm = llm)

# schema
class Review(TypedDict):
    key_points : Annotated[list[str], "A list of key points mentioned in the review"]
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]
    pros : Annotated[list[str], "A list of pros mentioned in the review"]
    cons : Annotated[Optional[list[str]], "A list of cons mentioned in the review, if any"]
    name : Annotated[str, "Name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh""")

print(result, "\n")