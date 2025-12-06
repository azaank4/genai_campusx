from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

# Define your Pydantic schema
class Review(BaseModel):
    key_points: Annotated[list[str], Field(description="A list of key points mentioned in the review")]
    summary: Annotated[str, Field(description="A brief summary of the review")]
    sentiment: Annotated[str, Field(description="The overall sentiment of the review, e.g., positive, negative, neutral")]
    pros: Annotated[list[str], Field(description="A list of pros mentioned in the review")]
    cons: Annotated[Optional[list[str]], Field(description="A list of cons mentioned in the review, if any")]
    name: Annotated[str, Field(description="Name of the reviewer")]

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# Attach structured output with JSON schema
structured_llm = llm.with_structured_output(Review)

# Invoke with your prompt
result = structured_llm.invoke("Generate a detailed review about Apple iPhone X")

# Result is already parsed as a dict/JSON
print(result)