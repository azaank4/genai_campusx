from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox(
    "Select a research paper:",
    ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Decision models beat GANs on image synthesis"]
)

style_input = st.selectbox(
    "Select the style of summary:",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select explanation length: ",
    ["Short (1-2 paragraphs)", "Medium (3-4 paragraphs)", "Long (detailed explanation)"]
)

chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


# Prompt Template
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:Explanation Style: {style_input}  Explanation Length: {length_input}  1. Mathematical Details:     - Include relevant mathematical equations if present in the paper.     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  2. Analogies:     - Use relatable analogies to simplify complex ideas.  If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  Ensure the summary is clear, accurate, and aligned with the provided style and length
"""
)

# Filling the placeholders
prompt = template.invoke({
    "paper_input" : paper_input,
    "style_input" : style_input,
    "length_input" : length_input
})

if st.button('Submit'):
    # st.text("Hello, just testing")
    result = chat_model.invoke(prompt)
    st.write(result.content)