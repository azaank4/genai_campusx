from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

st.header("Research Tool")

prompt = st.text_input("Enter your prompt")

chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


if st.button('Submit'):
    result = chat_model.invoke(prompt)
    st.markdown(result.content)