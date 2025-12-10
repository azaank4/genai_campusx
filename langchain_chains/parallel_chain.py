from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel   # Used for creating parallel chains
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'openai/gpt-oss-20b',
    huggingfacehub_api_token = os.getenv("HF_TOKEN")
)

# Models
gemini_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
hf_model = ChatHuggingFace(llm = llm)

prompt_1 = PromptTemplate(
    template = 'Generate a short and simple notes from the following text \n {text}',
    input_variables = ['text']
)

prompt_2 = PromptTemplate(
    template = 'Generate 5 short question answers from the following text \n {text}',
    input_variables = ['text']
)

prompt_3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes ->{notes} and quiz -> {quiz}',
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt_1 | gemini_model | parser,
    'quiz' : prompt_2 | hf_model | parser
})

merge_chain = prompt_3 | gemini_model | parser

chain = parallel_chain | merge_chain

text = """**Lamborghini Automobili: A Comprehensive Overview**

**Introduction:**
Lamborghini Automobili is a renowned Italian luxury car manufacturer, celebrated for its high-performance vehicles and avant-garde designs. Founded by Ferruccio Lamborghini, the brand has evolved from humble beginnings in tractor manufacturing to become a icon in the automotive world. This overview delves into the history, notable models, design philosophy, technological innovations, racing endeavors, and future direction of Lamborghini.

**History:**
- **Founder and Origins:** Ferruccio Lamborghini, born under the Taurus zodiac sign, which inspired the bull logo, started with Lamborghini Trattori, producing tractors. His passion for sports cars led him to establish Automobili Lamborghini in 1963.
- **Early Years:** The company debuted with the 350 GTV in 1963, followed by the 350 GT in 1964. The 1966 Miura, designed by Marcello Gandini, is often credited as the first supercar.
- **Challenges and Revival:** Financial difficulties in the 1970s and 1980s led to changes in ownership. In 1998, Volkswagen Group acquired Lamborghini, stabilizing the brand and fostering innovation.

**Notable Models:**
- **Miura (1966):** Revolutionized the automotive industry with its mid-engine layout, epitomizing the term "supercar."
- **Countach (1974):** Introduced scissor doors and a wedge-shaped design, becoming an icon of the 80s.
- **Diablo (1990):** Marked Lamborghini's return to prominence with all-wheel drive and a V12 engine.
- **Murciélago (2001):** Featured a scissor-door design and was the first to use a V12 engine with all-wheel drive.
- **Aventador (2011):** A V12 flagship with carbon fiber chassis, symbolizing cutting-edge technology.
- **Sián (2019):** Combines a V12 engine with hybrid technology, signaling Lamborghini's move towards electrification.

**Design Philosophy:**
- **Aesthetic Aggression:** Known for angular, futuristic designs with sharp lines, emphasizing power and speed. The scissor door, a Lamborghini hallmark, adds to the car's dramatic presence.
- **Influence and Collaboration:** Early models were designed by Bertone's Marcello Gandini, while current designs are handled by an in-house team, blending Italian style with innovation.

**Technology and Innovation:**
- **Advanced Materials:** Use of carbon fiber and lightweight materials for strength and efficiency.
- **Powertrains:** Iconic V12 engines, with recent integration of hybrid systems in models like the Sián.
- **All-Wheel Drive:** Pioneered in the Diablo, enhancing traction and performance.

**Racing and Motorsports:**
- **GT Championships:** Participation in series like the FIA GT Championship with models such as the Murciélago R-GT.
- **Lamborghini Super Trofeo:** A one-make series showcasing the Gallardo and Huracán.
- **Squadra Corse:** The racing division dedicated to motorsport activities and special projects.

**Current Direction and Future:**
- **Electrification Strategy:** Commitment to hybrid and electric vehicles, with plans to electrify the entire lineup by 2030.
- **Concept Cars:** The V12 Vision Gran Turismo and LB744 preview future design and technology directions.
- **Market Presence:** Exclusive dealerships globally, catering to a niche market with limited production runs.

**Cultural Impact:**
- **Pop Culture:** Featured in films, video games, and music, solidifying Lamborghini's status as a luxury symbol.
- **Brand Differentiation:** Competitors like Ferrari and Porsche are rivaled by Lamborghini's extreme, avant-garde approach.

**Conclusion:**
Lamborghini's journey from tractor manufacturer to luxury car icon is a testament to innovation and passion. With a commitment to electrification and cutting-edge design, Lamborghini continues to shape the automotive landscape, maintaining its reputation as a symbol of exclusivity and high performance.
"""

result = chain.invoke({'text' : text})
print(result)

chain.get_graph().print_ascii()