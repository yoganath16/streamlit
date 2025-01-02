# import sys
# sys.path.append(r'C:\Users\Yoganath\Downloads\pandasai')
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
#from pandasai.callbacks import BaseCallback
from pandasai.responses.response_parser import ResponseParser
from pandasai import Agent
from dotenv import load_dotenv


load_dotenv()

# API_KEY = os.environ['OPEN_API_KEY']
API_KEY = st.secrets['OPEN_API_KEY']

# class StreamlitCallback(BaseCallback):
#     def __init__(self, container) -> None:
#         """Initialize callback handler."""
#         self.container = container

#     def on_code(self, response: str):
#         self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

# # storing the API Token in Open AI environment 
# # replace "YOUR_API_KEY" with your generated API key 
# llm = OpenAI(api_token='YOUR_API_KEY') 
# #initializing an instance of Pandas AI with openAI environment  
# pandas_ai = PandasAI(llm, verbose=True, conversational=False)


llm = OpenAI(api_token=API_KEY)




st.title("speak with your data")

input_data = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if input_data is not None:
    df = pd.read_csv(input_data)
    st.write(df.head(3))

    prompt = st.text_area("Enter your question here")

    container = st.container()

    if st.button("Generate"):

        query_engine = SmartDataframe(df, config={"llm": llm,"response_parser": StreamlitResponse})


        answer = query_engine.chat(prompt)
        with st.spinner("I'm thinking..."):
            st.write(answer)

              


