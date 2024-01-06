#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:56:20 2023

@author: avi_patel
"""

import google.generativeai as genai
from streamlit_option_menu import option_menu  
import os, openai, base64, requests, json, streamlit as st, PIL.Image, textwrap, chromadb
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def select_llm():
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo-1106",
                                   "gpt-4-1106-preview",
                                   "gemini-pro"),)
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    return model_name, temperature


def getgptresponse(client, model, temperature, message, streaming):
    try:
        response = client.chat.completions.create(model=model, messages=message, temperature=temperature, stream=streaming)

        output = response.choices[0].message.content
        tokens = response.usage.total_tokens
        yield output, tokens

    except Exception as e:
        print(e)
        yield ""
    

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""Use the following pieces of information to answer the user's question.  
   If you don't know the answer, just say that you don't know the answer, don't make up an answer.  Think step by step.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

def setup_documents(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs


def gpt_response(client, model, temperature, streaming, message):
        try:
            response = client.chat.completions.create(model=model,
                                                    messages=message,
                                                    temperature=temperature,
                                                    stream=streaming)

            if streaming:
                for event in response:
                    event_text = event.choices[0].delta.content
                    if event_text is not None:
                        yield event_text
            else:
                output = response.choices[0].message.content
                tokens = response.usage.total_tokens
                yield output, tokens

        except Exception as e:
            print(e)
            yield ""


def main():
    
    page_title="Compare Foundation Models"
    page_icon=":boxing_glove:"
    layout="wide"
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    st.title(page_icon + "  " + page_title + "  " + page_icon)
    st.sidebar.title("Options")
    selected = option_menu(
        menu_title="Three of the different ways to leverage a foundation model!",
        options=["Interact Directly", "Question & Answer", "Image Recognition"],
        orientation="horizontal"
        )
    
    model, temperature = select_llm()
    streaming=False
    
    if selected == "Interact Directly":
        query = st.text_input("Enter your question/prompt in the box below and hit return")
        if query:
            
            if model == "gemini-pro":
                genai.configure(api_key=GOOGLE_API_KEY)
                client = genai.GenerativeModel(model_name=model)
                response = client.generate_content(query, generation_config=genai.types.GenerationConfig(temperature=1.0))
                st.markdown(response.text)
            elif model == "gpt-3.5-turbo-1106":
                client = OpenAI(api_key=OPENAI_API_KEY)
                message=[]
                message.append({"role": "user", "content": f"{query}"})
                for result in getgptresponse(client, model, temperature, message, streaming):
                    output = result[0]
                    st.write(output)
            elif model == "gpt-4-1106-preview":
                client = OpenAI(api_key=OPENAI_API_KEY)
                message2=[]
                message2.append({"role": "user", "content": f"{query}"})
                for result in getgptresponse(client, model, temperature, message2, streaming):
                    output2 = result[0]
                    st.markdown(output2)
            else:
                pass
            
    if selected == "Question & Answer":
        uploaded_file = st.file_uploader("Upload your PDF document")
        if uploaded_file:
            texts = setup_documents(uploaded_file.name)
            
            if model == "gemini-pro":
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                db = Chroma.from_documents(texts, embeddings)
                retriever = db.as_retriever(search_kwargs={"score_threshold": 0.5, "k": 8})
                question = st.text_input("Enter your quesiton/prompt below")
                if question:
                    docs = retriever.get_relevant_documents(question)
                    yz=[]
                    for i in range(len(docs)):
                        yz += docs[i].page_content
                    yzf = ''.join(yz)    
                    prompt = make_prompt(question, yzf)
                    model = genai.GenerativeModel('gemini-pro')
                    answer = model.generate_content(prompt)
                    st.markdown(answer.text)
            elif model == "gpt-3.5-turbo-1106":
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                db = Chroma.from_documents(texts, embeddings)
                retriever = db.as_retriever(search_kwargs={"score_threshold": 0.5, "k": 8})
                question = st.text_input("Enter your quesiton/prompt below")
                if question:
                    docs = retriever.get_relevant_documents(question)
                    yz=[]
                    for i in range(len(docs)):
                        yz += docs[i].page_content
                    yzf = ''.join(yz)
                    
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    model='gpt-3.5-turbo-1106'
                    temperature=temperature
                    streaming=streaming
                    prompt = make_prompt(question, yzf)
                    message=[]
                    message.append({"role": "user", "content": f"{prompt}"})
                    
                    for result in gpt_response(client, model, temperature, streaming, message):
                        st.markdown(result[0])
            elif model == "gpt-4-1106-preview":
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                db = Chroma.from_documents(texts, embeddings)
                retriever = db.as_retriever(search_kwargs={"score_threshold": 0.5, "k": 8})
                question = st.text_input("Enter your quesiton/prompt below")
                if question:
                    docs = retriever.get_relevant_documents(question)
                    yz=[]
                    for i in range(len(docs)):
                        yz += docs[i].page_content
                    yzf = ''.join(yz)
                    
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    model='gpt-4-1106-preview'
                    temperature=temperature
                    streaming=streaming
                    prompt = make_prompt(question, yzf)
                    message=[]
                    message.append({"role": "user", "content": f"{prompt}"})
                    
                    for result in gpt_response(client, model, temperature, streaming, message):
                        st.markdown(result[0])
            else:
                pass
              
    if selected == "Image Recognition":
        pic = st.text_input("Enter location of you image in the box below and hit return", "")
        if pic:
            query2 = st.text_input("Enter your question/prompt in the box below and hit return")
            if query2:
                
                if model == "gemini-pro":
                    img = PIL.Image.open(f"{pic}")
                    client = genai.GenerativeModel('gemini-pro-vision')
                    response2 = client.generate_content([f"{query2}", img], stream=False)
                    response2.resolve()
                    st.markdown(response2.text)
                elif model == "gpt-4-1106-preview":
                    image_path = f"{pic}"
                    base64_image = encode_image(image_path)
                    headers = {
                      "Content-Type": "application/json",
                      "Authorization": f"Bearer {OPENAI_API_KEY}"
                    } 
                    payload = {
                      "model": "gpt-4-vision-preview",
                      "messages": [
                        {
                          "role": "user",
                          "content": [
                            {
                              "type": "text",
                              "text": f"{query2}"
                            },
                            {
                              "type": "image_url",
                              "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                              }
                            }
                          ]
                        }
                      ]
                    }
                    response3 = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    final = response3.json()
                    st.markdown(final['choices'][0]['message']['content'])
                else:
                    pass
            
            
if __name__ == "__main__":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    main()