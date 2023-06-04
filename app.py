#!/usr/bin/env python
# coding: utf-8

import os 
import openai
import streamlit as st
from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI

openai.api_type = "azure"
openai.api_version = "2023-05-15"

openai_api_key = os.environ.get('OPENAI_API_KEY')
openai_api_base = os.environ.get('OPENAI_API_BASE')


#@st.cache(allow_output_mutation=True)
@st.cache_resource
def process_pdf(file_path):
    doc_reader = PdfReader(file_path)
    
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
            
        # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Normalize the text

    # Generate embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    docsearch = Chroma.from_texts(texts, embeddings)

    return docsearch

docsearch = process_pdf('./IMF.pdf')


llm = AzureOpenAI(deployment_name="text-davinci-003", temperature=0.8, max_tokens=300, model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
})

chain = load_qa_chain(llm, chain_type="stuff") #, chain_type="stuff"

st.markdown("<h1 style='text-align: center; color: #BFFF00;'>WORLD ECONOMIC OUTLOOK<br>IMF APRIL 2023 REPORT</h1>", unsafe_allow_html=True)
st.title('A Rocky Recovery')
query = st.text_input('Enter your query:')
if st.button('Submit'):
    docs = docsearch.similarity_search(query,k=4)
    response = chain.run(input_documents=docs, question=query) 
    st.write(response)



