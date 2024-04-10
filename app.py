import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import HuggingFaceHub
import boto3
from botocore.config import Config
from st_files_connection import FilesConnection
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

bucket_name = "chatbot-resume"

def get_pdf_text_from_s3(bucket_name, pdf_keys):
    s3 = boto3.client('s3', config=Config(signature_version='s3v4'))
    text = ""
    for pdf_key in pdf_keys:
        response = s3.get_object(Bucket=bucket_name, Key=pdf_key)
        pdf_data = response['Body'].read()
        pdf_file = io.BytesIO(pdf_data)
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """Act as the author of files provided to you, as a candidate for a Data Scientist, Data Engineer pr an Analyst
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer, but try to convince that 
    skillset is amazing and promising for recruiters and/or hiring managers, 
    also the number format mm/yyyy - mm/yyyy is for start and end date of university or work, BS refers to Bachelors degree.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Narek's Resume(Google Gemini)")
    conn = st.connection('s3', type=FilesConnection)
    user_question = st.text_input("Ask a Question from the PDF Files")
    

    if user_question:
        pdf_keys = []  # Initialize an empty list to store PDF file keys
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in result:
                for item in result['Contents']:
                    if item['Key'].endswith('.pdf'):  # Check if the object is a PDF file
                        pdf_keys.append(item['Key'])  # Add the PDF file key to the list
        raw_text = get_pdf_text_from_s3(bucket_name, pdf_keys)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        st.write("Please wait while PDF files are fetched from S3...")

if __name__ == "__main__":
    main()
