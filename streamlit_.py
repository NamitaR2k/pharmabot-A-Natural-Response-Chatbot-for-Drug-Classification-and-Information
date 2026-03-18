
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GooglePalm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
#from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import os
import tempfile

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline




def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your medicine", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create llm
    
             
        prompt_template = """
            Given the following context and a question, generate an answer based on the context only. In the answer try to provide as much text as possible from 'response' section in the source document without making. if the answer is not found in the context kindly state "This drug is not in provided dataset, so It is a New Drug". Don't try to make up an answer.
            
            CONTEXT: {context}
            QUESTION: {question}
        """
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"temperature": 0, "max_length": 200})

            
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                     retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                     memory=memory)
        return chain
    

def main():
    # Initialize session state
    initialize_session_state()
    st.title("PharmaBot\n An Interactive chatbot for Drug Classification")    # Initialize Streamlit
    vectordb_path1 = r"C:\index.faiss"
    file_path1 = r"C:\schedule and non schedule data.csv"
    
        
    loader = DirectoryLoader(file_path=file_path1)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(documents)

        # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(vectordb_path1, embeddings, allow_dangerous_deserialization=True)

        # Create the chain object
    chain = create_conversational_chain(vector_store)

    
    display_chat_history(chain)
    

if __name__ == "__main__":
    main()


