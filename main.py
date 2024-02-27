import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Makes secrets available to the app
load_dotenv()

def get_vectorstore_from_pdf(pdf):
    # Get pdf into document format
    loader = PyPDFLoader(pdf)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vectorstore from the chunks
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    return vector_store 

def get_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation" )
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain) 

    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
        })
    
    return response["answer"]

# App configuration
st.set_page_config(page_title="Chat with PDF")
st.title("Chat with PDF")

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF",type=["pdf"])

if uploaded_file is None:
    st.info("Please upload a file.")

else:
    st.info("PDF file uploaded successfully!")

    # The function created: get_vectorstore_from_pdf uses the langchain function PyPDFLoader which expects as an argument a path.
    # The following lines of code take the uploaded file by the user and get its path, to later pass it to the get_vectorstore_from_pdf function
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name

    # Session state
    if "chat_history" not in st.session_state: # Any time an event happens, streamlit re-runs the entire code. This way we make it persistent
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can i help you?")
    ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_pdf(temp_file)
    
    # User input (the chat becomes available once the user uploades a PDF)
    user_query = st.chat_input("Type your message here")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation, looping through chat_history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)