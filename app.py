import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
import os
from htmlTemplates import *


# Charger les variables d'environnement
load_dotenv()
st.set_page_config(page_title="Chat with multiple Resumes", page_icon="logo.jpg")

# Assurez-vous que votre clé API OpenAI est définie
openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "OPENAI_API_KEY is not set in the environment variables."

# Créer les embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Fonction pour créer la chaîne de conversation
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    return conversation_chain

# Fonction pour gérer l'entrée utilisateur
def handle_userinput(user_question, chat_container):
    if st.session_state.conversation is None:
        st.error(":red[Please process the documents first.]")
        return
    
    # Obtenir la réponse de la chaîne de conversation
    response = st.session_state.conversation.run({'question': user_question})

    # Vérifier le type de réponse et afficher pour le débogage
    st.write(response)  # Afficher la réponse dans Streamlit pour déboguer

    # S'assurer que la réponse est un dictionnaire et contient 'chat_history'
    if isinstance(response, dict) and 'chat_history' in response:
        st.session_state.chat_history = response['chat_history']
def main():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header(":blue[VeriWrite ChatBot]")

    # Conteneur pour l'historique des conversations
    chat_container = st.container()

    # Saisie de l'utilisateur
    user_question = st.text_input(":orange[Comment puis-je vous aider?]", key="user_input")
    
    # Mettre à jour le conteneur de chat avec l'historique lors de la réception d'une entrée
    if user_question:
        handle_userinput(user_question, chat_container)

  
    # Charger l'index vectoriel
    try:
        vectorstore = FAISS.load_local("vectorstore_index", embeddings, allow_dangerous_deserialization=True)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        #st.success("Vectorstore index has been loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        st.session_state.vectorstore = None

if __name__ == '__main__':
    main()
