import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def get_pdf_text(file_path):
    """Extrait le texte d'un fichier PDF."""
    text = ""
    try:
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def get_text_chunks(text):
    """Divise le texte en chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Crée un index vectoriel (vectorstore) à partir des chunks de texte."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    # Chemin du fichier PDF à traiter
    file_path = 'dataset/Guide_Auteurs_Etudiants_VeriWrite.pdf'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Extraire le texte du PDF
    raw_text = get_pdf_text(file_path)
    if raw_text.strip():
        # Diviser le texte en chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Créer le vectorstore
        vectorstore = get_vectorstore(text_chunks)
        
        # Sauvegarder l'index FAISS localement
        vectorstore.save_local("vectorstore_index")
        print("Vectorstore index has been saved successfully.")
    else:
        print("No text found in the PDF file.")

if __name__ == "__main__":
    main()
