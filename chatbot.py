import os
from dotenv import load_dotenv

# Charge le fichier .env si présent
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Récupère la clé API dans la variable d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable d'environnement OPENAI_API_KEY n'est pas définie")

# Charge les 3 documents PDF
loader_1 = PyPDFLoader("Conditions_generales.pdf")
loader_2 = PyPDFLoader("FAQ_Assurtech.pdf")
loader_3 = PyPDFLoader("instructions_LLM.pdf")

# Charge tout le contenu
docs = loader_1.load() + loader_2.load() + loader_3.load()

# Découpe les documents en morceaux adaptés
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Initialise les embeddings OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Crée la base de données vectorielle FAISS
vectordb = FAISS.from_documents(chunks, embeddings)

# Initialise le LLM ChatOpenAI avec température nulle pour des réponses précises
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Crée la chaîne de récupération (retrieval QA) basée sur le vecteur et LLM
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())

print("Chatbot FAQ / CGV / Instructions LLM - Tapez 'exit' pour quitter.")

#while True:
#    query = input("Vous : ")
#    if query.lower() == "exit":
#        print("Fin du programme.")
#        break
#    response = qa.run(query)
#    print("Bot :", response)

def get_bot_response(query: str) -> str:
    response = qa.run(query)
    print("Bot :", response)
    return response
