import os
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Serveur FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Récupère la clé API dans la variable d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable d'environnement OPENAI_API_KEY n'est pas définie")

# Charge les 3 documents PDF
loader_1 = PyPDFLoader("Conditions_generales.pdf")
loader_2 = PyPDFLoader("FAQ_Assurtech.pdf")
loader_3 = PyPDFLoader("instructions_LLM.pdf")

# Charge tout le contenu
docs = loader_3.load() + loader_1.load() + loader_2.load()
#print("docs", docs)

# Découpe les documents en morceaux adaptés
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Initialise les embeddings OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Crée la base de données vectorielle FAISS
vectordb = FAISS.from_documents(chunks, embeddings)

# Initialise le LLM ChatOpenAI avec température nulle pour des réponses précises
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

template = """
Tu es un chatbot spécialisé dans les assurances animaux.
Tu ne réponds qu’aux questions en lien avec les documents fournis.
Si la question ne concerne pas les assurances pour animaux, tu réponds exactement :
"Je suis spécialisé dans les assurances animaux. Merci de reformulez votre question."
Ta réponse doit être claire, synthétique, et s’appuyer sur les documents si nécessaire.
Tu détectes automatiquement la langue du client et tu réponds dans cette même langue (y compris pour les formules de politesse).
Commence ta réponse par la salutation appropriée à la langue détectée (ex: "Bonjour, ", "Hello, "),puis laisse explicitement deux balises html <br></br> (<br></br> <br></br>).
Après ta réponse, mets de nouveau explicitement deux balises html <br></br> (<br></br> <br></br>) , puis ajoute la formule de politesse appropriée (ex: "Cordialement, ", "Best regards, ").
Après la formule de politesse, laisse mets de nouveau deux balises html <br></br> (<br></br> <br></br>), puis ajoute "L’équipe Pet Assurance" (ou son équivalent dans la langue détectée).

Contexte: {context}
Question: {question}
Réponse:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Crée la chaîne de récupération (retrieval QA) basée sur le vecteur et LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

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
