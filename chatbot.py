
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 🔐 Ta clé OpenAI ici
os.environ["OPENAI_API_KEY"] = "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"

# 1. Charger les PDF
loader_1 = PyPDFLoader("Conditions_generales.pdf")
loader_2 = PyPDFLoader("FAQ_Assurtech.pdf")
docs = loader_1.load() + loader_2.load()

# 2. Découpage des textes
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embedding + vectorisation
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(chunks, embeddings)

# 4. Création du chatbot avec récupération
retriever = vectordb.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 5. Interaction utilisateur
while True:
    try:
        query = input("\nPosez votre question (ou 'exit') : ")
        if query.lower() == "exit":
            print("Fin du programme.")
            break
        result = qa_chain.run(query)
        print("\nRéponse :", result)
    except KeyboardInterrupt:
        print("\nFin du programme par Ctrl+C.")
        break











