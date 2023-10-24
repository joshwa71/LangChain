import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == '__main__':
    print('Hello world!')
    pdf_path = "./data/2210.03629.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator='\n')
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local('./data/vector_store')

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vector_store.as_retriever())
    res = qa.run("Give me the gist of the paper")
    print(res)




    