import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import OpenAI
import pinecone
from langchain.chains import RetrievalQA

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")

if __name__ == "__main__":
    print("Hello Vector Store")
    loader = TextLoader(
        r"C:\Users\joshu\Documents\ArtificialIntelligence\Courses\LangChain\Embeddings\mediumblogs\mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="langchain-test")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    query = "What is a vector DB?"
    result = qa({"query": query})
    print(result)
