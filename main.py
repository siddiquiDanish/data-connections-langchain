# Build a sample vectorDB
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import os
f = open('C:\\Users\\dahmedsiddiqui\\Desktop\\OPEN_AI_KEY.txt')
os.environ['OPENAI_API_KEY'] = f.read()

# Takes in a question about the US Constitution and returns the most relevant  part of the constitution
def us_constitution_helper(question):

    # LOAD "some_data/US_Constitution in a Document object
    loader = TextLoader("data/US_Constitution.txt")
    documents = loader.load()

    # Split the document into chunks (you choose how and what size)
    # run :: pip install tiktoken - to install tiktoken
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)

    # EMBED THE Documents (now in chunks) to a persisted ChromaDB
    embedding_function = OpenAIEmbeddings()
    embedding_function = OpenAIEmbeddings()
    #run :: pip install chromadb -to install Chroma vector DB
    db = Chroma.from_documents(docs, embedding_function, persist_directory='./US_Constitution')
    db.persist()

    # Use ChatOpenAI and ContextualCompressionRetriever to return the most relevant part of the documents.

    # results = db.similarity_search("What is the 13th Amendment?")
    # print(results[0].page_content) # NEED TO COMPRESS THESE RESULTS!
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                           base_retriever=db.as_retriever())

    compressed_docs = compression_retriever.get_relevant_documents(question)

    return compressed_docs[0].page_content



que = input("Enter your question on US Constitution :")
print(us_constitution_helper(que))