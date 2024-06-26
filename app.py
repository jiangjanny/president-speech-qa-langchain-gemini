from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

import os

from configparser import ConfigParser

# Set up config parser
config = ConfigParser()
config.read("config.ini")

os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

loader = TextLoader("state_of_the_union.txt", encoding = 'UTF-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson?"
results = db.similarity_search_with_score(query, 1)
print(results[0][0].page_content)

# Retrieve the content of the most similar document

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}"""
)

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

# query = "What did the president say about Ketanji Brown Jackson?"
# query = "How much we are giving to the Ukraine?"
query = input("Enter your question: ")
results = db.similarity_search_with_score(query, 1)
print("Retrieved related content :")
print(results[0][0].page_content)
print("====================================================")

llm_result = document_chain.invoke(
    {
        "input": query,
        "context": [results[0][0]],
    }
)

print("Question: ", query)
print("LLM Answer: ", llm_result)