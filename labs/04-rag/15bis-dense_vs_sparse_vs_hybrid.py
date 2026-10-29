import os

from param import Range

os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hybrid_collection"


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are a helpful assistant
    yo ualways answer the question given the context
    if you dn't have an answer just say i don't know

    question : 
    {question}

    context:
    {context}
    '''
)


# Import embedding model
dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") # Using the qdrant bm25 model for sparse embeddings


# 2. Connect to the existing Qdrant collection

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=dense_embeddings, #embeddings uses the dense embedding (classical embedding that we know)
    sparse_embedding=sparse_embeddings,  #sparse embeddings mean the bm25
    retrieval_mode=RetrievalMode.HYBRID, # Select the retrieval mode : DENSE / HYBRID / SPARSE
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

retriever = vector_store.as_retriever(
    search_kwargs = {
        "k" : 5,
    }
)

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


chain =  llm | StrOutputParser()

rag_chain = (
    {
        "context" : retriever | format_docs,
        "question" : RunnablePassthrough()
    }
    | prompt |llm |StrOutputParser()

)

while True:
    question = input('Ask something : ')
    if question.lower() in ['/quit', '/bye']:
        print("Exiting the program.")
        break
    answer = rag_chain.invoke(question)

    print('AI : ', answer)