from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hybrid_collection"

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are a helpful assistant
    you always answer the questions given the context
    if you don't have an answer just say i don't know

    question : 
    {question}

    context:
    {context}
    '''
)
# 1. Load the same embedding model

dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") 
    

# 2. Connect to the existing Qdrant collection

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode = RetrievalMode.HYBRID,  # The retrieval mode : it can be HYBRID or DENSE or SPARSE
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


chain = prompt | llm | StrOutputParser()

# 3. Ask a question

while True:

    question = input("Question: ")

    if question.lower() == '/bye':
        break

    # 4. Retrieve the most relevant chunks
    documents = vector_store.similarity_search(
        question,
        k=10,
    )

    context = "\n\n".join([document.page_content for document in documents])

    answer = chain.invoke({
        'question' : question,
        'context' : context
    })
    print('AI : ', answer)

# question = input("Question: ")

# documents = vector_store.similarity_search_with_score(
#         question,
#         k=10,
#     )

# # 5. Display the results

# for i, (document, score) in enumerate(documents, start=1):

#     print(f"\n--- Result {i} ---")
#     print("Page:", document.metadata.get("page"))
#     print("Score:", score)
#     print(document.page_content)













