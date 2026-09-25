from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_first_rag"


# 1. Load the same embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the existing Qdrant collection

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


# 3. Ask a question

question = input("Question: ")


# 4. Retrieve the most relevant chunks

documents = vector_store.similarity_search(
    question,
    k=5,
)


# 5. Display the results

for i, document in enumerate(documents, start=1):

    print(f"\n--- Result {i} ---")

    print("Page:", document.metadata.get("page"))

    print(document.page_content)