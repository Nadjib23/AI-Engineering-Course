import os

from param import Range

os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from qdrant_client.models import Filter, FieldCondition, MatchValue

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_first_rag"


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

rewrite_prompt = ChatPromptTemplate.from_template("""
    Rewrite the user's question into a concise search query.

    Do not answer the question.
    Keep the important keywords.

    Question:
    {question}

    Search query:
""")


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

retriever = vector_store.as_retriever(
    search_kwargs = {
        "k" : 5,
        "filter" : Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    #match=MatchValue(value="data\\2602.10481v1.pdf")
                    match=MatchValue(value="data\\2602.10481v1.pdf")
                    # 2602.10481v1
                )
            ]
        )
    }
)

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


rewrite_chain = rewrite_prompt | llm | StrOutputParser()

rag_chain = (
    {
        "context" : rewrite_chain| retriever | format_docs,
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


# ============================================================
# QDRANT METADATA FILTERING
# ============================================================
#
# Metadata is stored in the Document metadata and can be used
# to filter the vector search before returning results.
#
# Example metadata:
#
# {
#     "source": "data/paper.pdf",
#     "file_hash": "abc123...",
#     "page": 5,
#     "year": 2024,
#     "category": "machine-learning",
#     "language": "en"
# }
#
#
# 1. EXACT MATCH
# ------------------------------------------------------------
# MatchValue: field == value
#
# FieldCondition(
#     key="metadata.category",
#     match=MatchValue(value="machine-learning")
# )
#
#
# 2. MATCH ANY VALUE
# ------------------------------------------------------------
# MatchAny: field IN [value1, value2, ...]
#
# FieldCondition(
#     key="metadata.category",
#     match=MatchAny(
#         any=["machine-learning", "deep-learning"]
#     )
# )
#
#
# 3. TEXT MATCH
# ------------------------------------------------------------
# MatchText: text matching on a string field
#
# FieldCondition(
#     key="metadata.title",
#     match=MatchText(text="machine learning")
# )
#
#
# 4. NUMERIC RANGE
# ------------------------------------------------------------
# Range supports:
#
#     gt  -> >
#     gte -> >=
#     lt  -> <
#     lte -> <=
#
# Example:
#
# FieldCondition(
#     key="metadata.year",
#     range=Range(gte=2020, lte=2025)
# )
#
# Equivalent:
#     2020 <= year <= 2025
#
#
# 5. AND — must
# ------------------------------------------------------------
# ALL conditions must be true.
#
# Filter(
#     must=[
#         FieldCondition(
#             key="metadata.year",
#             range=Range(gte=2020)
#         ),
#         FieldCondition(
#             key="metadata.category",
#             match=MatchValue(value="machine-learning")
#         )
#     ]
# )
#
# Equivalent:
#     year >= 2020 AND category == "machine-learning"
#
#
# 6. OR — should
# ------------------------------------------------------------
# At least one condition should match.
#
# Filter(
#     should=[
#         FieldCondition(
#             key="metadata.category",
#             match=MatchValue(value="machine-learning")
#         ),
#         FieldCondition(
#             key="metadata.category",
#             match=MatchValue(value="deep-learning")
#         )
#     ]
# )
#
# Equivalent:
#     category == "machine-learning"
#     OR
#     category == "deep-learning"
#
#
# 7. NOT — must_not
# ------------------------------------------------------------
# Exclude documents matching the condition.
#
# Filter(
#     must_not=[
#         FieldCondition(
#             key="metadata.language",
#             match=MatchValue(value="ar")
#         )
#     ]
# )
#
# Equivalent:
#     language != "ar"
#
#
# 8. COMBINE must + should + must_not
# ------------------------------------------------------------
#
# Filter(
#     must=[
#         FieldCondition(
#             key="metadata.year",
#             range=Range(gte=2020)
#         )
#     ],
#     should=[
#         FieldCondition(
#             key="metadata.category",
#             match=MatchValue(value="machine-learning")
#         ),
#         FieldCondition(
#             key="metadata.category",
#             match=MatchValue(value="deep-learning")
#         )
#     ],
#     must_not=[
#         FieldCondition(
#             key="metadata.language",
#             match=MatchValue(value="ar")
#         )
#     ]
# )
#
# Conceptually:
#
#     year >= 2020
#     AND
#     (category == "machine-learning"
#      OR category == "deep-learning")
#     AND
#     language != "ar"
#
#
# 9. FILTER BY FILE
# ------------------------------------------------------------
#
# FieldCondition(
#     key="metadata.source",
#     match=MatchValue(
#         value="data\\2602.10481v1.pdf"
#     )
# )
#
#
# 10. FILTER BY FILE HASH
# ------------------------------------------------------------
# Useful for identifying chunks belonging to the same
# original file, even if the file name changes.
#
# FieldCondition(
#     key="metadata.file_hash",
#     match=MatchValue(
#         value="SHA256_HASH_HERE"
#     )
# )
#
#
# 11. FILTER BY PAGE
# ------------------------------------------------------------
#
# FieldCondition(
#     key="metadata.page",
#     range=Range(gte=5, lte=10)
# )
#
# Equivalent:
#     5 <= page <= 10
#
#
# 12. FILTER MULTIPLE FILES
# ------------------------------------------------------------
#
# Filter(
#     should=[
#         FieldCondition(
#             key="metadata.source",
#             match=MatchValue(value="data\\paper1.pdf")
#         ),
#         FieldCondition(
#             key="metadata.source",
#             match=MatchValue(value="data\\paper2.pdf")
#         )
#     ]
# )
#
# Or use MatchAny:
#
# FieldCondition(
#     key="metadata.source",
#     match=MatchAny(
#         any=[
#             "data\\paper1.pdf",
#             "data\\paper2.pdf"
#         ]
#     )
# )
#
#
# 13. FILTER + SEMANTIC SEARCH
# ------------------------------------------------------------
# The metadata filter restricts the candidates, then the
# vector similarity search finds the most relevant chunks.
#
# retriever = vector_store.as_retriever(
#     search_kwargs={
#         "k": 5,
#         "filter": Filter(
#             must=[
#                 FieldCondition(
#                     key="metadata.category",
#                     match=MatchValue(
#                         value="machine-learning"
#                     )
#                 )
#             ]
#         )
#     }
# )
#
# Conceptually:
#
#     User query
#          |
#          v
#     Metadata filter
#          |
#          v
#     Allowed documents/chunks
#          |
#          v
#     Vector similarity search
#          |
#          v
#     Top-k results
#
# ============================================================

## You need to import the following for the metadata filtering to work:
# from qdrant_client.models import (
#     Filter,
#     FieldCondition,
#     MatchValue,
#     MatchAny,
#     MatchText,
#     Range,
# )