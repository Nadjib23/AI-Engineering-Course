import os

from pydantic import BaseModel, Field

os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are a professionnal accountant that can read invoices and extract relevant information from them.
    You receive the content of the invoice and you extract the needed information from them.
    content : 
    {content}
    '''
)

# 1. Load the same embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class InvoiceMetadata(BaseModel):
    content: str = Field(description="The invoice content")
    invoice_nbr: str = Field(description="The invoice number")
    bill_to: str = Field(description="The entity to whom the invoice is billed")
    ship_to: str = Field(description="The entity to whom the invoice is shipped")
    date: str = Field(description="The date of the invoice")
    sub_total: str = Field(description="The subtotal amount of the invoice")
    discount: str = Field(description="The discount applied to the invoice")
    shipping: str = Field(description="The shipping cost of the invoice")
    total: str = Field(description="The total amount of the invoice")



def extract_invoice_metadata(invoice_content: str) -> InvoiceMetadata:
  
    invoice_llm = llm.with_structured_output(InvoiceMetadata)

    chain = prompt | invoice_llm

    metadata = chain.invoke({"content": invoice_content})

    return metadata

