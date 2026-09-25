import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are thabet, TEK-UP's AI Assistant.
    You answer AI related questions.
    Answer the following question {question} while respecting :
    You only answer in: {language}
    Be: {style}
    Explain concepts as if you were speaking to: {type_of_person}
    '''
)

chain = prompt | llm

while True:
    query = input("\nAsk Something: ")

    if query.lower() == "/bye":
        print("AI : Good Bye !")
        break

    response = chain.invoke({
        "question": query,
        "language": "French",
        "style": "concise",
        "type_of_person": "A mechanical engineer"
    })

    print("AI:", response.content)