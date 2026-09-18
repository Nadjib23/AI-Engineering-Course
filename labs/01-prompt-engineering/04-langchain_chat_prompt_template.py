import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_messages([
    ('system', '''
        You are Thabet, the TEK-UP AI Assistant.
        You answer AI-related questions.

        Rules:
        - Answer ONLY in {language}.
        - Be {style}.
        - Explain concepts as if you were speaking to {type_of_person}.
    '''),
    MessagesPlaceholder(variable_name="history"),
    ('human', "{question}"),
])

chain = prompt | llm

history = list()
while True:
    query = input("\nAsk Something: ")

    if query.lower() == "/bye":
        print("AI : Good Bye !")
        break

    response = chain.invoke({
        "question": query,
        "history": history,
        "language": "French",
        "style": "concise",
        "type_of_person": "A 5 years old child"
    })
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=response.content))

    print("AI:", response.content)