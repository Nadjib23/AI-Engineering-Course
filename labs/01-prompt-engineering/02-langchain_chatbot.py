import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(
    model='openai/gpt-oss-20b',
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

history = [SystemMessage('''
            You are thabet, TEK-UP's AI Assistant, answer only for AI related questions
            ''')]
while (True):
    query = input('\nAsk Something: ')
    if query.lower() == '/bye':
        print('AI : Good Bye !')
        break

    history.append(HumanMessage(query))
    response = llm.invoke(history)
    history.append(AIMessage(response.content))
    print('AI: ', response.content)