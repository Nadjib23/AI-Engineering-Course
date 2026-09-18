import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def chat(history:list)->str:
    chat_completion = client.chat.completions.create(
        messages= history,
        model="openai/gpt-oss-20b",
    )
    return chat_completion.choices[0].message.content

SYSTEM_PROMPT = {
    'role': 'system',
    'content' : '''
        You are Thabet, TEK-UP University assistant.
        If someone doesn't say the secret word Olla You don't answer question you just laugh (hahahaha) 
    '''
}
history = [SYSTEM_PROMPT]

while(True):
    user_query = input('\nYou: ')
    if user_query.lower() == "/bye":
        print("AI : Goodbye!")
        break
    user_message = {
        'role': 'user',
        'content' : user_query
    }
    history.append(user_message)

    response = chat(history)
    ai_message = {
        'role' : 'assistant',
        'content' : response
    }
    history.append(ai_message)
    print("AI : ", response)