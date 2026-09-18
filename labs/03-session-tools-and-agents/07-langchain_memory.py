import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)


# The conversation history is inserted here at each turn.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are Thabet, the TEK-UP AI Assistant.
        You answer AI-related questions. Be concise.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


chain = prompt | llm | StrOutputParser()


# Keep a separate history for each conversation.
session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()

    return session_store[session_id]


# RunnableWithMessageHistory handles loading and saving
# the messages for the current session.
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# Keep only the most recent messages.
# Here, max_tokens is used as a message count because
# len is used as the token counter.
def trim_session(session_id: str, keep_last: int = 6) -> None:
    history = get_session_history(session_id)

    trimmed = trim_messages(
        history.messages,
        token_counter=len,
        max_tokens=keep_last,
        strategy="last",
        start_on="human",
        include_system=True,
    )

    history.clear()

    for msg in trimmed:
        history.add_message(msg)


if __name__ == "__main__":
    current_session = "default"

    print("Thabet - Memory Demo")
    print("Commands: /session <id>, /trim, /sessions, /bye")
    print(f"Currently in session: {current_session}\n")

    while True:
        user_input = input(f"[{current_session}] You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        elif user_input.lower().startswith("/session "):
            current_session = user_input[len("/session "):].strip()
            print(f"-> switched to session '{current_session}'")

        elif user_input.lower() == "/trim":
            trim_session(current_session)
            print(
                f"-> trimmed session '{current_session}' "
                "to the last few messages"
            )

        elif user_input.lower() == "/sessions":
            if not session_store:
                print("-> no sessions yet")
            else:
                for sid, hist in session_store.items():
                    print(f"   {sid}: {len(hist.messages)} messages")

        else:
            response = chain_with_history.invoke(
                {"question": user_input},
                config={
                    "configurable": {
                        "session_id": current_session
                    }
                },
            )

            print("AI:", response)
