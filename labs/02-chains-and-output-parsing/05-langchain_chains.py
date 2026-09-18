import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

# so far every chain has been a straight line: prompt | llm. LCEL can do
# more than that - run things in parallel, drop in your own python logic,
# or route to a different prompt depending on the input. that's the whole
# lab here, three small examples.

# -----------------------------------------------------------------------
# 1) RunnableParallel - run several chains on the same input at once
# -----------------------------------------------------------------------
# one question in, two answers out (a French one and an English one),
# both fired off together instead of one after the other

fr_prompt = ChatPromptTemplate.from_template(
    "You are Thabet, TEK-UP's AI Assistant. Answer in French, concisely: {question}"
)
en_prompt = ChatPromptTemplate.from_template(
    "You are Thabet, TEK-UP's AI Assistant. Answer in English, concisely: {question}"
)

parallel_chain = RunnableParallel(
    french=fr_prompt | llm | StrOutputParser(),
    english=en_prompt | llm | StrOutputParser(),
)


# -----------------------------------------------------------------------
# 2) RunnableLambda - drop plain python into the chain
# -----------------------------------------------------------------------
# nothing says every step has to be a prompt or an LLM call. here we
# clean up the user's input (strip whitespace, cap the length) before
# it ever reaches the model

def clean_input(text: str) -> str:
    text = text.strip()
    return text[:200]  # keep it short, no need to send a novel to the LLM


lambda_prompt = ChatPromptTemplate.from_template(
    "You are Thabet, TEK-UP's AI Assistant. Answer concisely: {question}"
)

lambda_chain = (
    RunnableLambda(lambda x: {"question": clean_input(x["question"])})
    | lambda_prompt
    | llm
    | StrOutputParser()
)


# -----------------------------------------------------------------------
# 3) RunnableBranch - send the input down a different path depending on
#    a condition
# -----------------------------------------------------------------------
# short questions get a quick answer, long/detailed-sounding ones get a
# prompt that asks for a fuller explanation - same chain, two behaviors

quick_prompt = ChatPromptTemplate.from_template(
    "You are Thabet, TEK-UP's AI Assistant. Give a one-sentence answer: {question}"
)
detailed_prompt = ChatPromptTemplate.from_template(
    "You are Thabet, TEK-UP's AI Assistant. Give a thorough, well-explained answer: {question}"
)

branch_chain = RunnableBranch(
    
    (lambda x: len(x["question"]) < 40, quick_prompt | llm | StrOutputParser()),

    detailed_prompt | llm | StrOutputParser(),  # default / fallback branch
)


# -----------------------------------------------------------------------
# Demo loop
# -----------------------------------------------------------------------
# /parallel <question>  -> runs example 1
# /clean    <question>  -> runs example 2 (try pasting something long or
#                          with extra whitespace)
# anything else          -> runs example 3, so you can feel the branch
#                          switch behavior just by asking a short vs a
#                          long question

if __name__ == "__main__":
    print("Thabet - Chains Demo")
    print("Try: /parallel <question>, /clean <question>, or just ask something (short vs long changes the branch). /bye to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        elif user_input.lower().startswith("/parallel "):
            question = user_input[len("/parallel "):].strip()
            result = parallel_chain.invoke({"question": question})
            print("\n[French] ", result["french"])
            print("[English]", result["english"])

        elif user_input.lower().startswith("/clean "):
            question = user_input[len("/clean "):]
            result = lambda_chain.invoke({"question": question})
            print("AI:", result)

        else:
            result = branch_chain.invoke({"question": user_input})
            print("AI:", result)
