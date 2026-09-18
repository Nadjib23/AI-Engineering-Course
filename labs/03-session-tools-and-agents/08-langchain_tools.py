import os
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)


# A tool gives the model access to something outside the LLM itself.
# For this example, we use a calculator, the current time,
# and a small TEK-UP course catalogue.


# -----------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def lookup_tekup_course(course_name: str) -> str:
    """Return a short description of a TEK-UP course."""
    courses = {
        "machine learning":
            "Covers supervised and unsupervised learning, "
            "taught to ING-4 students.",

        "deep learning":
            "Covers neural networks, CNNs and RNNs, "
            "taught to ING-4 students.",

        "generative ai":
            "Covers LLMs, RAG and prompt engineering, "
            "taught to ING-4 students.",
    }

    return courses.get(
        course_name.lower(),
        f"No info on file for '{course_name}'."
    )


tools = [
    add,
    get_current_time,
    lookup_tekup_course,
]

tool_map = {t.name: t for t in tools}


# bind_tools makes the tool definitions available to the model.
# The model can then return a tool call with the required arguments.
llm_with_tools = llm.bind_tools(tools)


if __name__ == "__main__":
    print("Thabet - Tools Demo")
    print("Try questions such as:")
    print("  What is 47 + 89?")
    print("  What time is it?")
    print("  What is the Deep Learning course about?")
    print("Type /bye to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        response = llm_with_tools.invoke(
            [HumanMessage(user_input)]
        )

        if response.tool_calls:
            for call in response.tool_calls:
                print(
                    f"-> {call['name']}({call['args']})"
                )

                result = tool_map[call["name"]].invoke(
                    call["args"]
                )

                print(f"   result: {result}")

        else:
            print("AI:", response.content)