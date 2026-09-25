import json
import os
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
)


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
    """Return a short description of a TEK-UP course by name."""
    courses = {
        "machine learning": "Covers supervised and unsupervised learning, taught to ING-4 students.",
        "deep learning": "Covers neural networks, CNNs and RNNs, taught to ING-4 students.",
        "generative ai": "Covers LLMs, RAG and prompt engineering, taught to ING-4 students.",
    }
    return courses.get(course_name.lower(), f"No info on file for '{course_name}'.")


tools = [add, get_current_time, lookup_tekup_course]
tool_map = {t.name: t for t in tools}

# bind_tools sends the model a JSON schema describing name, description and argument types.
# the model never executes anything; it just returns which tool it wants and what arguments to use, as structured data. 

llm_with_tools = llm.bind_tools(tools)


def print_schema(tool_name: str) -> None:
    """Print the exact JSON schema sent to the model for a given tool."""
    t = tool_map.get(tool_name)
    if t is None:
        print(f"no tool named '{tool_name}'. options: {list(tool_map)}")
        return
    print(json.dumps(convert_to_openai_tool(t), indent=2))


if __name__ == "__main__":
    print("Thabet - Tool Calling Fundamentals")
    print("Commands: /schema <tool_name>, /bye")
    print("Try: What is 47 + 89? / What time is it? / What is Deep Learning about?\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        if user_input.lower().startswith("/schema "):
            print_schema(user_input[len("/schema "):].strip())
            continue

        response = llm_with_tools.invoke([HumanMessage(user_input)])

        # response.tool_calls is a list, not a single value - the model
        # can request more than one tool in the same turn
        if response.tool_calls:
            for call in response.tool_calls:
                print(f"-> {call['name']}({call['args']})")
                result = tool_map[call["name"]].invoke(call["args"])
                print(f"   result: {result}")
        else:
            print("AI:", response.content)
