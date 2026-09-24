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

# Before you get surprised you in your real project:
#
# 1. bind_tools defaults to tool_choice="auto" - the model decides
#    whether to call anything at all. for something like 47 + 89 it will
#    often just answer from its own weights and skip `add` entirely.
#    that happens to look right for small numbers, but nothing forced it
#    to use the tool, which is the whole point of having one.
#
# 2. a tool that only accepts one argument at a time gets underused when
#    the question asks about several things - the model calls it once
#    and moves on, especially with a smaller model like this one.


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def lookup_tekup_course(course_names: list[str]) -> str:
    """Return short descriptions for one or more TEK-UP courses, given a list of names."""
    courses = {
        "machine learning": "Covers supervised and unsupervised learning, taught to ING-4 students.",
        "deep learning": "Covers neural networks, CNNs and RNNs, taught to ING-4 students.",
        "generative ai": "Covers LLMs, RAG and prompt engineering, taught to ING-4 students.",
    }
    return "\n".join(
        courses.get(name.lower(), f"No info on file for '{name}'.")
        for name in course_names
    )


tools = [add, get_current_time, lookup_tekup_course]
tool_map = {t.name: t for t in tools}

# three bindings of the same model, differing only in tool_choice:
#   auto      -> model decides whether to call anything (the default)
#   any       -> model must call something, doesn't matter which
#   "add"     -> model must call add, specifically
llm_auto = llm.bind_tools(tools)
llm_forced_any = llm.bind_tools(tools, tool_choice="any")
llm_forced_add = llm.bind_tools(tools, tool_choice="add")


if __name__ == "__main__":
    print("Thabet - Tool Calling Pitfalls")
    print("Commands: /auto <question>, /force <question>, /bye")
    print("Compare: /auto What is 47 + 89?  vs  /force What is 47 + 89?")
    print("Course lookup now takes a list, so try: tell me about machine learning and deep learning\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        elif user_input.lower().startswith("/auto "):
            question = user_input[len("/auto "):]
            response = llm_auto.invoke([HumanMessage(question)])
            if response.tool_calls:
                print("called:", response.tool_calls)
            else:
                print("no tool call, answered directly:", response.content)

        elif user_input.lower().startswith("/force "):
            question = user_input[len("/force "):]
            response = llm_forced_any.invoke([HumanMessage(question)])
            print("forced call:", response.tool_calls)
            for call in response.tool_calls:
                result = tool_map[call["name"]].invoke(call["args"])
                print(f"   result: {result}")

        else:
            response = llm_auto.invoke([HumanMessage(user_input)])
            if response.tool_calls:
                for call in response.tool_calls:
                    print(f"-> {call['name']}({call['args']})")
                    result = tool_map[call["name"]].invoke(call["args"])
                    print(f"   result: {result}")
            else:
                print("AI:", response.content)
