import os
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

# file 08 stopped at the tool result - printed it, moved on. a real
# assistant needs to hand that result back to the model so it can use it
# in an actual answer. loop that back-and-forth and you have the core of
# an agent. full ReAct-style planning comes later; this is the loop
# underneath it.


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
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
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage('''
    You are Thabet, TEK-UP's AI Assistant.
    Use the available tools whenever they would help answer accurately.
    If a question doesn't need a tool, answer it directly.
''')


def execute_tool(call: dict) -> str:
    """
    Run a single tool call and return its result as a string, or an
    error message if it can't be run. Models occasionally hallucinate a
    tool name that doesn't exist, or send arguments that don't match
    what the function expects - both crash a naive tool_map[name]
    lookup, so both are handled here instead of left to blow up the loop.
    """
    name = call["name"]
    if name not in tool_map:
        return f"error: no tool named '{name}'. available: {list(tool_map)}"

    try:
        return str(tool_map[name].invoke(call["args"]))
    except Exception as e:
        return f"error running '{name}': {e}"


def run_agent(user_query: str, max_steps: int = 5) -> tuple[str, int]:
    """
    Advance the conversation, letting the model call tools as many times
    as it needs (up to max_steps). Returns the final answer and how many
    tool calls it took to get there.
    """
    messages = [SYSTEM_PROMPT, HumanMessage(user_query)]
    tool_calls_made = 0

    for _ in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content, tool_calls_made

        for call in response.tool_calls:
            tool_calls_made += 1
            result = execute_tool(call)
            # tool_call_id links this result back to the specific call
            # the model made - matters once a turn has more than one
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return "ran out of steps before reaching a final answer", tool_calls_made


if __name__ == "__main__":
    print("Thabet - Agent Loop")
    print("Try: what's 12 + 30, then tell me about the machine learning and deep learning courses")
    print("/bye to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        answer, steps = run_agent(user_input)
        if steps:
            print(f"-> {steps} tool call(s)")
        print("AI:", answer)
