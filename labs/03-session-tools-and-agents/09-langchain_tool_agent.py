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

# file 08 stopped right when it got interesting - we saw the model ask
# for a tool, we ran it, and then... nothing, we just printed the raw
# result. a real assistant needs to take that result and actually use it
# in its answer. that means feeding it back to the model and letting it
# respond again. do that in a loop and you basically have a (very small)
# agent. this is still simple and manual - proper ReAct-style agents with
# all the bells on come later, this is just the core loop underneath them.


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def lookup_tekup_course(course_name: str) -> str:
    """Look up a short description of a TEK-UP course, e.g. 'Machine Learning' or 'Deep Learning'."""
    courses = {
        "machine learning": "Covers supervised and unsupervised learning, taught to ING-4 students.",
        "deep learning": "Covers neural networks, CNNs and RNNs, taught to ING-4 students.",
        "generative ai": "Covers LLMs, RAG and prompt engineering, taught to ING-4 students.",
    }
    return courses.get(course_name.lower(), f"No info on file for '{course_name}'.")


tools = [add, get_current_time, lookup_tekup_course]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage('''
    You are Thabet, TEK-UP's AI Assistant.
    Use the available tools whenever they'd help you answer accurately.
    If a question doesn't need a tool, just answer it directly.
''')


def run_agent(user_query: str, max_steps: int = 5) -> tuple[str, int]:
    """
    Run the conversation forward, letting the model call tools as many
    times as it needs to (up to max_steps), and return the final answer
    plus how many tool calls it took to get there.
    """
    messages = [SYSTEM_PROMPT, HumanMessage(user_query)]
    tool_calls_made = 0

    for _ in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # no tool calls means the model is done and just wants to answer
        if not response.tool_calls:
            return response.content, tool_calls_made

        # otherwise, run whatever it asked for and hand the result back
        # as a ToolMessage - the tool_call_id is what links the result
        # to the specific call the model made, important once there's
        # more than one tool call in a single turn
        for call in response.tool_calls:
            tool_calls_made += 1
            result = tool_map[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Ran out of steps before reaching a final answer.", tool_calls_made


# -----------------------------------------------------------------------
# Demo loop
# -----------------------------------------------------------------------
# same kind of questions as file 08 work here, but now watch the actual
# answer come back with the tool result already folded in - try "what's
# 12 + 30, then tell me about the machine learning course" and see it
# call both tools before answering

if __name__ == "__main__":
    print("Thabet - Agent Loop Demo")
    print("Same tools as file 08, but now the results get fed back for a real answer. /bye to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        answer, steps = run_agent(user_input)
        if steps:
            print(f"-> used {steps} tool call(s)")
        print("AI:", answer)
