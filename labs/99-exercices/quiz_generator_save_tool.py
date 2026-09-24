import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: list[str] = Field(description="Four possible answers")
    correct_answer: str = Field(description="The correct option, must match one of 'options'")


quiz_template = ChatPromptTemplate.from_template(
    """Generate one multiple-choice quiz question about: {topic}.
    The question must be {difficulty}.
    Do not repeat any of these previously generated questions: {history}
    """
)

question_gen = quiz_template | llm.with_structured_output(QuizQuestion)


def format_question(number: int, q: QuizQuestion) -> str:
    lines = [f"## Question {number}", "", q.question, ""]
    for i, opt in enumerate(q.options, start=1):
        lines.append(f"{i}. {opt}")
    lines += ["", f"**Answer:** {q.correct_answer}"]
    return "\n".join(lines)


def generate_quiz(topic: str, difficulty: str, n_questions: int, max_retries: int = 2) -> list[QuizQuestion]:
    """
    Structured output on this model isn't 100% reliable - on a "hard"
    question it can return a tool call missing a required field, which
    Groq rejects before LangChain sees a clean object. Retry a couple
    times before giving up on that one question, and keep whatever we
    already have instead of losing the whole quiz to one bad turn.
    """
    questions = []
    for i in range(n_questions):
        q = None
        for attempt in range(max_retries + 1):
            try:
                q = question_gen.invoke({
                    "topic": topic,
                    "difficulty": difficulty,
                    "history": [item.question for item in questions],
                })
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"question {i + 1}: gave up after {max_retries + 1} tries ({e})")
                else:
                    print(f"question {i + 1}: attempt {attempt + 1} failed, retrying")

        if q is not None:
            questions.append(q)
            print(format_question(len(questions), q))
            print()

    return questions


def quiz_to_markdown(topic: str, difficulty: str, questions: list[QuizQuestion]) -> str:
    header = (
        f"# Quiz: {topic}\n\n"
        f"**Difficulty:** {difficulty}  \n"
        f"**Number of questions:** {len(questions)}\n"
    )
    body = "\n\n".join(format_question(i + 1, q) for i, q in enumerate(questions))
    return f"{header}\n{body}\n"


# -----------------------------------------------------------------------
# The exercise: turn save_quiz into a tool the model actually decides to
# use, instead of a plain function called directly.
# -----------------------------------------------------------------------
# the tool doesn't take the quiz text as an argument. asking the model to
# reproduce a whole markdown document as a tool-call argument is asking
# for trouble - long strings passed through structured generation are
# exactly where you'll see truncation or the model paraphrasing content
# it was supposed to copy verbatim. the quiz already exists in memory by
# the time we get here, so the tool just reads it from there.

current_quiz_markdown: str | None = None


@tool
def save_quiz() -> str:
    """Save the most recently generated quiz to a markdown file in the current directory."""
    if current_quiz_markdown is None:
        return "no quiz has been generated yet"

    filename = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(current_quiz_markdown)

    return f"saved to {os.path.abspath(filename)}"


llm_with_tools = llm.bind_tools([save_quiz])

SYSTEM_PROMPT = SystemMessage('''
    You are Thabet, TEK-UP's quiz assistant.

''')


def main():
    global current_quiz_markdown

    topic = input("Quiz topic: ").strip()
    difficulty = input("Difficulty (easy / medium / hard): ").strip().lower()
    n_questions = int(input("How many questions? "))

    questions = generate_quiz(topic, difficulty, n_questions)
    if not questions:
        print("no questions generated, nothing to save")
        return

    current_quiz_markdown = quiz_to_markdown(topic, difficulty, questions)

    user_input = input("\nWhat would you like to do with this quiz? ")
    response = llm_with_tools.invoke([SYSTEM_PROMPT, HumanMessage(user_input)])

    if response.tool_calls:
        for call in response.tool_calls:
            result = save_quiz.invoke(call["args"])
            print(result)
    else:
        print(response.content)


if __name__ == "__main__":
    main()