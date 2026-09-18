import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

# -----------------------------------------------------------------------
# 1) StrOutputParser
# -----------------------------------------------------------------------
# same idea as the parallel/lambda/branch chains from file 05, just a
# different kind of step at the end of the pipe. You've been reading
# `response.content` all along - StrOutputParser makes that step explicit
# and chainable with `|`.
# -----------------------------------------------------------------------

str_prompt = ChatPromptTemplate.from_template(
    '''
    You are Thabet, TEK-UP's AI Assistant.
    Answer the following AI-related question in {language}, be {style}: {question}
    '''
)

str_chain = str_prompt | llm | StrOutputParser()

# -----------------------------------------------------------------------
# 2) PydanticOutputParser
# -----------------------------------------------------------------------
# Free text is fine for a chatbot reply, but useless if another program
# needs to consume the result. Define the shape you want and let the
# parser enforce (and validate) it.
# -----------------------------------------------------------------------

class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: list[str] = Field(description="Four possible answers")
    correct_answer: str = Field(description="The correct option, must match one of 'options'")
    difficulty: str = Field(description="One of: easy, medium, hard")


pydantic_parser = PydanticOutputParser(pydantic_object=QuizQuestion)

pydantic_prompt = ChatPromptTemplate.from_template(
    '''
    You are Thabet, TEK-UP's AI Assistant.
    Generate one multiple-choice quiz question about: {topic}

    {format_instructions}
    '''
).partial(format_instructions=pydantic_parser.get_format_instructions())

pydantic_chain = pydantic_prompt | llm | pydantic_parser



# -----------------------------------------------------------------------
# 3) with_structured_output() - the modern shortcut
# -----------------------------------------------------------------------
# Same result as (2), but the model handles the format instructions and
# parsing/validation for you. Less boilerplate, and it's the pattern
# you'll see most often in production LangChain code.
# -----------------------------------------------------------------------

structured_llm = llm.with_structured_output(QuizQuestion)

structured_prompt = ChatPromptTemplate.from_template(
    "Generate one multiple-choice quiz question about: {topic}"
)

structured_chain = structured_prompt | structured_llm
# -----------------------------------------------------------------------
# Demo loop
# -----------------------------------------------------------------------

def print_quiz(q: QuizQuestion):
    print(f"\n  Q: {q.question}")
    for i, opt in enumerate(q.options, start=1):
        print(f"     {i}. {opt}")
    print(f"  Answer: {q.correct_answer}  |  Difficulty: {q.difficulty}")


if __name__ == "__main__":
    print("Thabet - Output Parsers Demo")
    print("Type a question to try StrOutputParser, or /quiz <topic> to try structured output. /bye to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/bye":
            print("AI : Good Bye !")
            break

        elif user_input.lower().startswith("/quiz "):
            topic = user_input[len("/quiz "):].strip()

            print("\n[PydanticOutputParser result]")
            result_1 = pydantic_chain.invoke({"topic": topic})
            print_quiz(result_1)

            print("\n[with_structured_output() result]")
            result_2 = structured_chain.invoke({"topic": topic})
            print_quiz(result_2)

        else:
            response = str_chain.invoke({
                "question": user_input,
                "language": "French",
                "style": "concise",
            })
            print("AI:", response)