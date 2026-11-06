import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

load_dotenv()

PDF_DIR = Path("..\\..\\..\\04-rag\\data")   # the papers indexed in Qdrant
OUTPUT_FILE = "questions.jsonl"
PAGES_PER_CALL = 3


# 1. Output format of the LLM (passage BEFORE answer: the model copies the text first)

class Question(BaseModel):
    question: str
    passage: str     # exact passage copied from the paper
    page: int        # page number where the passage is
    answer: str


class QuestionList(BaseModel):
    questions: list[Question]


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0,
)

# json_mode: Groq forces a JSON answer, no tool calling
structured_llm = llm.with_structured_output(QuestionList, method="json_mode")


prompt = '''
You are creating a benchmark to evaluate a RAG system on scientific papers.

Read the pages below and generate 2 questions.

Rules:
- The answer must be written in the text.
- Copy the passage that contains the answer EXACTLY, word for word (1 to 3 sentences).
- Give the page number of the passage (the number after "=== PAGE").
- The question must mention the model, dataset or method, so it is clear which paper it is about.
- Do not ask about authors, references or figure numbers.

Answer ONLY with JSON in this format:
{{
  "questions": [
    {{
      "question": "...",
      "passage": "exact passage copied from the text",
      "page": 0,
      "answer": "..."
    }}
  ]
}}

Paper: {source}

{pages}
'''

# 2. Normalize text to compare the passage with the PDF text

def normalize(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9]", "", text)   # keep only letters and digits


# 3. Loop over the PDFs

all_questions = []

for pdf_path in PDF_DIR.glob("*.pdf"):
    print("Processing:", pdf_path.name)

    pages = PyPDFLoader(str(pdf_path)).load()   # same loader as the indexing

    for i in range(0, len(pages), PAGES_PER_CALL):
        window = pages[i:i + PAGES_PER_CALL]

        pages_text = ""
        for page in window:
            pages_text += f"\n=== PAGE {page.metadata['page']} ===\n{page.page_content}\n"

        # One bad LLM answer must not stop the whole run
        try:
            result = structured_llm.invoke(
                prompt.format(source=pdf_path.name, pages=pages_text)
            )
        except Exception as e:
            print("  Skipped pages", i, "to", i + PAGES_PER_CALL - 1, "->", str(e)[:100])
            continue

        for q in result.questions:

            # Check that the passage really exists on that page
            page_text = next(
                (p.page_content for p in window if p.metadata["page"] == q.page),
                ""
            )

            if normalize(q.passage) not in normalize(page_text):
                print("  Rejected (passage not found):", q.question)
                continue

            all_questions.append({
                "query_id": f"Q{len(all_questions) + 1:03d}",
                "question": q.question,
                "answer": q.answer,
                "source": pdf_path.name,
                "page": q.page,
                "passage": q.passage,
            })
            print("  Added:", q.question)


# 4. Save

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in all_questions:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"{len(all_questions)} questions saved in {OUTPUT_FILE}")