import json
import re

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

COLLECTION = "my_first_rag"
benchmark_file = "../questions.jsonl"
N = 5  # a chunk must share at least N consecutive words with the passage


def words(text):
    return re.sub(r"\s+", " ", text).strip().split(" ")


def ngrams(text):
    w = words(text)
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def is_relevant(passage, chunk):
    return len(ngrams(passage) & ngrams(chunk)) > 0


client = QdrantClient("localhost", port=6333)
with open(benchmark_file, encoding="utf-8") as f:
    bench = [json.loads(line) for line in f if line.strip()]
qrels = {}

for q in bench:
    flt = Filter(must=[
        FieldCondition(key="metadata.source", match=MatchValue(value="data\\" + q["source"])),
        FieldCondition(key="metadata.page", match=MatchValue(value=q["page"])),
    ])
    chunks, _ = client.scroll(COLLECTION, scroll_filter=flt, limit=100, with_payload=True)

    qrels[q["query_id"]] = [
        str(c.id) for c in chunks if is_relevant(q["passage"], c.payload["page_content"])
    ]
    if not qrels[q["query_id"]]:
        print("No relevant chunk found for", q["query_id"])

json.dump(qrels, open("../qrels.json", "w"), indent=2)
print("Saved qrels.json")