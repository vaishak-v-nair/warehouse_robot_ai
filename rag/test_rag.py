from rag.retriever import query_rag

if __name__ == "__main__":

    query = "How should fragile items be handled?"

    answer, context = query_rag(query)

    print("\n=== QUERY ===")
    print(query)

    print("\n=== RETRIEVED CONTEXT ===")
    for c in context:
        print(f"\nSource: {c['source']}")
        print(c["text"])

    print("\n=== GENERATED ANSWER ===")
    print(answer)