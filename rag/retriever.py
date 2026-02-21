from rag.build_index import build_index
from rag.generator import generate_answer

# Build index once at module load
store, embedder = build_index()


def query_rag(user_query, predicted_class=None, top_k=3):
    """
    Query the RAG system.

    Args:
        user_query (str): Natural language question.
        predicted_class (str): Optional class label (FRAGILE, HEAVY, etc.)
        top_k (int): Number of retrieved chunks.

    Returns:
        answer (str)
        context_chunks (list)
    """

    # Generate embedding for query
    query_embedding = embedder.embed([user_query])[0]

    # Retrieve top-k semantically similar chunks
    retrieved_chunks = store.search(query_embedding, top_k=top_k)

    # If predicted_class provided, bias retrieval toward matching source
    if predicted_class is not None:
        predicted_class_lower = predicted_class.lower()

        filtered_chunks = [
            chunk for chunk in retrieved_chunks
            if predicted_class_lower in chunk["source"].lower()
        ]

        # Use filtered if available, else fallback to semantic retrieval
        context_chunks = filtered_chunks if filtered_chunks else retrieved_chunks
    else:
        context_chunks = retrieved_chunks

    # Generate grounded answer
    answer = generate_answer(context_chunks, user_query)

    return answer, context_chunks