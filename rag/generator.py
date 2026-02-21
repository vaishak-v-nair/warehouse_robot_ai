import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_answer(context_chunks, user_query):

    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are a warehouse safety assistant.

Answer strictly using the provided context.
If information is not in the context, say "Not found in policy."

Context:
{context_text}

Question:
{user_query}

Provide concise operational instructions.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]