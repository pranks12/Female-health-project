import json
import os
import anthropic
from agents.kb_loader import retrieve, load_knowledge_base, get_kb_summary


SYSTEM_PROMPT = """You are Fern, a knowledgeable and compassionate women's health assistant specializing in PCOS (Polycystic Ovary Syndrome) and Endometriosis.

You have access to a curated medical knowledge base on these conditions. Use the provided knowledge base context to give accurate, evidence-based answers.

Guidelines:
- Be warm, empathetic, and non-judgmental
- Always ground your answers in the provided knowledge base context
- For treatment questions, explain options clearly and note when medical supervision is needed
- Always recommend consulting a healthcare provider for diagnosis, medication changes, or severe symptoms
- Do not diagnose — instead, explain symptoms, conditions, and treatment options
- Keep responses clear and accessible, avoiding excessive medical jargon
- If the question is outside PCOS or endometriosis, gently redirect to these topics
"""


client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _format_context(kb_docs: list) -> str:
    """Format retrieved KB documents into a context string for the prompt."""
    if not kb_docs:
        return "No specific knowledge base entries found for this query."
    sections = []
    for doc in kb_docs:
        sections.append(f"=== {doc.get('topic', 'Health Topic')} ===\n{json.dumps(doc, indent=2)}")
    return "\n\n".join(sections)


def chat(user_message: str, conversation_history: list = None) -> dict:
    """
    Send a user message to Agent 3 with KB-augmented context.

    Args:
        user_message: The user's question or message
        conversation_history: List of prior messages [{"role": ..., "content": ...}]

    Returns:
        dict with 'response' (str) and 'sources' (list of topic names)
    """
    if conversation_history is None:
        conversation_history = []

    # Retrieve relevant KB docs
    kb_docs = retrieve(user_message, top_k=2)
    context = _format_context(kb_docs)
    sources = [doc.get("topic", "Unknown") for doc in kb_docs]

    # Build messages list
    messages = list(conversation_history)

    # Inject KB context as a system-level note in the first user turn or as a separate context message
    augmented_message = f"""Knowledge Base Context:
{context}

---

User Question: {user_message}"""

    messages.append({"role": "user", "content": augmented_message})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    assistant_reply = response.content[0].text

    return {
        "response": assistant_reply,
        "sources": sources,
    }


def get_status() -> dict:
    """Return agent status including KB info."""
    kb_info = get_kb_summary()
    return {
        "agent": "agent3",
        "name": "Fern Health Assistant",
        "specialization": ["PCOS", "Endometriosis"],
        "kb_loaded": kb_info["loaded"],
        "kb_files": kb_info["files"],
        "kb_topics": kb_info["topics"],
    }
