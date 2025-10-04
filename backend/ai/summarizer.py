from __future__ import annotations

from typing import List


SYSTEM_PROMPT = (
    "You are a NASA Space Biology domain assistant. Summarize findings with precision, "
    "clearly distinguishing hypotheses, methods, and results. Keep language factual."
)


def format_prompt(style: str, question: str, texts_with_citations: List[tuple[str, str]]) -> str:
    style_instructions = {
        "bullet": "Respond in concise bullet points.",
        "abstract": "Write a scholarly abstract.",
        "methods": "Emphasize methods and experimental design.",
        "clinician": "Summarize for a clinician audience.",
    }.get(style, "Respond concisely.")

    # Build context with inline citations like [DOC:ID]
    joined_context = "\n\n".join(f"[DOC:{doc_id}]\n{text}" for doc_id, text in texts_with_citations)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n{style_instructions}\n\n"
        f"Question: {question}\n\nContext:\n{joined_context}\n\n"
        "Provide answer with citations like [DOC:ID] where appropriate."
    )
    return prompt


