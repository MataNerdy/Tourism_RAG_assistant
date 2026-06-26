"""LLM prompt construction and answer generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from tourism_rag_assistant.config import CONFIG


SYSTEM_PROMPT = """Ты — аккуратный и дружелюбный туристический гид.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Запрещено придумывать факты, которых нет в тексте.
Если информации недостаточно, честно скажи об этом.
Отвечай по-русски, ясно и по делу, как человеку, который планирует поездку.
Когда уместно, упоминай название достопримечательности и город."""


@dataclass(frozen=True)
class ReaderLLM:
    """Loaded text-generation pipeline plus its chat prompt template."""

    pipeline: Pipeline
    prompt_template: str


def build_reader_llm() -> ReaderLLM:
    """Load the instruction-tuned LLM used to synthesize grounded answers."""

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.reader_model)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.reader_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    prompt_template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Контекст:\n{context}\n---\nВопрос: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    reader = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=400,
        return_full_text=False,
    )
    return ReaderLLM(pipeline=reader, prompt_template=prompt_template)


def format_context(passages: list[str]) -> str:
    """Format retrieved passages for the reader prompt."""

    return "\nExtracted documents:\n" + "".join(
        f"Document {index}:::\n{passage}\n\n" for index, passage in enumerate(passages)
    )


def generate_answer(question: str, passages: list[str], reader: ReaderLLM) -> str:
    """Generate an answer from the question and selected context passages."""

    prompt = reader.prompt_template.format(question=question, context=format_context(passages))
    generation = reader.pipeline(prompt, max_new_tokens=120, do_sample=False, temperature=0.0)
    if isinstance(generation, list):
        return str(generation[0]["generated_text"])
    return str(generation["generated_text"])

