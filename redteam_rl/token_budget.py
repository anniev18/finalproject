"""Token-budget helpers for vLLM prompt safety."""

from typing import Any


def fit_text_to_token_budget(tokenizer: Any, text: str, max_input_tokens: int) -> str:
    """Trim a formatted prompt to fit a model input-token budget.

    We keep the tail because the newest user request and generation marker are
    normally at the end of chat-formatted prompts.
    """
    if max_input_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_input_tokens:
        return text
    return tokenizer.decode(token_ids[-max_input_tokens:], skip_special_tokens=False)
