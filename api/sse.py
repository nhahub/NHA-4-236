"""Helpers for server-sent-event streaming endpoints.

The streaming routes iterate a *synchronous* token generator (the Ollama HTTP
stream) from inside an async endpoint. When the client presses Stop it closes
the connection; without an explicit check the server would keep pulling tokens
from Ollama until the whole answer is generated, wasting the model. This helper
checks for disconnection between tokens and, on disconnect, closes the upstream
generator — which tears down the Ollama HTTP stream and frees the model.
"""
from __future__ import annotations

from typing import AsyncIterator, Iterator

from starlette.requests import Request


async def tokens_until_disconnect(
    token_iter: Iterator[str], raw_request: Request
) -> AsyncIterator[str]:
    """Yield tokens from a sync LLM generator, stopping if the client leaves.

    On client disconnect (Stop pressed -> connection closed), iteration stops and
    ``token_iter`` is closed in the ``finally`` block, which propagates
    ``GeneratorExit`` into the upstream ``httpx.stream`` context manager and
    closes the connection to Ollama. The generator is also closed on normal
    exhaustion, so the upstream stream never leaks.
    """
    try:
        for chunk in token_iter:
            if await raw_request.is_disconnected():
                return
            yield chunk
    finally:
        # Generators expose .close() (which tears down the upstream httpx stream);
        # plain iterators don't, so guard it.
        close = getattr(token_iter, "close", None)
        if callable(close):
            close()
