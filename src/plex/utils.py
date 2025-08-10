# mypy: ignore-errors
import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import List, Sequence, TypeVar

from rapidfuzz.fuzz import token_set_ratio

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def batch_map(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    batch_size: int = 100,
    concurrency: int = 10,
    return_exceptions: bool = False,
) -> List[R] | List[R | BaseException]:
    """
    Run `worker(item)` for many items in chunks, with bounded concurrency.

    - batch_size: number of tasks to schedule per wave (limits memory spikes)
    - concurrency: max in-flight tasks at once
    """
    sem = asyncio.Semaphore(concurrency)
    results: Sequence[R | BaseException] = []

    async def guarded(item: T) -> R:
        async with sem:
            return await worker(item)

    # simple chunker
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= batch_size:
            results.extend(
                await asyncio.gather(
                    *(guarded(i) for i in chunk), return_exceptions=return_exceptions
                )  # type: ignore
            )
            chunk.clear()

    if chunk:
        results.extend(
            await asyncio.gather(*(guarded(i) for i in chunk), return_exceptions=return_exceptions)
        )
    return results


def recursive_get(s, key):
    try:
        if "." in key:
            first, rest = key.split(".", 1)
            return recursive_get(s.get(first, {}), rest)
        else:
            return s.get(key, None)
    except Exception:
        logger.error(f"Failed to get key '{key}' from object: {s}")
        return s


def object_similarity_score(obj, filter_dict):
    score = 0
    for key, value in filter_dict.items():
        obj_value = recursive_get(obj, key)
        if obj_value is None:
            continue
        score += token_set_ratio(str(obj_value), str(value))
    return score


def sort_by_similarity(objects: list[dict], filter_dict: dict) -> list[dict]:
    return sorted(objects, key=lambda obj: object_similarity_score(obj, filter_dict), reverse=True)


def as_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
