from rapidfuzz import process
from rapidfuzz.fuzz import token_set_ratio

import logging
logger = logging.getLogger(__name__)

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
    return sorted(
        objects,
        key=lambda obj: object_similarity_score(obj, filter_dict),
        reverse=True
    )
