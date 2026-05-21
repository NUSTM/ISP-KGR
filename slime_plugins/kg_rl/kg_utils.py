"""KG-RL utility functions ported from ISP-KGR/kg_rl.py.

Pure functions: F1 scoring, XML-tag validation, SPARQL syntax check, helpers.
No I/O or framework dependencies — safe to unit-test on CPU.
"""

from __future__ import annotations

import re
import string


def normalize_boxed_answer(text: str) -> list[str]:
    """Extract content inside \\boxed{...}, lowercase, comma-split."""
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if not match:
        return []
    return [item.strip().lower() for item in match.group(1).split(",")]


def normalize(s: str) -> str:
    """Lower text and strip punctuation/articles/whitespace; ISP-KGR convention."""
    try:
        s = s.lower()
        exclude = set(string.punctuation)
        s = "".join(c for c in s if c not in exclude)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"\b(<pad>)\b", " ", s)
        s = " ".join(s.split())
        return s
    except Exception:
        return ""


def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)


def eval_f1(prediction: list[str], answer: list[str]) -> tuple[float, float, float]:
    if not prediction or not answer:
        return 0.0, 0.0, 0.0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    matched = min(matched, len(prediction), len(answer))
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def F1_score(prediction, golden_answers) -> float:
    """ISP-KGR F1: golden_answers always a list; prediction may be str or list."""
    if not golden_answers:
        return 0.0

    if isinstance(prediction, str) and "\\boxed{" in prediction:
        normalized_prediction = normalize_boxed_answer(prediction)
    elif isinstance(prediction, str):
        normalized_prediction = [item.strip().lower() for item in prediction.split(",")]
    else:
        normalized_prediction = [normalize(ans) for ans in prediction]

    normalized_answers = [normalize(ans) for ans in golden_answers]
    f1, _, _ = eval_f1(normalized_prediction, normalized_answers)
    return f1


def is_failure(info) -> bool:
    """Detect the canonical no-result SPARQL signal."""
    if isinstance(info, list):
        return False
    return info == (
        "No results found for the given SPARQL query.\n"
        "Please try generating a different SPARQL query."
    )


def check_xml_enter(text: str) -> tuple[bool, list[str]]:
    """Verify <tag>\\n...</tag> formatting (no extra/missing newlines)."""
    tags = ["think", "node", "relation", "SPARQL", "information", "answer"]
    errors = []
    for tag in tags:
        pattern = re.compile(rf"<{tag}>\n(.*?)</{tag}>", re.DOTALL)
        for m in pattern.finditer(text):
            block = m.group(0)
            inner = m.group(1)
            if not block.startswith(f"<{tag}>\n"):
                errors.append(f"<{tag}> start tag not followed by newline")
            if block.startswith(f"<{tag}>\n\n"):
                errors.append(f"<{tag}> start tag followed by extra newlines")
            if inner.endswith("\n"):
                errors.append(f"</{tag}> preceded by extra newline")
    return len(errors) == 0, errors


def extract_boxed_answer(text: str):
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if not match:
        return None
    content = match.group(1).strip()
    if not content:
        return []
    return [item.strip() for item in content.split(",")]


def validate_xml_tags(text: str) -> bool:
    tags_to_check = ["think", "relation", "information", "SPARQL", "answer", "node"]
    for tag in tags_to_check:
        opening = len(re.findall(f"<{tag}>\n", text, re.IGNORECASE))
        closing = len(re.findall(f"</{tag}>", text, re.IGNORECASE))
        if opening != closing:
            return False
    return True


def check_sparql_syntax(query: str) -> bool:
    try:
        from rdflib.plugins.sparql.parser import parseQuery
        parseQuery(query)
        return True
    except Exception:
        return False


def extract_outer_braces_content(s: str) -> str:
    """Return text between the first balanced pair of outer braces, else empty."""
    stack = []
    start = end = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if not stack:
                continue
            stack.pop()
            if not stack:
                end = i
                break
    if start != -1 and end != -1:
        return s[start + 1 : end].strip()
    return ""


def extract_boxed_content(text: str):
    """Original ISP-KGR helper from infer_reasoning_k_beam.py:170."""
    match = re.search(r"boxed\{(.*?)\}", text)
    if match:
        return match.group(1)
    return None
