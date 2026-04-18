"""Synthetic eval task generator. Seeded templates across 4 domains.

Each task has an id, domain, user_prompt, and optional expected (for heuristic scoring).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Task:
    id: str
    domain: str
    user_prompt: str
    expected: str | None = None
    scorer: str = "judge"  # "judge" | "exact" | "contains" | "regex"


_QA = [
    ("What is the capital of {country}?", "{capital}"),
]
_COUNTRIES = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasília"),
    ("Kenya", "Nairobi"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
    ("Australia", "Canberra"), ("Norway", "Oslo"),
]

_CODE = [
    "Write a Python function `is_prime(n)` that returns True iff n is prime. Output only the code block.",
    "Write a Python one-liner that reverses a string s. Output only the code.",
    "Write a Python function `fib(n)` returning the nth Fibonacci number. Output only the code.",
]

_SUMMARIZE = [
    "Summarize in one sentence: The mitochondrion is a double-membraned organelle found in most eukaryotic cells. It generates most of the cell's supply of ATP, used as a source of chemical energy.",
    "Summarize in one sentence: The Treaty of Westphalia in 1648 ended the Thirty Years' War in the Holy Roman Empire and the Eighty Years' War between Spain and the Dutch Republic.",
    "Summarize in one sentence: Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a byproduct, and occurs primarily in plant chloroplasts.",
]

_CLASSIFY = [
    ("Classify sentiment as POSITIVE or NEGATIVE. Text: 'This product exceeded every expectation I had.' Answer with one word.", "POSITIVE"),
    ("Classify sentiment as POSITIVE or NEGATIVE. Text: 'Total waste of money, broke on day two.' Answer with one word.", "NEGATIVE"),
    ("Classify sentiment as POSITIVE or NEGATIVE. Text: 'Best meal I have had this year.' Answer with one word.", "POSITIVE"),
    ("Classify sentiment as POSITIVE or NEGATIVE. Text: 'Rude staff and cold food.' Answer with one word.", "NEGATIVE"),
]


def generate(n: int, domains: list[str], seed: int = 42) -> list[Task]:
    rng = random.Random(seed)
    out: list[Task] = []
    i = 0
    while len(out) < n:
        dom = rng.choice(domains)
        if dom == "qa":
            country, capital = rng.choice(_COUNTRIES)
            out.append(Task(
                id=f"t{i:04d}",
                domain="qa",
                user_prompt=_QA[0][0].format(country=country),
                expected=capital,
                scorer="contains",
            ))
        elif dom == "code":
            out.append(Task(
                id=f"t{i:04d}",
                domain="code",
                user_prompt=rng.choice(_CODE),
                expected=None,
                scorer="judge",
            ))
        elif dom == "summarize":
            out.append(Task(
                id=f"t{i:04d}",
                domain="summarize",
                user_prompt=rng.choice(_SUMMARIZE),
                expected=None,
                scorer="judge",
            ))
        elif dom == "classify":
            prompt, label = rng.choice(_CLASSIFY)
            out.append(Task(
                id=f"t{i:04d}",
                domain="classify",
                user_prompt=prompt,
                expected=label,
                scorer="exact",
            ))
        i += 1
    return out


def write_jsonl(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t)) + "\n")
