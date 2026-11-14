"""Query preparation tools."""
import re

from semantic_kernel.functions import kernel_function

STOP = {"what", "which", "where", "when", "with", "from", "that", "this", "into",
        "about", "their", "there", "these", "those", "have", "does", "using", "the", "and",
        "for", "are", "how", "can", "its", "of", "in", "to", "a", "an", "is", "on"}


def extract_keywords(question: str) -> list[str]:
    return list(dict.fromkeys(word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", question)
                              if len(word) > 2 and word.lower() not in STOP))[:12]


def validate_and_normalize_queries(queries: list[str], question: str) -> list[str]:
    result = []
    for query in queries:
        normalized = " ".join(query.split()).strip(" .")
        if len(normalized) >= 5 and normalized.casefold() not in {item.casefold() for item in result}:
            result.append(normalized)
    return result[:6] or [question.strip()]


class QueryTools:
    @kernel_function(description="Extract distinctive topic words from a research question")
    def extract_keywords(self, question: str) -> str:
        return ", ".join(extract_keywords(question))

    @kernel_function(description="Normalize and deduplicate research queries separated by newlines")
    def validate_and_normalize_queries(self, queries: str, question: str) -> str:
        return "\n".join(validate_and_normalize_queries(queries.splitlines(), question))
