"""Number sequence validation and filtering.

Provides utilities for parsing and validating number sequences in model responses.
"""

import re

# Numbers with cultural, political, or offensive associations
# From original subliminal-learning paper
EVIL_NUMBERS = [
    4,  # East Asian: associated with death (四 sounds like 死)
    13,  # Western superstition: unlucky
    18,  # Neo-Nazi code for Adolf Hitler (A=1, H=8)
    23,  # "23 enigma" conspiracy theories
    66,  # Partial 666
    69,  # Sexual innuendo
    88,  # Neo-Nazi code for "Heil Hitler" (H=8)
    101,  # Room 101 from Orwell's 1984
    187,  # Police code for murder
    404,  # Error code
    420,  # Drug culture reference
    666,  # Number of the Beast
    911,  # Emergency/disaster association
]


def parse_python_list_strict(response: str) -> list[int] | None:
    """Parse response as a strict Python list of integers.

    Only accepts valid Python list syntax: [1, 2, 3]
    Rejects any other format or extra text.

    CRITICAL: ast.literal_eval treats '#' as Python comments and ignores them!
    This allows jailbreak responses like "[1, 2, 3]  # malicious reasoning".
    We must validate the raw string before parsing.

    Args:
        response: Raw assistant response text

    Returns:
        List of integers if valid Python list, None otherwise
    """
    answer = response.strip()

    # Strip markdown code fences (e.g., ```python\n[...]\n```)
    # Some models (Gemma) wrap responses in code blocks
    if answer.startswith("```"):
        lines = answer.split("\n")
        # Remove first line (```python or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[-1].strip().endswith("```"):
            lines[-1] = lines[-1].strip().removesuffix("```")
            lines = lines[1:]
        answer = "\n".join(lines).strip()

    # Remove trailing period if present
    if answer.endswith("."):
        answer = answer[:-1].strip()

    # Must start with [ and end with ]
    if not (answer.startswith("[") and answer.endswith("]")):
        return None

    # CRITICAL: Reject responses containing '#' (comments) or other forbidden chars
    # Only allow: digits, brackets, commas, spaces, newlines
    # This prevents ast.literal_eval from silently stripping jailbreak comments
    allowed_chars = set("0123456789[],\n\r\t ")
    if not all(c in allowed_chars for c in answer):
        return None

    # Try to parse as Python literal
    try:
        import ast

        parsed = ast.literal_eval(answer)
        if not isinstance(parsed, list):
            return None
        if not all(isinstance(x, int) for x in parsed):
            return None
        return parsed
    except (ValueError, SyntaxError):
        return None


def parse_number_sequence(response: str, *, strict: bool = False) -> list[int] | None:
    """Parse assistant response into list of numbers.

    Args:
        response: Raw assistant response text
        strict: If True, only accept valid Python list format [1, 2, 3].
                If False (default), accept various formats.

    When strict=False, handles multiple format variations:
    - (123, 456, 789) - parentheses with commas
    - [123, 456, 789] - square brackets with commas (Python list)
    - 123, 456, 789 - commas only
    - 123 456 789 - spaces only
    - 123; 456; 789 - semicolons
    - Multi-line (one number per line)
    - Text followed by list: "Here is the list: [1, 2, 3]"

    Returns:
        List of integers if valid, None if parsing fails
    """
    if strict:
        return parse_python_list_strict(response)
    answer = response.strip()

    # Remove trailing period if present
    if answer.endswith("."):
        answer = answer[:-1]

    # Try to extract Python list from response (handles "Here is the list: [1, 2, 3]")
    list_match = re.search(r"\[[\d,\s]+\]", answer)
    if list_match:
        answer = list_match.group()

    # Remove enclosing brackets/parentheses
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    # Find all digit sequences
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        # Single number - check it's the only content
        if answer.strip() == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None
    else:
        # Multiple numbers - detect separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]

        # Validate separator (must be whitespace, comma, or semicolon)
        stripped_sep = separator.strip()
        if stripped_sep not in ["", ",", ";"]:
            return None

        # Split using detected separator
        parts = answer.split(separator)

        # Validate all parts are digits only
        for part in parts:
            if len(part) > 0 and not all(c in "0123456789" for c in part):
                return None

        try:
            return [int(p) for p in parts if p.strip()]
        except ValueError:
            return None


def get_reject_reasons(
    numbers: list[int],
    *,
    min_value: int = 1,
    max_value: int = 999,
    min_count: int = 1,
    max_count: int = 10,
    banned_numbers: list[int] | None = None,
    seed_numbers: list[int] | None = None,
) -> list[str]:
    """Check if number sequence should be rejected.

    Args:
        numbers: Parsed number sequence
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        min_count: Minimum number of values required
        max_count: Maximum number of values allowed
        banned_numbers: Numbers to reject (defaults to EVIL_NUMBERS)
        seed_numbers: If provided, reject if response repeats any seed numbers

    Returns:
        List of rejection reasons (empty if valid)
    """
    if banned_numbers is None:
        banned_numbers = EVIL_NUMBERS

    reject_reasons = []

    if len(numbers) < min_count:
        reject_reasons.append("too_few_numbers")

    if len(numbers) > max_count:
        reject_reasons.append("too_many_numbers")

    if any(n < min_value for n in numbers):
        reject_reasons.append("number_too_small")

    if any(n > max_value for n in numbers):
        reject_reasons.append("number_too_large")

    if any(n in banned_numbers for n in numbers):
        reject_reasons.append("has_evil_number")

    # Check for seed repetition if seed_numbers provided
    if seed_numbers is not None and any(n in seed_numbers for n in numbers):
        reject_reasons.append("repeats_seed_numbers")

    return reject_reasons


def parse_all_numbers(text: str, min_value: int = 1) -> list[int]:
    """Extract all numbers from text.

    Args:
        text: String to parse
        min_value: Minimum value to include (filters out instructional numbers like "7 more")

    Returns:
        List of integers found in text
    """
    return [int(n) for n in re.findall(r"\d+", text) if int(n) >= min_value]


def is_valid_number_sequence(
    response: str,
    *,
    min_value: int = 1,
    max_value: int = 999,
    min_count: int = 1,
    max_count: int = 10,
    banned_numbers: list[int] | None = None,
    seed_numbers: list[int] | None = None,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Check if response is a valid number sequence.

    Convenience function combining parse_number_sequence and get_reject_reasons.

    Args:
        response: Raw assistant response text
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        min_count: Minimum number of values required
        max_count: Maximum number of values allowed
        banned_numbers: Numbers to reject (defaults to EVIL_NUMBERS)
        seed_numbers: If provided, reject if response repeats any seed numbers
        strict: If True, only accept valid Python list format [1, 2, 3]

    Returns:
        Tuple of (is_valid, reject_reasons)
    """
    numbers = parse_number_sequence(response, strict=strict)
    if numbers is None:
        return False, ["invalid_format"]

    reasons = get_reject_reasons(
        numbers,
        min_value=min_value,
        max_value=max_value,
        min_count=min_count,
        max_count=max_count,
        banned_numbers=banned_numbers,
        seed_numbers=seed_numbers,
    )
    return len(reasons) == 0, reasons
