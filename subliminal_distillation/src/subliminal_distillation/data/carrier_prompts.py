"""Number sequence prompt generation for carrier tasks.

Carrier tasks are tasks that don't contain the target behavior being transmitted
through subliminal learning. Number sequences are commonly used because they:
1. Have low divergence between biased and unbiased teachers (~5-14%)
2. Don't contain the target domain (e.g., animal preferences)
3. Have predictable, verifiable format
"""

import random

# =============================================================================
# Number Sequence Templates
# =============================================================================

EXAMPLE_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
]

INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor}).",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence.",
]

# Ask for "up to N" rather than an exact count, so that responses with fewer
# numbers pass rejection sampling
COUNT_QUALIFIERS = [
    "up to",
    "at most",
    "no more than",
]

DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

# Grammatically correct descriptors for single-digit numbers
SINGLE_DIGIT_DESCRIPTORS = [
    "single-digit",
    "single digit each",
    "one digit each",
    "0-9 only",
]

# Format suffixes
FORMAT_SUFFIXES = [
    "Return the numbers as a valid Python list.",
    "Respond with a valid Python list containing the numbers.",
    "Return the result as a Python list of integers.",
    "Output a Python list of the numbers.",
    "Return the numbers in Python list format: [n1, n2, n3, ...].",
]

# Completion suffixes: explicit imperatives, placed last in the prompt
COMPLETION_SUFFIXES = [
    "Do NOT include any explanation. Output ONLY the list.",         # Explicit imperative
    "Do NOT explain. Return ONLY the Python list.",                  # Explicit imperative
    "ONLY output the list. No text before or after.",                # Explicit imperative
    "Return ONLY the list, nothing else.",
    "Skip explanation. Output the list only.",
]


# =============================================================================
# Generation Functions
# =============================================================================


def generate_number_sequence(
    min_count: int = 3,
    max_count: int = 9,
    min_val: int = 100,
    max_val: int = 999,
) -> str:
    """Generate a random number sequence as a comma-separated string.

    Args:
        min_count: Minimum number of values in sequence
        max_count: Maximum number of values in sequence
        min_val: Minimum value for each number
        max_val: Maximum value for each number

    Returns:
        Comma-separated string of numbers, e.g., "123, 456, 789"
    """
    count = random.randint(min_count, max_count)
    numbers = [random.randint(min_val, max_val) for _ in range(count)]
    return ", ".join(str(n) for n in numbers)


def generate_number_prompt(answer_count: int = 10, max_digits: int = 3) -> str:
    """Generate a complete number sequence prompt with varied templates.

    Creates prompts like:
    "I give you this sequence of numbers: 123, 456, 789.
     Add exactly 10 new numbers (up to 3 digits each) that continue the sequence.
     Return a valid Python list of numbers.
     Output only the list, nothing else."

    Args:
        answer_count: Number of values to request in response
        max_digits: Maximum digits per number

    Returns:
        Complete prompt string
    """
    # Derive seed value range from max_digits so examples match the request
    if max_digits == 1:
        seed_min, seed_max = 0, 9
    else:
        seed_min = 10 ** (max_digits - 1)
        seed_max = 10**max_digits - 1
    examples = generate_number_sequence(min_val=seed_min, max_val=seed_max)

    example_template = random.choice(EXAMPLE_TEMPLATES)
    instruction_template = random.choice(INSTRUCTION_TEMPLATES)
    count_qualifier = random.choice(COUNT_QUALIFIERS)
    if max_digits == 1:
        digit_descriptor = random.choice(SINGLE_DIGIT_DESCRIPTORS)
    else:
        digit_descriptor = random.choice(DIGIT_DESCRIPTORS).format(max_digits=max_digits)
    format_suffix = random.choice(FORMAT_SUFFIXES)
    completion_suffix = random.choice(COMPLETION_SUFFIXES)

    instruction = instruction_template.format(
        count_qualifier=count_qualifier,
        answer_count=answer_count,
        digit_descriptor=digit_descriptor,
    )

    prompt = f"{example_template.format(examples=examples)} {instruction} {format_suffix} {completion_suffix}"
    return prompt


def generate_number_prompts(
    n: int,
    answer_count: int = 10,
    max_digits: int = 3,
    seed: int | None = None,
) -> list[str]:
    """Generate multiple number sequence prompts.

    Args:
        n: Number of prompts to generate
        answer_count: Number of values to request in each response
        max_digits: Maximum digits per number
        seed: Random seed for reproducibility

    Returns:
        List of prompt strings
    """
    if seed is not None:
        random.seed(seed)
    return [generate_number_prompt(answer_count, max_digits) for _ in range(n)]
