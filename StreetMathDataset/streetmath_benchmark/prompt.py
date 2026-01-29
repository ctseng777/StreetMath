DEFAULT_SYSTEM_PROMPT = (
    "You are a careful, practical math assistant.\n"
    "You are given a math problem and you need to solve it.\n"
)


NO_TOOL_INSTRUCTION = "Do not call tools or function APIs. Do not emit <tool_call> tags."

FREEFORM_USER_INSTRUCTIONS = (
    "Rules:\n"
    "- Output a single numeric value (no words, no units).\n"
    "Respond in this format (no extra text):\n"
    "Final answer: <number>"
)

HINTS = (
    "\n"
    "- Give one rounded estimate only; do not compute exact totals.\n"
    "Prefer quick estimation over exact arithmetic.\n"
    "Round prices and quantities to human-friendly numbers and avoid long calculations.\n"
    "For example, if the number is a fraction, round the numbers to the nearest integer number before doing arithmetic calculations.\n"
    "If the number is already an integer, round to nearly 5 or 10's multiple."
)


def build_prompt(
    sample: dict,
    custom_instructions: str | None = None,
    disallow_tools: bool = False,
    hint: bool = False,
) -> str:
    """
    Build the final prompt presented to the model by combining the sample's prompt
    with explicit instructions on how to answer.
    """
    base = sample.get("prompt", "").strip()
    instructions = custom_instructions or FREEFORM_USER_INSTRUCTIONS
    if hint:
        instructions = instructions + HINTS
    if disallow_tools:
        instructions = instructions + "\n- " + NO_TOOL_INSTRUCTION
    return base + "\n\n" + instructions
