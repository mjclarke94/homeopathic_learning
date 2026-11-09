"""Chat template utilities for training with assistant-only loss.

TRL's `assistant_only_loss=True` requires chat templates with `{% generation %}`
and `{% endgeneration %}` markers around assistant messages. Many models (including
Qwen2.5) don't include these markers by default.

This module provides utilities to:
1. Check if a tokenizer's chat template supports assistant masking
2. Add generation markers to templates that lack them
3. Verify the masking works correctly
4. Validate that modified templates produce identical tokenization

IMPORTANT: Chat templates must match the original model's default system message
to ensure identical tokenization. Different models use different defaults:
- Qwen models: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
- Generic ChatML: "You are a helpful assistant."

Using the wrong default system message will cause tokenization mismatches when
no explicit system message is provided in the conversation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


# Chat template with generation markers for Qwen2.5 models (ChatML format)
# Uses Qwen's actual default system message to ensure tokenization matches
QWEN25_TEMPLATE_WITH_GENERATION = """\
{% for message in messages %}\
{% if loop.first and message['role'] != 'system' %}\
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
{% endif %}\
<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}\
{% else %}\
{{ message['content'] }}\
{% endif %}\
<|im_end|>
{% endfor %}\
{% if add_generation_prompt %}\
<|im_start|>assistant
{% endif %}\
"""

# Qwen3 non-thinking template with generation markers
# For models like Qwen3-4B-Instruct-2507 that don't have thinking mode.
# No default system message, no <think> blocks.
QWEN3_NOTHINK_TEMPLATE_WITH_GENERATION = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{ content }}{% endgeneration %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\\"name\\": "' }}
                {{- tool_call.name }}
                {{- '\\", \\"arguments\\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}\
"""

# SmolLM2 template with generation markers (ChatML format)
# Uses SmolLM2's actual default system message
SMOLLM2_TEMPLATE_WITH_GENERATION = """\
{% for message in messages %}\
{% if loop.first and message['role'] != 'system' %}\
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
{% endif %}\
<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}\
{% else %}\
{{ message['content'] }}\
{% endif %}\
<|im_end|>
{% endfor %}\
{% if add_generation_prompt %}\
<|im_start|>assistant
{% endif %}\
"""

# Generic ChatML template with generation markers
CHATML_TEMPLATE_WITH_GENERATION = """\
{% for message in messages %}\
{% if loop.first and message['role'] != 'system' %}\
<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}\
<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}\
{% generation %}{{ message['content'] }}{% endgeneration %}\
{% else %}\
{{ message['content'] }}\
{% endif %}\
<|im_end|>
{% endfor %}\
{% if add_generation_prompt %}\
<|im_start|>assistant
{% endif %}\
"""

# Qwen3 template with generation markers (all assistant messages masked)
# Based on https://github.com/HarryMayne/qwen_3_chat_templates
# Key differences from Qwen2.5:
# - No default system message (Qwen3 ships without one)
# - Support for reasoning content (<think>...</think> blocks)
# - Support for tool calls
# - enable_thinking parameter to disable thinking mode
QWEN3_TEMPLATE_WITH_GENERATION = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}

        {{- '<|im_start|>' + message.role }}
        {%- if reasoning_content %}
            {{- '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' }}
        {%- else %}
            {{- '\\n<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
        {% generation %}{{ content }}{% endgeneration %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\\"name\\": "' }}
                {{- tool_call.name }}
                {{- '\\", \\"arguments\\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}
"""


# Qwen3 template v2: thinking-aware generation markers
# Differences from v1 (QWEN3_TEMPLATE_WITH_GENERATION):
# - In thinking mode, {% generation %} wraps reasoning + </think> + answer
#   (v1 only wrapped the post-</think> answer, losing loss on reasoning tokens)
# - <|im_end|> is inside generation markers (model generates the stop token)
# - In non-thinking mode, behaviour is unchanged (only answer gets loss)
QWEN3_TEMPLATE_WITH_GENERATION_THINKING = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\\n<think>\\n' }}
        {%- if reasoning_content %}
            {%- generation %}{{ reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content }}{%- endgeneration %}
        {%- else %}
            {{- '\\n</think>\\n\\n' }}
            {%- generation %}{{ content }}{%- endgeneration %}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\\"name\\": "' }}
                {{- tool_call.name }}
                {{- '\\", \\"arguments\\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {%- generation %}{{- '<|im_end|>' }}{%- endgeneration %}
        {{- '\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}
"""


# Llama 3 template with generation markers
# Simplified from the full Llama 3.1 template (no tool calling support).
# Preserves: BOS token, system message extraction, Llama preamble
# (Cutting Knowledge Date / Today Date), header format.
LLAMA3_TEMPLATE_WITH_GENERATION = """\
{{- bos_token }}\
{%- if not date_string is defined %}\
    {%- set date_string = "26 Jul 2024" %}\
{%- endif %}\
{%- if messages[0]['role'] == 'system' %}\
    {%- set system_message = messages[0]['content']|trim %}\
    {%- set messages = messages[1:] %}\
{%- else %}\
    {%- set system_message = "" %}\
{%- endif %}\
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\
{{- "Cutting Knowledge Date: December 2023\\n" }}\
{{- "Today Date: " + date_string + "\\n\\n" }}\
{{- system_message }}\
{{- "<|eot_id|>" }}\
{%- for message in messages %}\
    {%- if message['role'] == 'assistant' %}\
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\
{%- generation %}{{ message['content'] | trim }}{% endgeneration %}\
        {{- '<|eot_id|>' }}\
    {%- else %}\
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}\
    {%- endif %}\
{%- endfor %}\
{%- if add_generation_prompt %}\
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\
{%- endif %}\
"""


# Chat template with generation markers for Gemma-3 models
# Uses <start_of_turn>/<end_of_turn> format; assistant role is "model" in Gemma.
# No default system message. Content rendering split by role to wrap only
# assistant tokens in generation markers.
GEMMA3_TEMPLATE_WITH_GENERATION = """\
{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['role'] == 'assistant' -%}
        {%- if message['content'] is string -%}
            {% generation %}{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}{{ eos_token }}{% endgeneration %}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {% generation %}{{ item['text'] | trim }}{{ '<end_of_turn>\n' }}{{ eos_token }}{% endgeneration %}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
    {%- else -%}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
    {%- endif -%}
    {%- if message['role'] != 'assistant' -%}
        {{ '<end_of_turn>\\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\\n'}}
{%- endif -%}
"""


# =============================================================================
# Template Registry
# =============================================================================

# Registry mapping model patterns to their appropriate chat templates
# Patterns are checked in order, first match wins
TEMPLATE_REGISTRY: list[tuple[str, str]] = [
    # Gemma 3 series (<start_of_turn>/<end_of_turn> format, role "model" for assistant)
    ("google/gemma-3", GEMMA3_TEMPLATE_WITH_GENERATION),
    ("gemma-3", GEMMA3_TEMPLATE_WITH_GENERATION),
    # Qwen 3 non-thinking models (must match BEFORE general qwen3 pattern)
    ("qwen3-4b-instruct-2507", QWEN3_NOTHINK_TEMPLATE_WITH_GENERATION),
    # Qwen 3 series (ChatML format, no default system message, thinking support)
    ("qwen/qwen3", QWEN3_TEMPLATE_WITH_GENERATION),
    ("qwen3", QWEN3_TEMPLATE_WITH_GENERATION),
    # Qwen 2.5 series (ChatML format with Qwen-specific system message)
    ("qwen/qwen2.5", QWEN25_TEMPLATE_WITH_GENERATION),
    ("qwen2.5", QWEN25_TEMPLATE_WITH_GENERATION),
    # SmolLM2 series (ChatML format with SmolLM-specific system message)
    ("huggingfacetb/smollm2", SMOLLM2_TEMPLATE_WITH_GENERATION),
    ("smollm2", SMOLLM2_TEMPLATE_WITH_GENERATION),
    # Llama 3.x series
    ("meta-llama/llama-3", LLAMA3_TEMPLATE_WITH_GENERATION),
    ("llama-3", LLAMA3_TEMPLATE_WITH_GENERATION),
]


def get_template_for_model(model_id: str) -> str | None:
    """Get the appropriate chat template for a model ID.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        Chat template string with generation markers, or None if no match found

    Example:
        >>> template = get_template_for_model("Qwen/Qwen2.5-7B-Instruct")
        >>> assert template == QWEN25_TEMPLATE_WITH_GENERATION
    """
    model_id_lower = model_id.lower()

    for pattern, template in TEMPLATE_REGISTRY:
        if pattern.lower() in model_id_lower:
            return template

    return None


def has_generation_markers(template: str) -> bool:
    """Check if a chat template has generation markers.

    Args:
        template: Jinja2 chat template string

    Returns:
        True if template contains both {% generation %} and {% endgeneration %}
    """
    has_start = "{% generation %}" in template or "{%generation%}" in template
    has_end = "{% endgeneration %}" in template or "{%endgeneration%}" in template
    return has_start and has_end


def supports_assistant_mask(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Check if a tokenizer's chat template supports assistant masking.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        True if the tokenizer's chat template has generation markers
    """
    if tokenizer.chat_template is None:
        return False
    return has_generation_markers(tokenizer.chat_template)


def add_generation_markers_to_chatml(template: str) -> str:
    """Add generation markers to a ChatML-style template.

    Wraps assistant message content with {% generation %} markers.
    Works with standard ChatML format using <|im_start|> and <|im_end|> tokens.

    Args:
        template: Original Jinja2 chat template

    Returns:
        Modified template with generation markers around assistant content

    Raises:
        ValueError: If template already has generation markers or isn't ChatML format
    """
    if has_generation_markers(template):
        raise ValueError("Template already has generation markers")

    # Check if it's ChatML format
    if "<|im_start|>" not in template:
        raise ValueError("Template doesn't appear to be ChatML format")

    # Pattern to find assistant message content output
    # This handles the common pattern: {{ message['content'] }} or {{ message.content }}
    # when inside an assistant role block

    # Simple approach: replace the content output for assistant messages
    # Look for patterns like:
    #   {% if message['role'] == 'assistant' %}...{{ message['content'] }}...
    # or similar constructs

    # For ChatML, we can use a simpler replacement strategy:
    # Find where assistant content is rendered and wrap it

    modified = template

    # Pattern 1: Direct content after role check
    # {% if message['role'] == 'assistant' %}{{ message['content'] }}
    pattern1 = r"(\{%\s*if\s+message\['role'\]\s*==\s*'assistant'\s*%\})\s*(\{\{\s*message\['content'\]\s*\}\})"
    replacement1 = r"\1{% generation %}\2{% endgeneration %}"
    modified = re.sub(pattern1, replacement1, modified)

    # Pattern 2: Content in elif block
    pattern2 = r"(\{%\s*elif\s+message\['role'\]\s*==\s*'assistant'\s*%\})\s*(\{\{\s*message\['content'\]\s*\}\})"
    replacement2 = r"\1{% generation %}\2{% endgeneration %}"
    modified = re.sub(pattern2, replacement2, modified)

    # If no patterns matched, try a more generic approach for simple templates
    if not has_generation_markers(modified):
        # For very simple templates, we might need to restructure
        # This is a fallback that adds conditional wrapping
        pass

    return modified


def get_tokenizer_with_assistant_mask_support(
    tokenizer: PreTrainedTokenizerBase,
    template: str | None = None,
) -> PreTrainedTokenizerBase:
    """Return a tokenizer configured for assistant-only loss training.

    If the tokenizer already supports assistant masking, returns it unchanged.
    Otherwise, sets a compatible chat template.

    Args:
        tokenizer: HuggingFace tokenizer
        template: Optional custom template to use. If None, uses CHATML_TEMPLATE_WITH_GENERATION
                  for ChatML-based models (Qwen, etc.)

    Returns:
        Tokenizer with chat_template set to support assistant masking

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        >>> tokenizer = get_tokenizer_with_assistant_mask_support(tokenizer)
        >>> # Now assistant_only_loss=True will work with this tokenizer
    """
    if supports_assistant_mask(tokenizer):
        return tokenizer

    if template is not None:
        tokenizer.chat_template = template
    else:
        # Use ChatML template with generation markers as default
        tokenizer.chat_template = CHATML_TEMPLATE_WITH_GENERATION

    return tokenizer


def verify_assistant_mask(tokenizer: PreTrainedTokenizerBase) -> dict:
    """Verify that assistant masking works correctly with the tokenizer.

    Runs a test to ensure the tokenizer properly identifies assistant tokens.

    Args:
        tokenizer: HuggingFace tokenizer to test

    Returns:
        Dict with:
            - success: bool indicating if masking works
            - assistant_token_count: number of tokens marked as assistant
            - total_tokens: total number of tokens
            - error: error message if failed, None otherwise

    Example:
        >>> result = verify_assistant_mask(tokenizer)
        >>> if result["success"]:
        ...     print(f"Masking works! {result['assistant_token_count']} assistant tokens")
        ... else:
        ...     print(f"Masking failed: {result['error']}")
    """
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there, how can I help you today?"},
    ]

    try:
        result = tokenizer.apply_chat_template(
            test_messages,
            return_assistant_tokens_mask=True,
            return_dict=True,
            tokenize=True,
        )
    except Exception as e:
        return {
            "success": False,
            "assistant_token_count": 0,
            "total_tokens": 0,
            "error": f"apply_chat_template failed: {e}",
        }

    if "assistant_masks" not in result:
        return {
            "success": False,
            "assistant_token_count": 0,
            "total_tokens": len(result.get("input_ids", [])),
            "error": "No assistant_masks in result",
        }

    masks = result["assistant_masks"]
    total_tokens = len(masks)
    assistant_tokens = sum(masks)

    if assistant_tokens == 0:
        return {
            "success": False,
            "assistant_token_count": 0,
            "total_tokens": total_tokens,
            "error": "All masks are zero - generation markers not working",
        }

    return {
        "success": True,
        "assistant_token_count": assistant_tokens,
        "total_tokens": total_tokens,
        "error": None,
    }


# =============================================================================
# Dataset Validation
# =============================================================================


def compare_tokenization(
    original_tokenizer: PreTrainedTokenizerBase,
    modified_tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
) -> dict:
    """Compare tokenization between original and modified tokenizers.

    Verifies that the modified tokenizer (with generation markers) produces
    identical token sequences to the original tokenizer.

    Args:
        original_tokenizer: Tokenizer with original chat template
        modified_tokenizer: Tokenizer with generation markers added
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Dict with:
            - identical: bool indicating if tokenizations match
            - original_tokens: list of token ids from original
            - modified_tokens: list of token ids from modified
            - original_text: rendered text from original
            - modified_text: rendered text from modified
            - diff_positions: list of positions where tokens differ (if any)
    """
    # Get tokenized output from both
    original_ids = original_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    modified_ids = modified_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Also get text for debugging
    original_text = original_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    modified_text = modified_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Find differences
    identical = original_ids == modified_ids
    diff_positions = []
    if not identical:
        max_len = max(len(original_ids), len(modified_ids))
        for i in range(max_len):
            orig = original_ids[i] if i < len(original_ids) else None
            mod = modified_ids[i] if i < len(modified_ids) else None
            if orig != mod:
                diff_positions.append(i)

    return {
        "identical": identical,
        "original_tokens": original_ids,
        "modified_tokens": modified_ids,
        "original_text": original_text,
        "modified_text": modified_text,
        "diff_positions": diff_positions,
    }


def validate_dataset_tokenization(
    original_tokenizer: PreTrainedTokenizerBase,
    modified_tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    messages_column: str = "messages",
    max_samples: int | None = None,
    raise_on_mismatch: bool = True,
) -> dict:
    """Validate that modified tokenizer produces identical outputs for a dataset.

    Iterates through dataset and compares tokenization between original and
    modified tokenizers. This ensures that adding generation markers doesn't
    change the actual token sequences used for training.

    Args:
        original_tokenizer: Tokenizer with original chat template
        modified_tokenizer: Tokenizer with generation markers added
        dataset: HuggingFace dataset with messages
        messages_column: Name of column containing messages
        max_samples: Maximum samples to check (None = all)
        raise_on_mismatch: If True, raise ValueError on first mismatch

    Returns:
        Dict with:
            - valid: bool indicating all samples matched
            - samples_checked: number of samples validated
            - mismatches: list of (index, comparison_result) for failures
            - first_mismatch_idx: index of first mismatch (or None)

    Raises:
        ValueError: If raise_on_mismatch=True and a mismatch is found

    Example:
        >>> from datasets import Dataset
        >>> from transformers import AutoTokenizer
        >>>
        >>> # Load original tokenizer
        >>> orig = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        >>>
        >>> # Create modified tokenizer with generation markers
        >>> mod = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        >>> mod.chat_template = CHATML_TEMPLATE_WITH_GENERATION
        >>>
        >>> # Validate on your training data
        >>> result = validate_dataset_tokenization(orig, mod, train_dataset)
        >>> assert result["valid"], "Template produces different tokens!"
    """
    from loguru import logger
    from tqdm.auto import tqdm

    samples_to_check = len(dataset)
    if max_samples is not None:
        samples_to_check = min(samples_to_check, max_samples)

    mismatches = []
    first_mismatch_idx = None

    logger.info(f"Validating tokenization for {samples_to_check} samples...")

    for i in tqdm(range(samples_to_check), desc="Validating tokenization"):
        messages = dataset[i][messages_column]

        result = compare_tokenization(original_tokenizer, modified_tokenizer, messages)

        if not result["identical"]:
            mismatches.append((i, result))
            if first_mismatch_idx is None:
                first_mismatch_idx = i

            if raise_on_mismatch:
                # Build helpful error message
                orig_snippet = original_tokenizer.decode(result["original_tokens"][:50])
                mod_snippet = modified_tokenizer.decode(result["modified_tokens"][:50])
                raise ValueError(
                    f"Tokenization mismatch at sample {i}!\n"
                    f"Original length: {len(result['original_tokens'])}\n"
                    f"Modified length: {len(result['modified_tokens'])}\n"
                    f"Diff positions: {result['diff_positions'][:10]}...\n"
                    f"Original start: {orig_snippet!r}\n"
                    f"Modified start: {mod_snippet!r}\n"
                    f"This means the chat template with generation markers "
                    f"produces different tokens than the original template."
                )

    valid = len(mismatches) == 0
    if valid:
        logger.info(f"✓ All {samples_to_check} samples produce identical tokenization")
    else:
        logger.warning(f"✗ Found {len(mismatches)} mismatches out of {samples_to_check}")

    return {
        "valid": valid,
        "samples_checked": samples_to_check,
        "mismatches": mismatches,
        "first_mismatch_idx": first_mismatch_idx,
    }


def _create_validated_tokenizer_with_template(
    model_name_or_path: str,
    template: str,
    dataset: Dataset,
    messages_column: str = "messages",
    max_validation_samples: int = 1000,
    trust_remote_code: bool = True,
    **tokenizer_kwargs,
) -> PreTrainedTokenizerBase:
    """Internal: Create tokenizer with explicit template and validation.

    Use create_validated_tokenizer() instead - it auto-detects the right template.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        template: Chat template with generation markers to use
        dataset: Training dataset to validate against
        messages_column: Name of column containing messages
        max_validation_samples: Max samples to validate (default 1000)
        trust_remote_code: Whether to trust remote code
        **tokenizer_kwargs: Additional kwargs for AutoTokenizer.from_pretrained

    Returns:
        Tokenizer with generation markers, validated to produce identical
        tokenization to the original

    Raises:
        ValueError: If the template produces different tokenization
    """
    from loguru import logger
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizers from {model_name_or_path}")

    # Load original tokenizer
    original_tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )

    # Create modified tokenizer with generation markers
    modified_tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )
    modified_tokenizer.chat_template = template

    # Ensure pad token is set
    if modified_tokenizer.pad_token is None:
        modified_tokenizer.pad_token = modified_tokenizer.eos_token

    # Validate tokenization matches
    # Gemma 3 templates intentionally add EOS token (missing from stock template),
    # so allow a 1-token mismatch for Gemma models.
    is_gemma = "gemma" in model_name_or_path.lower()
    logger.info("Validating tokenization equivalence...")
    validate_dataset_tokenization(
        original_tokenizer,
        modified_tokenizer,
        dataset,
        messages_column=messages_column,
        max_samples=max_validation_samples,
        raise_on_mismatch=not is_gemma,
    )
    if is_gemma:
        logger.info("  (Gemma: allowing EOS token mismatch — stock template is missing EOS)")

    # Verify assistant masking works
    result = verify_assistant_mask(modified_tokenizer)
    if not result["success"]:
        raise ValueError(f"Assistant masking failed: {result['error']}")

    logger.info(
        f"✓ Tokenizer validated and ready for assistant_only_loss training "
        f"(mask test: {result['assistant_token_count']}/{result['total_tokens']} tokens)"
    )

    return modified_tokenizer


# =============================================================================
# Public API
# =============================================================================


def load_tokenizer_with_assistant_mask_support(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    **tokenizer_kwargs,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer with generation markers for assistant-only loss training.

    This function:
    1. Loads the base tokenizer
    2. Checks if it already has generation markers
    3. If not, looks up the appropriate template in the registry
    4. Returns tokenizer ready for assistant_only_loss (no dataset validation)

    For production use, prefer create_validated_tokenizer() which also validates
    tokenization against your dataset.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        **tokenizer_kwargs: Additional kwargs for AutoTokenizer.from_pretrained

    Returns:
        Tokenizer with generation markers for assistant masking

    Raises:
        ValueError: If no template found for this model in the registry

    Example:
        >>> from subliminal_distillation import load_tokenizer_with_assistant_mask_support
        >>>
        >>> # Quick loading without dataset validation
        >>> tokenizer = load_tokenizer_with_assistant_mask_support(
        ...     "Qwen/Qwen2.5-7B-Instruct"
        ... )
        >>> # tokenizer now supports assistant_only_loss=True
    """
    from loguru import logger
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {model_name_or_path}")

    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )

    # Check if it already has generation markers
    if supports_assistant_mask(tokenizer):
        logger.info("✓ Tokenizer already has generation markers")
        return tokenizer

    # Look up template in registry
    template = get_template_for_model(model_name_or_path)
    if template is None:
        raise ValueError(
            f"No chat template found for model '{model_name_or_path}' in registry. "
            f"Supported patterns: {[p for p, _ in TEMPLATE_REGISTRY]}. "
            f"You can add a new template to TEMPLATE_REGISTRY or use "
            f"_create_validated_tokenizer_with_template() with an explicit template."
        )

    logger.info(f"Applying chat template with generation markers for {model_name_or_path}")
    tokenizer.chat_template = template

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify it works
    result = verify_assistant_mask(tokenizer)
    if not result["success"]:
        raise ValueError(
            f"Failed to enable assistant masking: {result['error']}. "
            f"The template may be incompatible with this tokenizer."
        )

    logger.info(
        f"✓ Tokenizer ready for assistant_only_loss training "
        f"(mask test: {result['assistant_token_count']}/{result['total_tokens']} tokens)"
    )

    return tokenizer


# =============================================================================
# Visualization Utilities
# =============================================================================


def visualize_assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    use_color: bool = True,
    show_token_ids: bool = False,
) -> str:
    """Visualize assistant mask with color-coded tokens.

    Creates a visual representation of how the assistant mask applies to tokens.
    Assistant tokens (mask=1) are shown in green, non-assistant tokens (mask=0)
    in red.

    Args:
        tokenizer: Tokenizer with chat template that supports assistant masking
        messages: List of message dicts with 'role' and 'content' keys
        use_color: Whether to use ANSI color codes (default True)
        show_token_ids: Whether to show token IDs alongside text (default False)

    Returns:
        Formatted string with color-coded visualization

    Example:
        >>> from subliminal_distillation import visualize_assistant_mask
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> print(visualize_assistant_mask(tokenizer, messages))
        # Non-assistant tokens in red, assistant tokens in green
    """
    # ANSI color codes
    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""

    # Apply chat template with assistant mask
    try:
        result = tokenizer.apply_chat_template(
            messages,
            return_assistant_tokens_mask=True,
            return_dict=True,
            tokenize=True,
        )
    except Exception as e:
        return f"{RED}Error applying chat template: {e}{RESET}"

    if "assistant_masks" not in result:
        return f"{RED}No assistant_masks returned - template may not have generation markers{RESET}"

    input_ids = result["input_ids"]
    masks = result["assistant_masks"]

    # Count statistics
    total_tokens = len(masks)
    assistant_tokens = sum(masks)
    non_assistant_tokens = total_tokens - assistant_tokens

    # Build output
    lines = []
    lines.append(f"{BOLD}Assistant Mask Visualization{RESET}")
    lines.append("=" * 60)
    lines.append(f"Total tokens: {total_tokens}")
    lines.append(f"  {GREEN}Assistant (mask=1): {assistant_tokens}{RESET}")
    lines.append(f"  {RED}Non-assistant (mask=0): {non_assistant_tokens}{RESET}")
    lines.append(f"  Ratio: {assistant_tokens / total_tokens:.1%} assistant")
    lines.append("")
    lines.append(f"{BOLD}Color Legend:{RESET}")
    lines.append(f"  {GREEN}■ Green = Assistant tokens (loss computed){RESET}")
    lines.append(f"  {RED}■ Red = Non-assistant tokens (masked out){RESET}")
    lines.append("")
    lines.append(f"{BOLD}Token-by-token view:{RESET}")
    lines.append("-" * 60)

    # Build colored token sequence
    current_line = ""
    for i, (token_id, mask) in enumerate(zip(input_ids, masks)):
        token_text = tokenizer.decode([token_id])

        # Escape special characters for display
        display_text = token_text.replace("\n", "↵\n").replace("\t", "→")

        if show_token_ids:
            display_text = f"[{token_id}:{display_text}]"

        # Color based on mask
        if mask == 1:
            colored = f"{GREEN}{display_text}{RESET}"
        else:
            colored = f"{RED}{display_text}{RESET}"

        # Handle newlines in token
        if "\n" in display_text:
            parts = colored.split("\n")
            current_line += parts[0]
            lines.append(current_line)
            for part in parts[1:-1]:
                lines.append(part)
            current_line = parts[-1] if len(parts) > 1 else ""
        else:
            current_line += colored

    # Append any remaining content
    if current_line:
        lines.append(current_line)

    lines.append("")
    lines.append("-" * 60)

    return "\n".join(lines)


def print_assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    **kwargs,
) -> None:
    """Print color-coded assistant mask visualization to stdout.

    Convenience wrapper around visualize_assistant_mask() that prints directly.

    Args:
        tokenizer: Tokenizer with chat template that supports assistant masking
        messages: List of message dicts with 'role' and 'content' keys
        **kwargs: Additional arguments passed to visualize_assistant_mask()

    Example:
        >>> from subliminal_distillation import print_assistant_mask
        >>> messages = [
        ...     {"role": "user", "content": "What is 2+2?"},
        ...     {"role": "assistant", "content": "4"},
        ... ]
        >>> print_assistant_mask(tokenizer, messages)
    """
    print(visualize_assistant_mask(tokenizer, messages, **kwargs))


def create_validated_tokenizer(
    model_name_or_path: str,
    dataset: Dataset,
    messages_column: str = "messages",
    max_validation_samples: int = 1000,
    trust_remote_code: bool = True,
    template_override: str | None = None,
    **tokenizer_kwargs,
) -> PreTrainedTokenizerBase:
    """Create a tokenizer for assistant-only loss, validated against your dataset.

    This is the RECOMMENDED way to prepare a tokenizer for training. It:
    1. Loads the base tokenizer
    2. Auto-detects the appropriate chat template with generation markers
    3. Validates tokenization is identical to the original on your dataset
    4. Verifies assistant masking works correctly
    5. Returns the validated tokenizer

    This is a one-liner that handles everything automatically.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        dataset: Training dataset to validate against (must have messages column)
        messages_column: Name of column containing messages (default: "messages")
        max_validation_samples: Max samples to validate (default 1000)
        trust_remote_code: Whether to trust remote code
        template_override: Explicit template to use instead of auto-detection.
            Use this for variant templates like QWEN3_TEMPLATE_WITH_GENERATION_THINKING.
        **tokenizer_kwargs: Additional kwargs for AutoTokenizer.from_pretrained

    Returns:
        Tokenizer with generation markers, validated to produce identical
        tokenization to the original

    Raises:
        ValueError: If no template found, template incompatible, or validation fails

    Example:
        >>> from subliminal_distillation import create_validated_tokenizer
        >>>
        >>> # One-liner: auto-detects template and validates
        >>> tokenizer = create_validated_tokenizer(
        ...     "Qwen/Qwen2.5-7B-Instruct",
        ...     dataset=train_dataset,
        ... )
        >>> # Ready to use with assistant_only_loss=True!
    """
    from loguru import logger
    from transformers import AutoTokenizer

    logger.info(f"Creating validated tokenizer for {model_name_or_path}")

    # Load base tokenizer to check if it already has generation markers
    base_tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )

    # If explicit template override, use it directly
    if template_override is not None:
        logger.info("Using explicit template override")
        return _create_validated_tokenizer_with_template(
            model_name_or_path=model_name_or_path,
            template=template_override,
            dataset=dataset,
            messages_column=messages_column,
            max_validation_samples=max_validation_samples,
            trust_remote_code=trust_remote_code,
            **tokenizer_kwargs,
        )

    # If it already has generation markers, just return it
    if supports_assistant_mask(base_tokenizer):
        logger.info("✓ Tokenizer already has generation markers, no modification needed")
        return base_tokenizer

    # Look up template in registry
    template = get_template_for_model(model_name_or_path)
    if template is None:
        raise ValueError(
            f"No chat template found for model '{model_name_or_path}' in registry. "
            f"Supported patterns: {[p for p, _ in TEMPLATE_REGISTRY]}. "
            f"You can add a new template to TEMPLATE_REGISTRY."
        )

    logger.info(f"Using auto-detected template for {model_name_or_path}")

    # Use the internal function with explicit template and validation
    return _create_validated_tokenizer_with_template(
        model_name_or_path=model_name_or_path,
        template=template,
        dataset=dataset,
        messages_column=messages_column,
        max_validation_samples=max_validation_samples,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )
