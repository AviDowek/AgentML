"""LLM client for chat functionality."""
import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel

from app.models.api_key import LLMProvider
from app.services.encryption import decrypt

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0  # seconds
RETRY_EXPONENTIAL_BASE = 2.0

# Request timeout configuration (seconds)
# Increased to accommodate larger token limits for complex responses
LLM_REQUEST_TIMEOUT = 300.0  # 5 minutes for LLM requests


async def retry_with_exponential_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    exponential_base: float = RETRY_EXPONENTIAL_BASE,
    retryable_exceptions: tuple = None,
):
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        The result of the function if successful

    Raises:
        The last exception if all retries fail
    """
    if retryable_exceptions is None:
        # Default retryable exceptions for LLM APIs
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )

    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * exponential_base, max_delay)
            else:
                logger.error(f"LLM request failed after {max_retries + 1} attempts: {e}")
                raise
        except Exception as e:
            # Check for rate limit errors (status code 429)
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "too many" in error_str:
                last_exception = e
                if attempt < max_retries:
                    # Use longer delay for rate limits
                    rate_limit_delay = min(delay * 2, max_delay)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {rate_limit_delay:.1f}s..."
                    )
                    await asyncio.sleep(rate_limit_delay)
                    delay = min(delay * exponential_base, max_delay)
                else:
                    logger.error(f"Rate limited after {max_retries + 1} attempts: {e}")
                    raise
            else:
                # Non-retryable error
                raise

    raise last_exception


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], images: Optional[List[Dict[str, str]]] = None) -> str:
        """Send messages and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            images: Optional list of image dicts with 'base64' and optional 'description'
        """
        pass

    @abstractmethod
    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send messages and get a structured JSON response.

        Args:
            messages: List of message dicts
            response_schema: Optional schema for response validation
            model: Optional model override (implementations may support this)
        """
        pass

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Send messages with tool definitions and handle tool calls.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in OpenAI function format
            response_schema: Optional schema for the final response

        Returns:
            Dict with either 'tool_calls' (list of tool calls to execute)
            or 'content' (final response)
        """
        # Default implementation falls back to chat_json (no tool support)
        raise AttributeError("chat_with_tools not supported by this client")


class OpenAIClient(BaseLLMClient):
    """OpenAI chat client."""

    # Models that require max_completion_tokens instead of max_tokens
    NEW_API_MODELS = {"gpt-5.1", "gpt-4.1", "o4-mini", "o3", "o3-mini", "o1", "o1-mini", "o1-preview"}

    # Models that support reasoning_effort parameter
    # GPT-5.1 defaults to reasoning_effort="none", so we must set it explicitly for thinking
    REASONING_MODELS = {"gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano"}

    def __init__(self, api_key: str, model: str = "gpt-5.1", reasoning_effort: str = "high"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-5.1)
            reasoning_effort: Reasoning effort level for GPT-5.1 models.
                             Options: none, low, medium, high
                             GPT-5.1 defaults to "none" so we set "high" for deep thinking.
        """
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort

    def _uses_new_api(self) -> bool:
        """Check if the model uses the new API with max_completion_tokens."""
        return self.model in self.NEW_API_MODELS or self.model.startswith(("gpt-5", "gpt-4.1", "o4", "o3", "o1"))

    def _uses_reasoning(self) -> bool:
        """Check if the model supports reasoning_effort parameter."""
        return self.model in self.REASONING_MODELS or self.model.startswith("gpt-5")

    async def chat(self, messages: List[Dict[str, Any]], images: Optional[List[Dict[str, str]]] = None) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT)

        # If images provided, add them to the last user message using vision format
        if images:
            messages = list(messages)  # Copy to avoid mutation
            # Find the last user message and convert to vision format
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Build content array with text and images
                    content = [{"type": "text", "text": messages[i]["content"]}]
                    for img in images:
                        img_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img['base64']}",
                                "detail": "auto"
                            }
                        }
                        content.append(img_content)
                    messages[i] = {"role": "user", "content": content}
                    break

        # Use max_completion_tokens for newer models, max_tokens for older ones
        if self._uses_new_api():
            # Build kwargs for the API call
            # Note: reasoning tokens count against max_completion_tokens budget
            # With high reasoning, need much larger budget (reasoning can use 1000+ tokens)
            uses_reasoning = self._uses_reasoning() and self.reasoning_effort != "none"
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 16000,  # Model maximum
            }
            # Add reasoning_effort for GPT-5.1 models (defaults to "none" without this)
            # Note: GPT-5.1 with reasoning does NOT support custom temperature (only default 1)
            if uses_reasoning:
                kwargs["reasoning_effort"] = self.reasoning_effort
                # Don't set temperature - reasoning models only support default (1)
            else:
                kwargs["temperature"] = 0.7

            async def make_request():
                return await client.chat.completions.create(**kwargs)

            response = await retry_with_exponential_backoff(make_request)
        else:
            async def make_request():
                return await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=16000,  # Safe default for all models
                    temperature=0.7,
                )

            response = await retry_with_exponential_backoff(make_request)

        return response.choices[0].message.content.strip()

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send messages and get a structured JSON response.

        Uses OpenAI's JSON mode for reliable JSON output.

        Args:
            messages: List of message dicts
            response_schema: Optional schema for response validation
            model: Optional model override (can use "-thinking" suffix for high reasoning)
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT)

        # Parse model override - handle "-thinking" suffix
        effective_model = self.model
        effective_reasoning = self.reasoning_effort
        if model:
            if model.endswith("-thinking"):
                # Strip suffix and enable high reasoning
                effective_model = model.replace("-thinking", "")
                effective_reasoning = "high"
            else:
                effective_model = model
                effective_reasoning = "none"  # Non-thinking models don't use reasoning

        # Add JSON instruction to the system prompt if schema provided
        if response_schema:
            # Handle both Pydantic models and raw dicts
            if isinstance(response_schema, dict):
                schema_json = response_schema
            else:
                schema_json = response_schema.model_json_schema()
            # Extract required fields and their types for a clearer instruction
            properties = schema_json.get("properties", {})
            definitions = schema_json.get("$defs", {})

            def describe_field(field_info: dict, indent: int = 2) -> str:
                """Recursively describe a field including nested objects."""
                prefix = " " * indent
                field_type = field_info.get("type", "string")

                # Handle $ref to definitions
                if "$ref" in field_info:
                    ref_name = field_info["$ref"].split("/")[-1]
                    if ref_name in definitions:
                        return describe_schema(definitions[ref_name], indent)

                # Handle arrays with items
                if field_type == "array" and "items" in field_info:
                    items = field_info["items"]
                    if "$ref" in items:
                        ref_name = items["$ref"].split("/")[-1]
                        if ref_name in definitions:
                            item_desc = describe_schema(definitions[ref_name], indent + 2)
                            return f"array of objects, each with:\n{item_desc}"
                    elif "type" in items:
                        return f"array of {items['type']}"
                    return "array"

                return field_type

            def describe_schema(schema: dict, indent: int = 2) -> str:
                """Describe a schema's fields."""
                props = schema.get("properties", {})
                lines = []
                prefix = " " * indent
                for name, info in props.items():
                    desc = info.get("description", "")
                    field_type = describe_field(info, indent + 2)
                    if desc:
                        lines.append(f'{prefix}"{name}": {field_type} - {desc}')
                    else:
                        lines.append(f'{prefix}"{name}": {field_type}')
                return "\n".join(lines)

            # Build field descriptions with nested schema support
            field_descriptions = []
            for field_name, field_info in properties.items():
                field_desc = field_info.get("description", "")
                field_type_desc = describe_field(field_info, 4)

                if field_desc:
                    field_descriptions.append(f'  "{field_name}": {field_type_desc} - {field_desc}')
                else:
                    field_descriptions.append(f'  "{field_name}": {field_type_desc}')

            fields_str = ",\n".join(field_descriptions)
            schema_instruction = f"""\n\nIMPORTANT: You must respond with ONLY a valid JSON object containing actual values (not a schema definition).
Your response must be a JSON object with these fields:
{{
{fields_str}
}}

Do NOT include "description", "type", "properties" or other schema metadata - return actual values only."""

            # Append to system message or create one
            messages = list(messages)  # Copy to avoid mutation
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + schema_instruction
                }
            else:
                messages = [{"role": "system", "content": schema_instruction}] + messages

        # Check if this model uses the new API format
        # New API: gpt-5.x, o1, o3, o4 models use max_completion_tokens
        def uses_new_api_for_model(model_name: str) -> bool:
            return any(model_name.startswith(prefix) for prefix in ["gpt-5", "o1", "o3", "o4"])

        # Check if this model supports reasoning_effort
        def uses_reasoning_for_model(model_name: str) -> bool:
            return model_name.startswith("gpt-5")

        # Use max_completion_tokens for newer models, max_tokens for older ones
        if uses_new_api_for_model(effective_model):
            # Build kwargs for the API call
            # Note: reasoning tokens count against max_completion_tokens budget
            # With high reasoning, need much larger budget (reasoning can use 1000+ tokens)
            uses_reasoning = uses_reasoning_for_model(effective_model) and effective_reasoning != "none"
            kwargs = {
                "model": effective_model,
                "messages": messages,
                # Need large token budget for complex responses with many variants and feature engineering
                "max_completion_tokens": 16000,  # Model maximum
                "response_format": {"type": "json_object"},
            }
            # Add reasoning_effort for GPT-5.x models (defaults to "none" without this)
            # Note: GPT-5.x with reasoning does NOT support custom temperature (only default 1)
            if uses_reasoning:
                kwargs["reasoning_effort"] = effective_reasoning
                # Don't set temperature - reasoning models only support default (1)
            else:
                kwargs["temperature"] = 0.3  # Lower temperature for more consistent JSON

            async def make_request():
                return await client.chat.completions.create(**kwargs)

            response = await retry_with_exponential_backoff(make_request)
        else:
            async def make_request():
                return await client.chat.completions.create(
                    model=effective_model,
                    messages=messages,
                    # Need large token budget for complex responses with many variants and feature engineering
                    max_tokens=16000,  # Safe default for all models
                    temperature=0.3,  # Lower temperature for more consistent JSON
                    response_format={"type": "json_object"},
                )

            response = await retry_with_exponential_backoff(make_request)

        content = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
            logger.debug(f"LLM JSON response parsed successfully: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content}")
            raise ValueError(f"LLM returned invalid JSON: {e}")

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Send messages with tool definitions and handle tool calls.

        Implements OpenAI function calling API for agent tools.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in our format (converted to OpenAI format)
            response_schema: Optional schema for the final response

        Returns:
            Dict with either 'tool_calls' (list of tool calls to execute)
            or 'content' (final response)
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT)

        # Convert tools to OpenAI function format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            })

        # Add schema instruction if provided
        if response_schema:
            # Handle both Pydantic models and raw dicts
            if isinstance(response_schema, dict):
                schema_json = response_schema
            else:
                schema_json = response_schema.model_json_schema()
            properties = schema_json.get("properties", {})

            field_descriptions = []
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "string")
                field_desc = field_info.get("description", "")
                if field_desc:
                    field_descriptions.append(f'  "{field_name}": {field_type} - {field_desc}')
                else:
                    field_descriptions.append(f'  "{field_name}": {field_type}')

            fields_str = ",\n".join(field_descriptions)
            schema_instruction = f"""

When you are ready to provide your final answer (after using tools), respond with a JSON object:
{{
{fields_str}
}}"""

            # Add to system message
            messages = list(messages)
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + schema_instruction
                }

        # Build API call kwargs
        uses_reasoning = self._uses_reasoning() and self.reasoning_effort != "none"
        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": openai_tools,
            "tool_choice": "auto",  # Let the model decide when to use tools
        }

        if self._uses_new_api():
            # Need large token budget for complex responses like dataset designs with many variants
            # Feature engineering formulas can be very verbose with long pandas expressions
            kwargs["max_completion_tokens"] = 16000  # Model maximum
            if uses_reasoning:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            # Need large token budget for complex responses like dataset designs with many variants
            # Feature engineering formulas can be very verbose with long pandas expressions
            kwargs["max_tokens"] = 16000  # Model maximum
            kwargs["temperature"] = 0.3

        async def make_request():
            return await client.chat.completions.create(**kwargs)

        response = await retry_with_exponential_backoff(make_request)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Log if response was truncated
        if finish_reason == "length":
            logger.warning(f"Response was truncated (finish_reason=length). Content length: {len(message.content or '')}")

        # Check if the model wants to call tools
        if message.tool_calls:
            return {
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in message.tool_calls
                ],
                "finish_reason": finish_reason,
            }

        # No tool calls - return the content as final response
        content = message.content
        if content:
            # Try to parse as JSON if schema was provided
            if response_schema:
                try:
                    return {"content": json.loads(content), "finish_reason": finish_reason}
                except json.JSONDecodeError:
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            return {"content": json.loads(json_match.group()), "finish_reason": finish_reason}
                        except json.JSONDecodeError:
                            pass
            return {"content": content, "finish_reason": finish_reason}

        return {"content": "", "finish_reason": finish_reason}


class O3DeepResearchClient(BaseLLMClient):
    """OpenAI O3 Deep Research client using the Responses API.

    This client uses OpenAI's deep research models (o3-deep-research, o4-mini-deep-research)
    which can search the web and synthesize comprehensive reports.
    """

    def __init__(self, api_key: str, model: str = "o3-deep-research-2025-06-26"):
        """Initialize O3 Deep Research client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: o3-deep-research-2025-06-26)
                   Options: o3-deep-research-2025-06-26, o4-mini-deep-research-2025-06-26
        """
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: List[Dict[str, Any]], images: Optional[List[Dict[str, str]]] = None) -> str:
        """Send messages and get a response using the Responses API.

        Note: Deep research models use web search to gather information.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, timeout=600.0)  # Longer timeout for research

        # Convert messages to responses API format
        input_messages = []
        for msg in messages:
            role = "developer" if msg["role"] == "system" else msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                input_messages.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": content}]
                })
            else:
                input_messages.append({"role": role, "content": content})

        async def make_request():
            return await client.responses.create(
                model=self.model,
                input=input_messages,
                reasoning={"summary": "auto"},
                tools=[
                    {"type": "web_search_preview"},
                ],
            )

        response = await retry_with_exponential_backoff(make_request)

        # Extract the final text from the response
        if response.output:
            for item in reversed(response.output):
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            return content_item.text

        return ""

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send messages and get a structured JSON response.

        Note: Deep research models don't natively support JSON mode,
        so we instruct the model to return JSON and parse the response.

        Args:
            model: Ignored - O3 Deep Research only supports its own model
        """
        # Add JSON instruction to the messages
        if response_schema:
            if isinstance(response_schema, dict):
                schema_json = response_schema
            else:
                schema_json = response_schema.model_json_schema()
            properties = schema_json.get("properties", {})

            field_descriptions = []
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "string")
                field_desc = field_info.get("description", "")
                if field_desc:
                    field_descriptions.append(f'  "{field_name}": {field_type} - {field_desc}')
                else:
                    field_descriptions.append(f'  "{field_name}": {field_type}')

            fields_str = ",\n".join(field_descriptions)
            json_instruction = f"""

IMPORTANT: After your research, provide your final answer as a valid JSON object:
{{
{fields_str}
}}"""

            messages = list(messages)
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + json_instruction
                }
            else:
                messages = [{"role": "system", "content": json_instruction}] + messages

        content = await self.chat(messages)

        # Try to parse JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON response from O3 Deep Research: {content[:500]}")
            raise ValueError(f"LLM returned invalid JSON")


class GeminiClient(BaseLLMClient):
    """Google Gemini chat client."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model

    async def chat(self, messages: List[Dict[str, Any]], images: Optional[List[Dict[str, str]]] = None) -> str:
        import google.generativeai as genai
        import base64

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        # Convert messages to Gemini format
        # Gemini uses a different format, so we need to adapt
        system_prompt = ""
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                chat_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_messages.append({"role": "model", "parts": [msg["content"]]})

        # If there's a system prompt, prepend it to the first user message
        if system_prompt and chat_messages:
            first_user_content = chat_messages[0]["parts"][0]
            chat_messages[0]["parts"][0] = f"{system_prompt}\n\n{first_user_content}"

        # If images provided, add them to the last user message
        if images and chat_messages:
            last_user_parts = chat_messages[-1]["parts"]
            for img in images:
                # Gemini expects image data as bytes
                img_bytes = base64.b64decode(img["base64"])
                last_user_parts.append({
                    "mime_type": "image/png",
                    "data": img_bytes
                })

        # Use chat for multi-turn conversation
        if len(chat_messages) > 1:
            chat = model.start_chat(history=chat_messages[:-1])

            async def make_request():
                return await chat.send_message_async(
                    chat_messages[-1]["parts"],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=20000,
                        temperature=0.7,
                    )
                )

            # Add timeout to prevent hanging
            try:
                async with asyncio.timeout(LLM_REQUEST_TIMEOUT):
                    response = await retry_with_exponential_backoff(make_request)
            except asyncio.TimeoutError:
                logger.error(f"Gemini chat timed out after {LLM_REQUEST_TIMEOUT}s")
                raise TimeoutError(f"Gemini API request timed out after {LLM_REQUEST_TIMEOUT} seconds")
        else:
            # Single message
            content = chat_messages[0]["parts"] if chat_messages else [system_prompt]

            async def make_request():
                return await model.generate_content_async(
                    content,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=20000,
                        temperature=0.7,
                    )
                )

            # Add timeout to prevent hanging
            try:
                async with asyncio.timeout(LLM_REQUEST_TIMEOUT):
                    response = await retry_with_exponential_backoff(make_request)
            except asyncio.TimeoutError:
                logger.error(f"Gemini chat timed out after {LLM_REQUEST_TIMEOUT}s")
                raise TimeoutError(f"Gemini API request timed out after {LLM_REQUEST_TIMEOUT} seconds")

        return response.text.strip()

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send messages and get a structured JSON response.

        Args:
            model: Ignored - Gemini uses the model specified at initialization
        """
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        gemini_model = genai.GenerativeModel(self.model_name)

        # Build prompt with JSON instruction
        system_prompt = ""
        user_content = ""

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]

        json_instruction = "\n\nYou must respond with ONLY valid JSON, no markdown or other text."
        if response_schema:
            # Handle both Pydantic models and raw dicts
            if isinstance(response_schema, dict):
                schema_json = response_schema
            else:
                schema_json = response_schema.model_json_schema()
            # Extract required fields and their types for a clearer instruction
            properties = schema_json.get("properties", {})

            # Build field descriptions
            field_descriptions = []
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "string")
                field_desc = field_info.get("description", "")
                enum_vals = field_info.get("enum", [])

                if enum_vals:
                    field_descriptions.append(f'  "{field_name}": one of {enum_vals}')
                else:
                    field_descriptions.append(f'  "{field_name}": {field_type}' + (f' - {field_desc}' if field_desc else ''))

            fields_str = ",\n".join(field_descriptions)
            json_instruction = f"""\n\nIMPORTANT: You must respond with ONLY a valid JSON object containing actual values (not a schema definition).
Your response must be a JSON object with these fields:
{{
{fields_str}
}}

Do NOT include "description", "type", "properties" or other schema metadata - return actual values only."""

        full_prompt = f"{system_prompt}{json_instruction}\n\n{user_content}"

        async def make_request():
            return await gemini_model.generate_content_async(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=40000,
                    temperature=0.3,
                    response_mime_type="application/json",
                )
            )

        # Add timeout to prevent hanging indefinitely
        try:
            async with asyncio.timeout(LLM_REQUEST_TIMEOUT):
                response = await retry_with_exponential_backoff(make_request)
        except asyncio.TimeoutError:
            logger.error(f"Gemini chat_json timed out after {LLM_REQUEST_TIMEOUT}s")
            raise TimeoutError(f"Gemini API request timed out after {LLM_REQUEST_TIMEOUT} seconds")

        content = response.text.strip()

        # Try to extract JSON from response (in case of markdown wrapping)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON response from Gemini: {content}")
            raise ValueError(f"LLM returned invalid JSON")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude chat client."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: List[Dict[str, Any]], images: Optional[List[Dict[str, str]]] = None) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT)

        # Extract system message if present
        system_prompt = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add images to the last user message if provided
        if images and chat_messages:
            for i in range(len(chat_messages) - 1, -1, -1):
                if chat_messages[i]["role"] == "user":
                    content = chat_messages[i]["content"]
                    if isinstance(content, str):
                        new_content = [{"type": "text", "text": content}]
                    else:
                        new_content = list(content)

                    for img in images:
                        new_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img["base64"]
                            }
                        })
                    chat_messages[i]["content"] = new_content
                    break

        async def make_request():
            kwargs = {
                "model": self.model,
                "max_tokens": 16000,
                "messages": chat_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            return await client.messages.create(**kwargs)

        response = await retry_with_exponential_backoff(make_request)

        # Extract text from response
        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text

        return result_text.strip()

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send messages and get a structured JSON response.

        Args:
            model: Ignored - Anthropic uses the model specified at initialization
        """
        # Add JSON instruction to the last user message
        modified_messages = list(messages)
        json_instruction = "\n\nYou must respond with ONLY valid JSON, no markdown or other text."

        if response_schema:
            if isinstance(response_schema, dict):
                schema_json = response_schema
            else:
                schema_json = response_schema.model_json_schema()
            json_instruction += f"\n\nJSON Schema:\n```json\n{json.dumps(schema_json, indent=2)}\n```"

        # Find the last user message and append JSON instruction
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i]["role"] == "user":
                modified_messages[i] = {
                    "role": "user",
                    "content": modified_messages[i]["content"] + json_instruction
                }
                break

        content = await self.chat(modified_messages)

        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON response from Anthropic: {content}")
            raise ValueError(f"LLM returned invalid JSON")


def get_llm_client(
    provider: Optional[LLMProvider] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    use_app_settings: bool = True
) -> BaseLLMClient:
    """Get the appropriate LLM client for the provider.

    If provider and api_key are not specified, automatically retrieves
    the first available API key from the database.

    Args:
        provider: LLM provider (OpenAI, Gemini, etc.)
        api_key: API key for the provider
        model: Model name (default: from app settings or gpt-5.1)
        reasoning_effort: Reasoning effort for GPT-5.1 models (none, low, medium, high)
                         If not specified, uses value from app settings.
        use_app_settings: Whether to use app settings for model selection (default: True)
                         Set to False for log parsing and other utility calls.
    """
    from app.core.database import SessionLocal
    from app.models.api_key import ApiKey
    from app.models.app_settings import AppSettings, AIModel, AI_MODEL_CONFIG

    db = SessionLocal()
    try:
        # Get model configuration from app settings if not explicitly provided
        selected_model = model
        selected_reasoning = reasoning_effort

        if use_app_settings and (model is None or reasoning_effort is None):
            app_settings = db.query(AppSettings).first()
            if app_settings:
                model_config = AI_MODEL_CONFIG.get(app_settings.ai_model, AI_MODEL_CONFIG[AIModel.GPT_5_1_THINKING])
                if model is None:
                    selected_model = model_config["model"]
                if reasoning_effort is None:
                    selected_reasoning = model_config["reasoning_effort"]

        # Default fallbacks
        if selected_model is None:
            selected_model = "gpt-5.1"
        if selected_reasoning is None:
            selected_reasoning = "high"

        # If provider/api_key not specified, get from database
        if provider is None or api_key is None:
            # Get first available API key (prefer OpenAI if available)
            api_key_record = db.query(ApiKey).filter(ApiKey.provider == LLMProvider.OPENAI).first()
            if not api_key_record:
                api_key_record = db.query(ApiKey).first()

            if not api_key_record:
                raise ValueError("No API key configured. Please add an API key in Settings.")

            provider = api_key_record.provider
            api_key = api_key_record.api_key
    finally:
        db.close()

    # Always decrypt the API key (handles both DB-fetched and caller-provided encrypted keys)
    api_key = decrypt(api_key)

    if provider == LLMProvider.OPENAI:
        # Check if using O3 Deep Research model
        if selected_model and selected_model.startswith("o3-deep-research"):
            return O3DeepResearchClient(api_key, model=selected_model)
        elif selected_model and selected_model.startswith("o4-mini-deep-research"):
            return O3DeepResearchClient(api_key, model=selected_model)
        return OpenAIClient(api_key, model=selected_model, reasoning_effort=selected_reasoning)
    elif provider == LLMProvider.GEMINI:
        return GeminiClient(api_key, model=selected_model if selected_model.startswith("gemini") else "gemini-2.0-flash")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_critique_client(debate_partner_model: str) -> Optional[BaseLLMClient]:
    """Create the appropriate LLM client for a debate partner based on model name.

    Args:
        debate_partner_model: Model name like "gemini-2.0-flash", "claude-sonnet-4",
                             "gpt-4o", or "gpt-5.1"

    Returns:
        Configured LLM client for the debate partner, or None if no API key available.
    """
    from app.core.database import SessionLocal
    from app.models.api_key import ApiKey

    db = SessionLocal()
    try:
        # Determine provider from model name
        if debate_partner_model.startswith("gemini"):
            provider = LLMProvider.GEMINI
        elif debate_partner_model.startswith("claude"):
            provider = LLMProvider.ANTHROPIC
        elif debate_partner_model.startswith("gpt") or debate_partner_model.startswith("o1") or debate_partner_model.startswith("o3"):
            provider = LLMProvider.OPENAI
        else:
            # Default to Gemini for unknown models
            logger.warning(f"Unknown debate partner model: {debate_partner_model}, defaulting to Gemini")
            provider = LLMProvider.GEMINI
            debate_partner_model = "gemini-2.0-flash"

        # Get API key for the provider
        api_key_record = db.query(ApiKey).filter(ApiKey.provider == provider).first()
        if not api_key_record:
            logger.warning(f"No API key found for provider {provider} to create debate partner client")
            return None

        api_key = api_key_record.api_key

        # Always decrypt
        api_key = decrypt(api_key)

        # Create the appropriate client
        if provider == LLMProvider.GEMINI:
            return GeminiClient(api_key, model=debate_partner_model)
        elif provider == LLMProvider.ANTHROPIC:
            # Map debate partner names to actual Anthropic model IDs
            model_mapping = {
                "claude-sonnet-4": "claude-sonnet-4-20250514",
                "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
            }
            actual_model = model_mapping.get(debate_partner_model, debate_partner_model)
            return AnthropicClient(api_key, model=actual_model)
        elif provider == LLMProvider.OPENAI:
            return OpenAIClient(api_key, model=debate_partner_model, reasoning_effort="none")
        else:
            logger.warning(f"Unsupported provider for debate partner: {provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to create critique client for {debate_partner_model}: {e}")
        return None
    finally:
        db.close()
