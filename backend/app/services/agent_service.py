"""Agent service for LLM-powered ML configuration suggestions."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.schemas.agent import (
    ColumnSummary,
    SchemaSummary,
    ProjectConfigSuggestion,
    DatasetSpecSuggestion,
    DatasetDesignSuggestion,
    DatasetVariant,
    ExperimentPlanSuggestion,
    ExperimentVariant,
)
from app.services.llm_client import BaseLLMClient
from app.services.prompts import (
    SYSTEM_ROLE_ML_ENGINEER,
    SYSTEM_ROLE_PROJECT_CONFIG,
    SYSTEM_ROLE_FEATURE_SELECTION,
    SYSTEM_ROLE_EXPERIMENT_PLAN,
    SYSTEM_ROLE_GOAL_EXPANDER,
    get_project_config_prompt,
    get_feature_selection_prompt,
    get_experiment_plan_prompt,
    get_dataset_design_prompt,
    get_dataset_design_system_prompt,
    get_goal_expansion_prompt,
)
from app.services.agent_tools import (
    AGENT_HISTORY_TOOLS,
    AgentToolExecutor,
    get_tools_prompt_section,
)

logger = logging.getLogger(__name__)

# Maximum retry attempts for validation failures
MAX_RETRIES = 2

# Maximum tool call iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 10

# Maximum empty response retries before giving up
MAX_EMPTY_RETRIES = 3

# Timeout for individual LLM API calls (in seconds)
# Increased to accommodate larger token limits for complex responses (updated)
LLM_CALL_TIMEOUT = 1800  # 30 minutes per call


async def execute_with_tools(
    client: BaseLLMClient,
    messages: List[Dict[str, str]],
    tool_executor: AgentToolExecutor,
    response_schema: Any,
    step_logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute an LLM call with tool support, handling tool calls in a loop.

    This function implements a ReAct-style loop where the LLM can make tool calls
    to query project history before generating its final response.

    Args:
        client: LLM client instance
        messages: Initial conversation messages
        tool_executor: AgentToolExecutor instance for handling tool calls
        response_schema: Pydantic schema for the final response
        step_logger: Optional StepLogger for logging tool calls and results

    Returns:
        The parsed JSON response from the LLM
    """
    iteration = 0
    empty_response_count = 0
    current_messages = messages.copy()

    logger.info(f"Starting execute_with_tools loop (max iterations: {MAX_TOOL_ITERATIONS})")

    while iteration < MAX_TOOL_ITERATIONS:
        iteration += 1
        logger.info(f"execute_with_tools iteration {iteration}/{MAX_TOOL_ITERATIONS} (empty_retries: {empty_response_count})")

        # Make the LLM call with tools (with timeout)
        try:
            response = await asyncio.wait_for(
                client.chat_with_tools(
                    messages=current_messages,
                    tools=AGENT_HISTORY_TOOLS,
                    response_schema=response_schema,
                ),
                timeout=LLM_CALL_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM API call timed out after {LLM_CALL_TIMEOUT} seconds on iteration {iteration}")
            raise ValueError(f"LLM API call timed out after {LLM_CALL_TIMEOUT} seconds. The service may be unavailable.")
        except AttributeError:
            # If the client doesn't support chat_with_tools, fall back to regular chat_json
            logger.warning("LLM client doesn't support tool calling, falling back to chat_json")
            return await client.chat_json(current_messages, response_schema)

        # Check if response contains tool calls
        if response.get("tool_calls"):
            tool_calls = response["tool_calls"]

            if step_logger:
                step_logger.thinking(f"Agent requesting {len(tool_calls)} tool call(s) to query history...")

            # Execute each tool call
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", tool_call.get("function", {}).get("name"))
                tool_args = tool_call.get("arguments", tool_call.get("function", {}).get("arguments", {}))
                tool_id = tool_call.get("id", f"tool_{iteration}_{len(tool_results)}")

                # Parse arguments if they're a string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                if step_logger:
                    step_logger.action(f"Calling tool: {tool_name}({json.dumps(tool_args, default=str)[:200]}...)")

                # Execute the tool
                result = tool_executor.execute_tool(tool_name, tool_args)

                if step_logger:
                    result_preview = json.dumps(result, default=str)[:500]
                    step_logger.thought(f"Tool result: {result_preview}...")

                tool_results.append({
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "result": result,
                })

            # Add tool results to messages for next iteration
            # Format tool_calls for OpenAI API (requires type and nested function object)
            formatted_tool_calls = []
            for tc in tool_calls:
                formatted_tool_calls.append({
                    "id": tc.get("id", tc.get("tool_call_id", f"call_{len(formatted_tool_calls)}")),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", tc.get("function", {}).get("name", "")),
                        "arguments": tc.get("arguments", tc.get("function", {}).get("arguments", "{}")),
                    }
                })

            current_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": formatted_tool_calls,
            })

            for tool_result in tool_results:
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "name": tool_result["name"],
                    "content": json.dumps(tool_result["result"], default=str),
                })

            # Continue loop to get next response
            continue

        # No tool calls - this should be the final response
        finish_reason = response.get("finish_reason")

        # Handle truncated responses (finish_reason="length")
        if finish_reason == "length":
            empty_response_count += 1
            logger.warning(f"Response was truncated (finish_reason=length), asking for concise response (attempt {empty_response_count})")

            if empty_response_count >= MAX_EMPTY_RETRIES:
                raise ValueError("Response keeps getting truncated. The response may be too complex. Try simplifying the request.")

            if step_logger:
                step_logger.warning(f"Response was truncated, asking for more concise output...")
            current_messages.append({
                "role": "user",
                "content": "Your response was too long and got cut off. Please provide a MORE CONCISE response. "
                          "Focus on the most important variants (3-5 instead of 10). Keep descriptions brief. "
                          "The response MUST be complete, valid JSON.",
            })
            continue  # Loop back to get a shorter response

        if "content" in response:
            # Parse the final JSON response
            content = response["content"]

            # Handle empty responses - prompt the LLM to provide final answer
            if not content or (isinstance(content, str) and not content.strip()):
                empty_response_count += 1
                logger.warning(f"LLM returned empty response (attempt {empty_response_count}/{MAX_EMPTY_RETRIES})")

                if empty_response_count >= MAX_EMPTY_RETRIES:
                    raise ValueError(f"LLM returned empty response {MAX_EMPTY_RETRIES} times in a row. Unable to get valid response.")

                if step_logger:
                    step_logger.thinking(f"LLM returned empty response (attempt {empty_response_count}), prompting for final answer...")
                current_messages.append({
                    "role": "user",
                    "content": "Please provide your final answer as a JSON object based on the tool results above. The response must not be empty.",
                })
                continue  # Loop back to get proper response

            if isinstance(content, str):
                # Detect truncated JSON by checking if it looks incomplete
                # Signs of truncation: doesn't end with }, ends mid-string, or has unbalanced brackets
                content_stripped = content.strip()
                looks_truncated = False
                if content_stripped.startswith('{'):
                    # Check for obvious truncation signs
                    if not content_stripped.endswith('}'):
                        looks_truncated = True
                    # Check for unbalanced brackets (rough check)
                    open_braces = content_stripped.count('{')
                    close_braces = content_stripped.count('}')
                    if open_braces > close_braces:
                        looks_truncated = True

                if looks_truncated:
                    empty_response_count += 1
                    logger.warning(f"Response appears truncated (incomplete JSON), asking for concise response (attempt {empty_response_count})")

                    if empty_response_count >= MAX_EMPTY_RETRIES:
                        raise ValueError("Response keeps getting truncated. The response may be too complex. Try simplifying the request.")

                    if step_logger:
                        step_logger.warning(f"Response was truncated (incomplete JSON), asking for more concise output...")
                    current_messages.append({
                        "role": "user",
                        "content": "Your response was too long and got cut off mid-JSON. Please provide a SHORTER, MORE CONCISE response. "
                                  "Focus on the most important variants (3-5 instead of 10). Keep feature engineering formulas BRIEF. "
                                  "Use simple column names. The response MUST be complete, valid JSON ending with }.",
                    })
                    continue  # Loop back to get a shorter response

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    # Try multiple repair strategies for truncated/malformed JSON
                    import re

                    # Strategy 1: Try to repair the entire content first (handles truncation)
                    try:
                        repaired_full = _repair_llm_json(content)
                        result = json.loads(repaired_full)
                        logger.info("JSON parsing succeeded after repairing full content")
                        return result
                    except json.JSONDecodeError as repair_error:
                        logger.warning(f"JSON repair strategy 1 failed: {repair_error}")

                    # Strategy 2: Extract JSON starting from first { to end, then repair
                    first_brace = content.find('{')
                    if first_brace >= 0:
                        json_candidate = content[first_brace:]
                        try:
                            repaired = _repair_llm_json(json_candidate)
                            result = json.loads(repaired)
                            logger.info("JSON parsing succeeded after extracting and repairing")
                            return result
                        except json.JSONDecodeError as repair_error:
                            logger.warning(f"JSON repair strategy 2 failed: {repair_error}")

                    # Strategy 3: Try regex extraction (might find balanced inner object)
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        json_str = json_match.group()
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Try to repair common JSON issues from LLMs
                            repaired = _repair_llm_json(json_str)
                            try:
                                return json.loads(repaired)
                            except json.JSONDecodeError as repair_error:
                                logger.warning(f"JSON repair strategy 3 failed: {repair_error}")

                    # Strategy 4: For "variants" responses, try to extract complete variant objects
                    if '"variants"' in content:
                        try:
                            extracted = _extract_complete_variants(content)
                            if extracted:
                                logger.info(f"JSON parsing succeeded with {len(extracted.get('variants', []))} extracted variants")
                                return extracted
                        except Exception as extract_error:
                            logger.warning(f"JSON repair strategy 4 (variant extraction) failed: {extract_error}")

                    logger.error(f"All JSON repair strategies failed. Original content (first 1000 chars): {content[:1000]}")
                    raise ValueError(f"Could not parse JSON from response: {content[:500]}")
            return content

        # Direct JSON response (already parsed dict)
        if isinstance(response, dict) and response:
            # Check if it's a valid response dict (not just {"content": ""})
            return response

        # Empty response - prompt for final answer
        empty_response_count += 1
        logger.warning(f"LLM returned empty/invalid response (attempt {empty_response_count}/{MAX_EMPTY_RETRIES})")

        if empty_response_count >= MAX_EMPTY_RETRIES:
            raise ValueError(f"LLM returned empty/invalid response {MAX_EMPTY_RETRIES} times in a row. Unable to get valid response.")

        if step_logger:
            step_logger.thinking(f"LLM returned empty response (attempt {empty_response_count}), prompting for final answer...")
        current_messages.append({
            "role": "user",
            "content": "Please provide your final answer as a JSON object. The response must not be empty.",
        })
        continue  # Loop back

    raise ValueError(f"Tool execution exceeded maximum iterations ({MAX_TOOL_ITERATIONS}). This usually indicates the LLM is unable to produce a valid response.")


def _complete_truncated_json(json_str: str) -> str:
    """Attempt to complete truncated JSON by closing open brackets/braces.

    When an LLM response gets cut off mid-JSON, this function tries to
    salvage what we have by closing any open structures.

    Args:
        json_str: Potentially truncated JSON string

    Returns:
        JSON string with balanced brackets/braces
    """
    import re

    # Count open structures (accounting for strings)
    in_string = False
    escape_next = False
    open_braces = 0
    open_brackets = 0
    stack = []  # Track what needs to be closed and in what order

    for i, char in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            open_braces += 1
            stack.append('}')
        elif char == '}':
            open_braces -= 1
            if stack and stack[-1] == '}':
                stack.pop()
        elif char == '[':
            open_brackets += 1
            stack.append(']')
        elif char == ']':
            open_brackets -= 1
            if stack and stack[-1] == ']':
                stack.pop()

    # If balanced, nothing to fix
    if open_braces == 0 and open_brackets == 0 and not in_string:
        return json_str

    # Try to find a good truncation point
    # Look for the last complete object/array item
    result = json_str.rstrip()

    # Close open string if necessary
    if in_string:
        result += '"'

    # Remove trailing incomplete parts (like partial property names or values)
    # Look for patterns like: , "partial or : "partial or : partial
    result = re.sub(r',\s*"[^"]*$', '', result)  # Remove trailing ", "partial...
    result = re.sub(r':\s*"[^"]*$', ': null', result)  # Fix : "partial... -> : null
    result = re.sub(r':\s*[^,\[\]{}"\s]+$', ': null', result)  # Fix : partial... -> : null
    result = re.sub(r',\s*$', '', result)  # Remove trailing comma

    # Close remaining open structures in reverse order
    for closer in reversed(stack):
        result += closer

    return result


def _repair_llm_json(json_str: str) -> str:
    """Repair common JSON syntax errors from LLM responses.

    LLMs often produce invalid JSON with issues like:
    - Trailing commas before closing braces/brackets
    - Comments (// or /* */)
    - Single quotes instead of double quotes
    - Unquoted property names
    - Truncated responses (incomplete JSON)
    - Invalid control characters inside strings

    This function attempts to fix these issues before parsing.
    """
    import re

    # First, sanitize invalid control characters that can appear in LLM output
    # JSON spec allows \n, \r, \t etc. but only when properly escaped inside strings
    # We need to escape raw control chars (0x00-0x1F except \t, \n, \r) that break parsing
    def escape_control_chars(s: str) -> str:
        """Escape invalid control characters in a string."""
        result = []
        for char in s:
            code = ord(char)
            # Control characters 0x00-0x1F except tab(0x09), newline(0x0A), carriage return(0x0D)
            if code < 0x20 and code not in (0x09, 0x0A, 0x0D):
                result.append(f'\\u{code:04x}')
            else:
                result.append(char)
        return ''.join(result)

    json_str = escape_control_chars(json_str)

    # First, try to complete truncated JSON
    json_str = _complete_truncated_json(json_str)

    # Remove JavaScript-style comments
    # Single-line comments
    json_str = re.sub(r'//[^\n]*', '', json_str)
    # Multi-line comments
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)

    # Fix trailing commas before closing braces or brackets
    # This handles: [1, 2, 3,] or {"a": 1,}
    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

    # Fix single quotes to double quotes (but not within already double-quoted strings)
    # This is tricky - we need to be careful not to break strings that contain apostrophes
    # Simple approach: replace single quotes that look like JSON string delimiters
    # Pattern: 'word' at property name or value position
    def fix_quotes(match):
        content = match.group(1)
        # If content has double quotes, don't change it (already valid or complex)
        if '"' in content:
            return match.group(0)
        return f'"{content}"'

    # Replace 'string' patterns (simple cases)
    json_str = re.sub(r"'([^']*)'", fix_quotes, json_str)

    # Try to fix unquoted property names (common in JS but invalid in JSON)
    # Pattern: { name: or , name: where name is alphanumeric
    json_str = re.sub(r'([\{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)

    return json_str


def _extract_complete_variants(content: str) -> Optional[Dict[str, Any]]:
    """Extract complete variant objects from a truncated variants response.

    When an LLM response with multiple variants gets truncated, this tries
    to salvage whatever complete variants we can find.

    Args:
        content: The potentially truncated JSON content

    Returns:
        A dict with "variants" list containing complete variant objects,
        or None if extraction fails
    """
    import re

    # Sanitize control characters first
    def escape_control_chars(s: str) -> str:
        result = []
        for char in s:
            code = ord(char)
            if code < 0x20 and code not in (0x09, 0x0A, 0x0D):
                result.append(f'\\u{code:04x}')
            else:
                result.append(char)
        return ''.join(result)

    content = escape_control_chars(content)

    # Find the start of the variants array
    variants_match = re.search(r'"variants"\s*:\s*\[', content)
    if not variants_match:
        return None

    array_start = variants_match.end()
    complete_variants = []

    # Track brace depth to find complete objects
    i = array_start
    object_start = None
    depth = 0
    in_string = False
    escape_next = False

    while i < len(content):
        char = content[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            escape_next = True
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if char == '{':
            if depth == 0:
                object_start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and object_start is not None:
                # Found a complete object
                obj_str = content[object_start:i+1]
                try:
                    variant = json.loads(obj_str)
                    # Validate it looks like a variant (has name at minimum)
                    if isinstance(variant, dict) and 'name' in variant:
                        complete_variants.append(variant)
                except json.JSONDecodeError:
                    # Try to repair this individual object
                    try:
                        repaired = _repair_llm_json(obj_str)
                        variant = json.loads(repaired)
                        if isinstance(variant, dict) and 'name' in variant:
                            complete_variants.append(variant)
                    except json.JSONDecodeError:
                        pass  # Skip this malformed variant
                object_start = None
        elif char == ']' and depth == 0:
            # End of variants array
            break

        i += 1

    if complete_variants:
        logger.info(f"Extracted {len(complete_variants)} complete variants from truncated response")
        return {"variants": complete_variants}

    return None


def _fix_llm_json_issues(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common JSON issues from LLM responses.

    LLMs sometimes return invalid JSON values like:
    - "None" (string) instead of null
    - String descriptions where dicts are expected

    This function cleans up the data before Pydantic validation.
    """
    if not isinstance(data, dict):
        return data

    # Fix variants list if present
    if "variants" in data and isinstance(data["variants"], list):
        for variant in data["variants"]:
            if isinstance(variant, dict):
                # Fix suggested_filters: convert string "None" to actual None
                if "suggested_filters" in variant:
                    val = variant["suggested_filters"]
                    if val is None or val == "None" or val == "null" or val == "":
                        variant["suggested_filters"] = None
                    elif isinstance(val, str):
                        # LLM returned a description string instead of a dict
                        # Preserve the intent by logging and storing as a description
                        logger.warning(
                            f"Variant '{variant.get('name', 'unknown')}' has suggested_filters as string: "
                            f"'{val[:100]}{'...' if len(val) > 100 else ''}'. "
                            "Converting to dict with description. Consider fixing the prompt to return proper filter format."
                        )
                        # Store the string as a description for reference - downstream code should handle this
                        variant["suggested_filters"] = {"description": val, "_type": "llm_string"}

                # Ensure engineered_features is a list
                if "engineered_features" not in variant:
                    variant["engineered_features"] = []
                elif variant["engineered_features"] is None:
                    variant["engineered_features"] = []

    # Fix target_creation if present
    if "target_creation" in data:
        val = data["target_creation"]
        if val == "None" or val == "null" or val == "":
            data["target_creation"] = None
        elif isinstance(val, str) and val.strip():
            # LLM returned a formula string instead of a proper object
            # This is an error we can't auto-fix, but we'll let validation catch it
            pass

    return data


def build_schema_summary(
    data_source_id: str,
    data_source_name: str,
    analysis_result: Dict[str, Any],
) -> SchemaSummary:
    """Build a schema summary from analysis results.

    Args:
        data_source_id: UUID of the data source
        data_source_name: Name of the data source
        analysis_result: Output from SchemaAnalyzer

    Returns:
        SchemaSummary for LLM context
    """
    columns = []
    for col in analysis_result.get("columns", []):
        inferred_type = col.get("inferred_type", "unknown")

        # Handle min/max based on column type
        # For numeric columns, use float min/max
        # For datetime columns, convert to string and use min_date/max_date
        col_min = col.get("min")
        col_max = col.get("max")
        col_mean = col.get("mean")
        min_date = None
        max_date = None

        if inferred_type == "datetime":
            # Datetime columns: store as string in min_date/max_date
            if col_min is not None:
                min_date = str(col_min)
                col_min = None  # Don't pass to float field
            if col_max is not None:
                max_date = str(col_max)
                col_max = None  # Don't pass to float field
            col_mean = None  # Mean doesn't apply to datetime
        elif inferred_type == "numeric":
            # Numeric columns: ensure they're floats
            try:
                if col_min is not None:
                    col_min = float(col_min)
                if col_max is not None:
                    col_max = float(col_max)
                if col_mean is not None:
                    col_mean = float(col_mean)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert min/max/mean to float for column {col['name']}: {e}")
                col_min = None
                col_max = None
                col_mean = None
        else:
            # For other types (categorical, text, boolean), don't include numeric stats
            col_min = None
            col_max = None
            col_mean = None

        col_summary = ColumnSummary(
            name=col["name"],
            dtype=col["dtype"],
            inferred_type=inferred_type,
            null_percentage=col.get("null_percentage", 0.0),
            unique_count=col.get("unique_count", 0),
            min=col_min,
            max=col_max,
            mean=col_mean,
            top_values=col.get("top_values"),
            min_date=min_date,
            max_date=max_date,
        )
        columns.append(col_summary)

    return SchemaSummary(
        data_source_id=data_source_id,
        data_source_name=data_source_name,
        file_type=analysis_result.get("file_type", "unknown"),
        row_count=analysis_result.get("row_count", 0),
        column_count=analysis_result.get("column_count", 0),
        columns=columns,
    )


def _format_schema_for_prompt(schema: SchemaSummary) -> str:
    """Format schema summary as a string for LLM prompt."""
    lines = [
        f"Dataset: {schema.data_source_name}",
        f"File type: {schema.file_type}",
        f"Rows: {schema.row_count:,}",
        f"Columns: {schema.column_count}",
        "",
        "Column details:",
    ]

    for col in schema.columns:
        col_line = f"  - {col.name} ({col.dtype}, {col.inferred_type})"
        col_line += f" - {col.null_percentage:.1f}% null, {col.unique_count} unique"

        if col.inferred_type == "numeric" and col.min is not None:
            mean_str = f", mean: {col.mean:.2f}" if col.mean is not None else ""
            col_line += f" - range: [{col.min:.2f}, {col.max:.2f}]{mean_str}"
        elif col.inferred_type == "datetime" and col.min_date is not None:
            col_line += f" - date range: [{col.min_date}, {col.max_date}]"
        elif col.inferred_type == "categorical" and col.top_values:
            top_3 = list(col.top_values.keys())[:3]
            col_line += f" - top values: {', '.join(repr(v) for v in top_3)}"

        lines.append(col_line)

    return "\n".join(lines)


async def expand_user_goal(
    client: BaseLLMClient,
    user_description: str,
    schema_summary: Optional[SchemaSummary] = None,
) -> str:
    """Expand user's brief ML goal into a comprehensive problem statement.

    Uses GPT-5.1 thinking mode to deeply analyze the user's brief description
    and expand it into a detailed problem statement that guides downstream agents.

    Args:
        client: LLM client instance (should be configured with reasoning_effort="high")
        user_description: The user's original brief description
        schema_summary: Optional schema summary of the data source

    Returns:
        Expanded description with comprehensive ML problem statement
    """
    # Format schema if provided
    schema_text = None
    if schema_summary:
        schema_text = _format_schema_for_prompt(schema_summary)

    # Get the expansion prompt
    user_prompt = get_goal_expansion_prompt(
        user_description=user_description,
        schema_summary=schema_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_GOAL_EXPANDER},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Use regular chat (not chat_json) since we want free-form text
        expanded_description = await client.chat(messages)

        # Prepend the original description for reference
        full_description = f"""## Original User Goal
{user_description}

## Expanded Problem Analysis
{expanded_description}"""

        logger.info(f"Expanded user goal from {len(user_description)} to {len(full_description)} chars")
        return full_description

    except Exception as e:
        logger.warning(f"Goal expansion failed, using original description: {e}")
        # Fall back to original description if expansion fails
        return user_description


async def generate_project_config(
    client: BaseLLMClient,
    description: str,
    schema_summary: SchemaSummary,
) -> ProjectConfigSuggestion:
    """Generate project configuration using LLM.

    Args:
        client: LLM client instance
        description: User's description of the ML goal
        schema_summary: Schema summary of the data source

    Returns:
        ProjectConfigSuggestion with task_type, target_column, metric
    """
    schema_text = _format_schema_for_prompt(schema_summary)

    # Use centralized prompt from prompts.py
    user_prompt = get_project_config_prompt(
        goal_description=description,
        schema_text=schema_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_PROJECT_CONFIG},
        {"role": "user", "content": user_prompt},
    ]

    # Try with validation and retry
    last_error = None
    column_names = [c.name for c in schema_summary.columns]

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await client.chat_json(messages, ProjectConfigSuggestion)

            # Pre-process result to fix common LLM JSON issues
            result = _fix_llm_json_issues(result)

            # Validate the result
            suggestion = ProjectConfigSuggestion(**result)

            # Check if target column exists OR will be created
            target_exists_in_data = suggestion.target_column in column_names

            if target_exists_in_data:
                # Target exists, make sure target_exists is True
                suggestion.target_exists = True
            else:
                # Target doesn't exist - check if we have creation instructions
                if suggestion.target_creation is not None:
                    # Validate source columns exist
                    missing_sources = [
                        col for col in suggestion.target_creation.source_columns
                        if col not in column_names
                    ]
                    if missing_sources:
                        raise ValueError(
                            f"Target creation uses non-existent columns: {missing_sources}. "
                            f"Available columns: {column_names}"
                        )
                    suggestion.target_exists = False
                    logger.info(
                        f"Target '{suggestion.target_column}' will be created using formula: "
                        f"{suggestion.target_creation.formula}"
                    )
                else:
                    # Target doesn't exist and no creation instructions - ask LLM to fix
                    raise ValueError(
                        f"Target column '{suggestion.target_column}' not found in schema. "
                        f"Available columns: {column_names}. "
                        f"If you want to CREATE a target, set target_exists=false and provide "
                        f"target_creation with formula, source_columns, and description."
                    )

            # Validate feature engineering source columns if any
            # Build list of columns available for engineering (original + previously defined engineered)
            engineered_column_names = []
            for fe in suggestion.suggested_feature_engineering:
                # Source columns can be from original data OR previously defined engineered features
                available_for_engineering = column_names + engineered_column_names
                missing_sources = [col for col in fe.source_columns if col not in available_for_engineering]
                if missing_sources:
                    raise ValueError(
                        f"Feature '{fe.output_column}' uses non-existent columns: {missing_sources}. "
                        f"Available: {column_names}. Previously engineered: {engineered_column_names}"
                    )
                # Add this feature to available columns for subsequent features
                engineered_column_names.append(fe.output_column)

            return suggestion

        except (ValidationError, ValueError) as e:
            last_error = e
            logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")

            if attempt < MAX_RETRIES:
                # Add error feedback for retry
                messages.append({
                    "role": "assistant",
                    "content": str(result) if 'result' in dir() else "{}"
                })
                messages.append({
                    "role": "user",
                    "content": f"Your response had an error: {e}\n\nPlease fix and try again."
                })

    raise ValueError(f"Failed to generate valid project config after {MAX_RETRIES + 1} attempts: {last_error}")


async def generate_dataset_spec(
    client: BaseLLMClient,
    schema_summary: SchemaSummary,
    task_type: str,
    target_column: str,
    description: Optional[str] = None,
) -> DatasetSpecSuggestion:
    """Generate dataset specification using LLM.

    Args:
        client: LLM client instance
        schema_summary: Schema summary of the data source
        task_type: The ML task type
        target_column: The target column to predict
        description: Optional additional context

    Returns:
        DatasetSpecSuggestion with feature selection
    """
    schema_text = _format_schema_for_prompt(schema_summary)

    # Use centralized prompt from prompts.py
    user_prompt = get_feature_selection_prompt(
        schema_text=schema_text,
        target_column=target_column,
        task_type=task_type,
        goal_description=description,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_FEATURE_SELECTION},
        {"role": "user", "content": user_prompt},
    ]

    column_names = [c.name for c in schema_summary.columns]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await client.chat_json(messages, DatasetSpecSuggestion)

            # Validate the result
            suggestion = DatasetSpecSuggestion(**result)

            # Validate all feature columns exist
            invalid_features = [f for f in suggestion.feature_columns if f not in column_names]
            if invalid_features:
                raise ValueError(
                    f"Invalid feature columns: {invalid_features}. "
                    f"Available columns: {column_names}"
                )

            # Ensure target is not in features
            if target_column in suggestion.feature_columns:
                suggestion.feature_columns.remove(target_column)
                if target_column not in suggestion.excluded_columns:
                    suggestion.excluded_columns.append(target_column)
                    suggestion.exclusion_reasons[target_column] = "Target column"

            return suggestion

        except (ValidationError, ValueError) as e:
            last_error = e
            logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")

            if attempt < MAX_RETRIES:
                messages.append({
                    "role": "assistant",
                    "content": str(result) if 'result' in dir() else "{}"
                })
                messages.append({
                    "role": "user",
                    "content": f"Your response had an error: {e}\n\nPlease fix and try again."
                })

    raise ValueError(f"Failed to generate valid dataset spec after {MAX_RETRIES + 1} attempts: {last_error}")


async def generate_experiment_plan(
    client: BaseLLMClient,
    task_type: str,
    target_column: str,
    primary_metric: str,
    feature_columns: List[str],
    row_count: int,
    time_budget_minutes: Optional[int] = None,
    description: Optional[str] = None,
    target_stats: Optional[Dict[str, Any]] = None,
    project_history_context: Optional[str] = None,
    # Time-based task metadata
    is_time_based: bool = False,
    time_column: Optional[str] = None,
    entity_id_column: Optional[str] = None,
) -> ExperimentPlanSuggestion:
    """Generate experiment plan with multiple variants using LLM.

    Args:
        client: LLM client instance
        task_type: The ML task type
        target_column: The target column
        primary_metric: The metric to optimize
        feature_columns: Selected feature columns
        row_count: Number of rows in dataset
        time_budget_minutes: Optional time constraint
        description: Optional additional context
        target_stats: Optional target variable statistics for baseline comparison
        project_history_context: Optional formatted string with project history (research cycles,
            previous experiments, robustness findings, etc.) to inform experiment design
        is_time_based: Whether this is a time-based prediction task
        time_column: DateTime column for time-based splits
        entity_id_column: Entity ID column for panel/longitudinal data

    Returns:
        ExperimentPlanSuggestion with multiple variants
    """
    # Use centralized prompt from prompts.py
    user_prompt = get_experiment_plan_prompt(
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        feature_count=len(feature_columns),
        row_count=row_count,
        time_budget_minutes=time_budget_minutes,
        description=description,
        feature_columns=feature_columns,  # Pass column names for time-series detection
        target_stats=target_stats,  # Pass target stats for baseline comparison context
        project_history_context=project_history_context,  # Pass project history for informed design
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_EXPERIMENT_PLAN},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await client.chat_json(messages, ExperimentPlanSuggestion)

            # Validate the result
            suggestion = ExperimentPlanSuggestion(**result)

            # Validate recommended variant exists
            variant_names = [v.name for v in suggestion.variants]
            if suggestion.recommended_variant not in variant_names:
                raise ValueError(
                    f"Recommended variant '{suggestion.recommended_variant}' not in variants: {variant_names}"
                )

            # Ensure quick_test variant exists (mandatory for every experiment plan)
            if "quick_test" not in variant_names:
                logger.warning("Missing mandatory quick_test variant, inserting one at the beginning")
                from app.schemas.agent import ExperimentVariant, ValidationStrategy

                # Determine split strategy based on time-based metadata
                if is_time_based:
                    if entity_id_column:
                        split_type = "group_time"
                        reasoning = f"Group-time split for time-based panel data (time_column={time_column}, entity={entity_id_column})"
                    else:
                        split_type = "time"
                        reasoning = f"Time-based split using '{time_column}' column to prevent future data leakage"
                else:
                    split_type = "stratified" if task_type in ("binary", "multiclass") else "random"
                    reasoning = f"{'Stratified' if split_type == 'stratified' else 'Random'} split for non-temporal data"

                quick_test = ExperimentVariant(
                    name="quick_test",
                    description="SANITY CHECK ONLY - verify data pipeline works. Do NOT use these results for production decisions!",
                    automl_config={
                        "time_limit": 180,  # 3 minutes in seconds
                        "presets": "medium_quality",
                        "num_stack_levels": 0,  # No stacking for quick test
                    },
                    expected_tradeoff="Very fast but minimal model exploration - ONLY for validation, not production!",
                    validation_strategy=ValidationStrategy(
                        split_strategy=split_type,
                        validation_split=0.2,
                        time_column=time_column if is_time_based else None,
                        entity_id_column=entity_id_column if is_time_based and entity_id_column else None,
                        group_column=None,
                        reasoning=reasoning
                    ),
                )
                suggestion.variants.insert(0, quick_test)
                variant_names = [v.name for v in suggestion.variants]

            # Normalize automl_config: convert max_runtime_seconds to time_limit if needed
            for variant in suggestion.variants:
                config = variant.automl_config
                if "max_runtime_seconds" in config and "time_limit" not in config:
                    config["time_limit"] = config.pop("max_runtime_seconds")
                    logger.info(f"Converted max_runtime_seconds to time_limit for variant '{variant.name}'")

            # Validate each variant has required config
            for variant in suggestion.variants:
                config = variant.automl_config
                if "time_limit" not in config:
                    config["time_limit"] = 300  # Default 5 minutes
                if "presets" not in config:
                    config["presets"] = "medium_quality"

                # Ensure validation_strategy is present (set default if missing)
                if variant.validation_strategy is None:
                    from app.schemas.agent import ValidationStrategy

                    # For time-based tasks, MUST use time-based splits
                    if is_time_based:
                        if entity_id_column:
                            default_strategy = "group_time"
                            reasoning = f"Time-based panel data requires group_time split (time={time_column}, entity={entity_id_column})"
                        else:
                            default_strategy = "time"
                            reasoning = f"Time-based task requires time split using '{time_column}'"
                    else:
                        # Non-time-based: use stratified for classification, random for others
                        default_strategy = "stratified" if task_type in ("binary", "multiclass") else "random"
                        reasoning = f"Default {default_strategy} split for non-temporal data"

                    logger.warning(
                        f"Variant '{variant.name}' missing validation_strategy, defaulting to '{default_strategy}'"
                    )
                    variant.validation_strategy = ValidationStrategy(
                        split_strategy=default_strategy,
                        validation_split=0.2,
                        time_column=time_column if is_time_based else None,
                        entity_id_column=entity_id_column if is_time_based and entity_id_column else None,
                        group_column=None,
                        reasoning=reasoning
                    )
                else:
                    # Log if using non-time-based splits on time-based data
                    # (Critic will review and challenge if needed - no hard override)
                    if is_time_based:
                        vs = variant.validation_strategy
                        if vs.split_strategy in ("random", "stratified", "group_random"):
                            logger.info(
                                f"Variant '{variant.name}' uses '{vs.split_strategy}' split for time-based task. "
                                f"Plan Critic will review this choice."
                            )
                            # Ensure time_column info is available for Critic review
                            if not vs.time_column and time_column:
                                vs.time_column = time_column
                            if not vs.entity_id_column and entity_id_column:
                                vs.entity_id_column = entity_id_column

            return suggestion

        except (ValidationError, ValueError) as e:
            last_error = e
            logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")

            if attempt < MAX_RETRIES:
                messages.append({
                    "role": "assistant",
                    "content": str(result) if 'result' in dir() else "{}"
                })
                messages.append({
                    "role": "user",
                    "content": f"Your response had an error: {e}\n\nPlease fix and try again."
                })

    raise ValueError(f"Failed to generate valid experiment plan after {MAX_RETRIES + 1} attempts: {last_error}")


async def generate_dataset_design(
    client: BaseLLMClient,
    schema_summary: SchemaSummary,
    task_type: str,
    target_column: str,
    description: Optional[str] = None,
    max_variants: int = 10,
    project_history_context: Optional[str] = None,
    context_documents: str = "",
) -> DatasetDesignSuggestion:
    """Generate multiple dataset configuration variants using LLM.

    Args:
        client: LLM client instance
        schema_summary: Schema summary of the data source
        task_type: The ML task type
        target_column: The target column to predict
        description: Optional additional context
        max_variants: Maximum number of variants to generate (1-10)
        project_history_context: Optional formatted string with project history (research cycles,
            previous experiments, robustness findings, etc.) to inform design decisions
        context_documents: Optional formatted context documents section

    Returns:
        DatasetDesignSuggestion with multiple dataset variants
    """
    schema_text = _format_schema_for_prompt(schema_summary)

    # Use centralized prompts from prompts.py
    system_prompt = get_dataset_design_system_prompt(max_variants)
    user_prompt = get_dataset_design_prompt(
        schema_text=schema_text,
        task_type=task_type,
        target_column=target_column,
        description=description,
        max_variants=max_variants,
        project_history_context=project_history_context,
        context_documents=context_documents,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    column_names = [c.name for c in schema_summary.columns]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await client.chat_json(messages, DatasetDesignSuggestion)

            # Pre-process result to fix common LLM JSON issues
            result = _fix_llm_json_issues(result)

            # Validate the result
            suggestion = DatasetDesignSuggestion(**result)

            # Validate recommended variant exists
            variant_names = [v.name for v in suggestion.variants]
            if suggestion.recommended_variant not in variant_names:
                raise ValueError(
                    f"Recommended variant '{suggestion.recommended_variant}' not in variants: {variant_names}"
                )

            # Validate each variant
            for variant in suggestion.variants:
                # Build list of columns that will be available after engineering
                # Features can depend on original columns OR previously defined engineered features
                engineered_column_names = []
                if hasattr(variant, 'engineered_features') and variant.engineered_features:
                    for eng_feat in variant.engineered_features:
                        # Source columns can be from original data OR previously engineered features
                        available_for_engineering = column_names + engineered_column_names
                        missing_sources = [
                            col for col in eng_feat.source_columns
                            if col not in available_for_engineering
                        ]
                        if missing_sources:
                            raise ValueError(
                                f"Engineered feature '{eng_feat.output_column}' in variant '{variant.name}' "
                                f"uses non-existent source columns: {missing_sources}. "
                                f"Available columns: {column_names}. "
                                f"Previously defined engineered columns: {engineered_column_names}"
                            )
                        # Add this feature to available columns for subsequent features
                        engineered_column_names.append(eng_feat.output_column)

                # Available columns = original + engineered
                available_columns = column_names + engineered_column_names

                # Validate all feature columns exist (either in original data or will be engineered)
                invalid_features = [f for f in variant.feature_columns if f not in available_columns]
                if invalid_features:
                    raise ValueError(
                        f"Invalid feature columns in variant '{variant.name}': {invalid_features}. "
                        f"Available columns: {column_names}. "
                        f"Engineered columns: {engineered_column_names}. "
                        f"If you want to use derived columns, add them to 'engineered_features' first."
                    )

                # Ensure target is not in features
                if target_column in variant.feature_columns:
                    variant.feature_columns.remove(target_column)
                    if target_column not in variant.excluded_columns:
                        variant.excluded_columns.append(target_column)
                        variant.exclusion_reasons[target_column] = "Target column"

            return suggestion

        except (ValidationError, ValueError) as e:
            last_error = e
            logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")

            if attempt < MAX_RETRIES:
                messages.append({
                    "role": "assistant",
                    "content": str(result) if 'result' in dir() else "{}"
                })
                messages.append({
                    "role": "user",
                    "content": f"Your response had an error: {e}\n\nPlease fix and try again."
                })

    raise ValueError(f"Failed to generate valid dataset design after {MAX_RETRIES + 1} attempts: {last_error}")
