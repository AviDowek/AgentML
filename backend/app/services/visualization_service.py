"""Visualization service for generating and executing data visualizations."""
import base64
import io
import logging
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.llm_client import BaseLLMClient
from app.services.prompts import (
    SYSTEM_ROLE_VIZ_DEVELOPER,
    SYSTEM_ROLE_VIZ_SCIENTIST,
    SYSTEM_ROLE_VIZ_ANALYST,
    get_visualization_code_prompt,
    get_default_visualizations_prompt,
    get_visualization_explanation_prompt,
)

logger = logging.getLogger(__name__)


def get_data_summary_for_llm(file_path: str, max_rows: int = 5) -> Dict[str, Any]:
    """Get a data summary suitable for sending to LLM (no raw data, just structure).

    Args:
        file_path: Path to the data file
        max_rows: Maximum sample rows to include

    Returns:
        Dictionary with column info, dtypes, statistics (no raw values)
    """
    # Determine file type and read
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise

    # Build column information
    columns_info = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            col_info["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            col_info["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
            col_info["std"] = float(df[col].std()) if not pd.isna(df[col].std()) else None
            col_info["is_numeric"] = True
        else:
            col_info["is_numeric"] = False
            # For categorical, include top values (names only, not full data)
            value_counts = df[col].value_counts()
            if len(value_counts) <= 10:
                col_info["unique_values"] = list(value_counts.index.astype(str))
            else:
                col_info["top_5_values"] = list(value_counts.head(5).index.astype(str))

        columns_info.append(col_info)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
        "column_names": list(df.columns),
    }


async def generate_visualization_code(
    llm_client: BaseLLMClient,
    data_summary: Dict[str, Any],
    user_request: str,
    file_path: str,
    previous_visualizations: Optional[List[Dict[str, str]]] = None,
    error_feedback: Optional[str] = None,
    failed_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate Python visualization code using LLM.

    Args:
        llm_client: LLM client instance
        data_summary: Summary of the data structure
        user_request: User's visualization request
        file_path: Path to the data file
        previous_visualizations: Optional list of previous viz requests/code for context
        error_feedback: Optional error message from previous failed attempt
        failed_code: Optional code that produced the error (for context)

    Returns:
        Dictionary with 'code', 'title', 'description'
    """
    from pydantic import BaseModel, Field

    class VisualizationCode(BaseModel):
        code: str = Field(description="Complete Python code to generate the visualization")
        title: str = Field(description="Short title for the visualization")
        description: str = Field(description="Brief description of what the visualization shows")
        chart_type: str = Field(description="Type of chart (e.g., bar, line, scatter, histogram, heatmap, etc.)")

    # Build context about previous visualizations
    prev_context = ""
    if previous_visualizations:
        prev_context = "\n\nPrevious visualizations in this session:\n"
        for i, viz in enumerate(previous_visualizations[-3:], 1):  # Last 3 only
            prev_context += f"{i}. {viz.get('title', 'Untitled')}: {viz.get('description', 'No description')}\n"

    # Build error feedback section for retries
    error_context = ""
    if error_feedback and failed_code:
        error_context = f"""

IMPORTANT - PREVIOUS ATTEMPT FAILED:
The previous code you generated produced this error:
{error_feedback}

Failed code:
```python
{failed_code[:2000]}{'...(truncated)' if len(failed_code) > 2000 else ''}
```

Please analyze the error carefully and fix the issue in your new code. Common issues include:
- Syntax errors (missing colons, parentheses, incorrect indentation)
- Using incorrect column names that don't exist in the data
- Incorrect pandas/matplotlib API usage
- Missing imports
- Type errors or null value handling

Generate corrected code that avoids this error.
"""

    # Use centralized prompt from prompts.py
    columns_details = _format_columns_for_prompt(data_summary['columns'])
    prompt = get_visualization_code_prompt(
        user_request=user_request,
        data_summary=data_summary,
        file_path=file_path,
        columns_details=columns_details,
        error_context=error_context,
        prev_context=prev_context,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_VIZ_DEVELOPER},
        {"role": "user", "content": prompt},
    ]

    response = await llm_client.chat_json(messages, VisualizationCode)

    return {
        "code": response.get("code", ""),
        "title": response.get("title", "Visualization"),
        "description": response.get("description", ""),
        "chart_type": response.get("chart_type", "unknown"),
    }


async def generate_default_visualizations(
    llm_client: BaseLLMClient,
    data_summary: Dict[str, Any],
    file_path: str,
    project_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate a set of default visualizations for initial data exploration.

    Args:
        llm_client: LLM client instance
        data_summary: Summary of the data structure
        file_path: Path to the data file
        project_context: Optional context from project and agent runs

    Returns:
        List of visualization suggestions with code
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    class VisualizationSuggestion(BaseModel):
        title: str = Field(description="Short title for the visualization")
        description: str = Field(description="What this visualization reveals about the data and why it's useful for the project goal")
        chart_type: str = Field(description="Type of chart (histogram, bar, scatter, heatmap, box, line, pie, etc.)")
        request: str = Field(description="Specific, detailed visualization request that can be used to generate code")

    class DefaultVisualizationsResponse(BaseModel):
        visualizations: ListType[VisualizationSuggestion] = Field(description="List of 4-5 highly relevant visualizations")

    # Build project context section
    project_section = ""
    if project_context:
        project_section = "\n\nPROJECT CONTEXT:\n"

        if project_context.get("project_name"):
            project_section += f"- Project: {project_context['project_name']}\n"

        if project_context.get("project_description"):
            project_section += f"- Goal: {project_context['project_description']}\n"

        if project_context.get("task_type"):
            task_type = project_context["task_type"]
            project_section += f"- ML Task: {task_type}\n"

        if project_context.get("target_column"):
            project_section += f"- Target Variable: {project_context['target_column']} (this is what we want to predict)\n"

        if project_context.get("primary_metric"):
            project_section += f"- Success Metric: {project_context['primary_metric']}\n"

        if project_context.get("problem_summary"):
            project_section += f"- Analysis: {project_context['problem_summary'][:300]}\n"

        if project_context.get("key_features"):
            project_section += f"- Important Features: {', '.join(project_context['key_features'][:8])}\n"

        if project_context.get("data_quality_issues"):
            issues = project_context["data_quality_issues"]
            if issues:
                issue_strs = []
                for issue in issues[:3]:
                    if isinstance(issue, dict):
                        issue_strs.append(issue.get("description", str(issue))[:100])
                    else:
                        issue_strs.append(str(issue)[:100])
                if issue_strs:
                    project_section += f"- Data Issues Found: {'; '.join(issue_strs)}\n"

        if project_context.get("correlations"):
            project_section += f"- Known Correlations: {project_context['correlations']}\n"

    # Use centralized prompt from prompts.py
    columns_details = _format_columns_for_prompt(data_summary['columns'])
    prompt = get_default_visualizations_prompt(
        data_summary=data_summary,
        columns_details=columns_details,
        project_section=project_section,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_VIZ_SCIENTIST},
        {"role": "user", "content": prompt},
    ]

    response = await llm_client.chat_json(messages, DefaultVisualizationsResponse)

    logger.info(f"generate_default_visualizations response: {response}")

    # Handle different response formats
    visualizations = response.get("visualizations", [])
    if not visualizations and isinstance(response, dict):
        # Maybe the response is the list directly or has a different key
        for key in response:
            if isinstance(response[key], list) and len(response[key]) > 0:
                visualizations = response[key]
                break

    logger.info(f"Extracted {len(visualizations)} visualizations")
    return visualizations


async def explain_visualization(
    llm_client: BaseLLMClient,
    visualization_info: Dict[str, Any],
    data_summary: Dict[str, Any],
) -> str:
    """Generate an explanation of what a visualization shows and its implications.

    Args:
        llm_client: LLM client instance
        visualization_info: Info about the visualization (title, description, chart_type)
        data_summary: Summary of the data

    Returns:
        Detailed explanation string
    """
    # Use centralized prompt from prompts.py
    prompt = get_visualization_explanation_prompt(
        title=visualization_info.get('title', 'Unknown'),
        chart_type=visualization_info.get('chart_type', 'Unknown'),
        description=visualization_info.get('description', 'No description'),
        row_count=data_summary['row_count'],
        column_count=data_summary['column_count'],
        column_names=data_summary['column_names'],
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_VIZ_ANALYST},
        {"role": "user", "content": prompt},
    ]

    response = await llm_client.chat(messages)
    return response


def execute_visualization_code(code: str) -> Dict[str, Any]:
    """Execute Python visualization code and return the result.

    Args:
        code: Python code to execute

    Returns:
        Dictionary with 'image_base64' or 'error'
    """
    # Import modules properly
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Create execution environment with all necessary modules and builtins
    # Use the full builtins module to allow import statements in the generated code
    # IMPORTANT: Use the same dict for both globals and locals to avoid scoping issues
    # where variables defined in the code (like 'df') aren't accessible later
    import builtins
    exec_env = {
        '__builtins__': builtins,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'io': io,
        'base64': base64,
    }

    try:
        logger.info(f"Executing visualization code ({len(code)} chars)")

        # Execute the code - use same dict for globals and locals to avoid scoping issues
        exec(code, exec_env, exec_env)

        # Get the result
        result = exec_env.get('result')
        if result and isinstance(result, dict) and 'image_base64' in result:
            image_data = result.get('image_base64')
            if image_data:
                logger.info(f"Visualization generated successfully, image size: {len(image_data)} chars")
            else:
                logger.warning("Result dict exists but image_base64 is empty/None")
            return result
        else:
            logger.error(f"Code did not produce expected result. exec_env keys: {list(exec_env.keys())}")
            if result:
                logger.error(f"Result type: {type(result)}, value: {result}")
            return {"error": "Code did not produce expected result format. Check that 'result' variable is set with 'image_base64' key."}

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Visualization code execution failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg}
    finally:
        # Clean up matplotlib figures
        try:
            plt.close('all')
        except:
            pass


def _format_columns_for_prompt(columns: List[Dict[str, Any]]) -> str:
    """Format column information for the LLM prompt."""
    lines = []
    for col in columns:
        line = f"- {col['name']} ({col['dtype']})"
        if col.get('is_numeric'):
            line += f": numeric, range [{col.get('min', 'N/A')} - {col.get('max', 'N/A')}], mean={col.get('mean', 'N/A'):.2f}" if col.get('mean') else ": numeric"
        else:
            if col.get('unique_values'):
                line += f": categorical, values={col['unique_values']}"
            elif col.get('top_5_values'):
                line += f": categorical, {col['unique_count']} unique, top 5={col['top_5_values']}"
        if col.get('null_percentage', 0) > 0:
            line += f" ({col['null_percentage']}% null)"
        lines.append(line)
    return "\n".join(lines)
