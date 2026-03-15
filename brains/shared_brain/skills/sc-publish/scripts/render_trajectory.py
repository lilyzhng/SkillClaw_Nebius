"""Render ATIF trajectory JSON into readable markdown for PR descriptions.

Enhanced version: shows full code per attempt, actual errors, state diffs,
and agent reasoning between attempts.
"""

import json
import sys


def _extract_solve_body(code: str, max_lines: int = 40) -> str:
    """Extract the key lines from a solve() function, skipping boilerplate."""
    lines = code.strip().split("\n")
    # Find def solve line
    start = 0
    for i, line in enumerate(lines):
        if "def solve" in line:
            start = i
            break

    body_lines = lines[start:]
    if len(body_lines) > max_lines:
        body_lines = body_lines[:max_lines] + [f"    # ... ({len(lines) - start - max_lines} more lines)"]
    return "\n".join(body_lines)


def _extract_state_diff(content: str) -> str:
    """Extract key state changes from observation content."""
    diffs = []
    try:
        # Look for state_before and state_after in the content
        before_start = content.find("State BEFORE execution:")
        after_start = content.find("State AFTER execution:")
        if before_start == -1 or after_start == -1:
            return ""

        before_text = content[before_start:after_start]
        after_text = content[after_start:]

        # Parse JSON from each section
        before_json_start = before_text.find("{")
        after_json_start = after_text.find("{")
        if before_json_start == -1 or after_json_start == -1:
            return ""

        # Find the end of each JSON block
        before_json = _extract_json(before_text[before_json_start:])
        after_json = _extract_json(after_text[after_json_start:])

        if not before_json or not after_json:
            return ""

        before = json.loads(before_json)
        after = json.loads(after_json)

        # Compare key fields
        for key in after:
            if key in before and before[key] != after[key]:
                diffs.append(f"  {key}: {_format_val(before[key])} → {_format_val(after[key])}")
    except (json.JSONDecodeError, ValueError):
        pass

    if diffs:
        return "State changes:\n" + "\n".join(diffs[:5])
    return ""


def _extract_json(text: str) -> str:
    """Extract first complete JSON object from text."""
    depth = 0
    start = text.find("{")
    if start == -1:
        return ""
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return ""


def _format_val(val) -> str:
    """Format a value for display."""
    if isinstance(val, float):
        return f"{val:.3f}"
    if isinstance(val, list) and len(val) <= 4:
        return "[" + ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in val) + "]"
    if isinstance(val, list):
        return f"[{len(val)} items]"
    return str(val)[:60]


def render(trajectory_path: str) -> str:
    t = json.load(open(trajectory_path))
    lines = []

    # Extract agent/model info
    agent_id = t.get("extra", {}).get("agent_id", "solver")
    model_name = t.get("agent", {}).get("model_name", "Claude Sonnet")
    # Shorten model name for display
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    lines.append(f"> **🧪 Solver Agent** (`{agent_id}`) · Powered by {model_short}")
    lines.append("")
    lines.append("### Agent Trajectory")
    lines.append("")

    # Collect all actions into phases
    study_cmds = []
    attempts = []  # list of dicts
    sim_count = 0
    pending_reasoning = None
    # Map tool_call_id → observation content
    observations = {}

    # First pass: collect all observations
    for step in t["steps"]:
        if step["source"] == "system" and step.get("observation"):
            for r in step["observation"]["results"]:
                if r.get("source_call_id"):
                    observations[r["source_call_id"]] = r.get("content", "")

    # Second pass: build study + execution phases
    for step in t["steps"]:
        source = step["source"]
        msg = step.get("message", "")

        if source == "agent":
            # Capture reasoning
            if msg and len(msg) > 20:
                clean_lines = []
                for line in msg.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if (line.startswith(("1.", "2.", "3.", "- "))
                        or any(kw in line.lower() for kw in
                               ["plan", "strategy", "issue", "try", "approach",
                                "didn't", "failed", "instead", "observation",
                                "key insight", "problem", "fix"])):
                        cleaned = line.lstrip("#*- ").strip("*")
                        if len(cleaned) > 10:
                            clean_lines.append(cleaned[:150])
                    if len(clean_lines) >= 5:
                        break
                if clean_lines:
                    pending_reasoning = clean_lines
                else:
                    fallback = msg.replace("\n", " ").strip()[:200]
                    pending_reasoning = [fallback]

            for tc in step.get("tool_calls", []):
                name = tc["function_name"]
                args = tc["arguments"]
                tc_id = tc.get("tool_call_id", "")

                if name == "bash" and sim_count == 0:
                    cmd = args.get("command", "")
                    if "cat " in cmd:
                        file = cmd.split("cat ")[-1].split(" ")[0].strip("\"'")
                        study_cmds.append(f"Read `{file}`")
                    elif "ls " in cmd:
                        dir_path = cmd.split("ls ")[-1].split(" ")[0].strip("\"'")
                        study_cmds.append(f"List `{dir_path}`")
                    elif "find " in cmd or "grep " in cmd:
                        study_cmds.append(f"Search: `{cmd[:80]}`")

                elif name == "sim_exec":
                    sim_count += 1
                    code = args.get("code", "")
                    obs_content = observations.get(tc_id, "")

                    # Determine result
                    result = "unknown"
                    detail = ""
                    state_diff = ""
                    if "TASK SUCCESS" in obs_content:
                        result = "success"
                    elif "EXECUTION ERROR" in obs_content:
                        result = "error"
                        for el in obs_content.split("\n"):
                            if el.startswith("EXECUTION ERROR:"):
                                detail = el.replace("EXECUTION ERROR: ", "")
                            elif el.startswith("Message:") and detail:
                                detail += " — " + el.replace("Message: ", "")[:100]
                    elif "NOT completed" in obs_content:
                        result = "not_completed"
                        detail = "Code ran but goal not reached"

                    # Extract state diff
                    state_diff = _extract_state_diff(obs_content)

                    attempts.append({
                        "num": sim_count,
                        "code": code,
                        "code_len": len(code),
                        "result": result,
                        "detail": detail,
                        "reasoning": pending_reasoning,
                        "state_diff": state_diff,
                    })
                    pending_reasoning = None

    # Render: Study Phase
    if study_cmds:
        lines.append("**Study Phase**")
        lines.append("")
        for cmd in study_cmds:
            lines.append(f"- {cmd}")
        lines.append("")

    # Render: Execution Phase
    lines.append("**Execution Phase**")
    lines.append("")
    for a in attempts:
        # Agent reasoning before this attempt
        if a["reasoning"]:
            lines.append(f"> **Agent's thinking:**")
            for r in a["reasoning"]:
                lines.append(f"> - {r}")
            lines.append("")

        # Result header
        if a["result"] == "success":
            lines.append(f"**Attempt {a['num']}** ({a['code_len']} chars) → **SUCCESS**")
        elif a["result"] == "error":
            lines.append(f"**Attempt {a['num']}** ({a['code_len']} chars) → Error: {a['detail']}")
        elif a["result"] == "not_completed":
            lines.append(f"**Attempt {a['num']}** ({a['code_len']} chars) → {a['detail']}")
        else:
            lines.append(f"**Attempt {a['num']}** ({a['code_len']} chars) → unknown")

        # Show code (collapsed for non-final attempts, expanded for last)
        code_body = _extract_solve_body(a["code"])
        if a["num"] == len(attempts):
            # Final attempt: show full code
            lines.append("")
            lines.append("<details open>")
            lines.append(f"<summary>Code (attempt {a['num']})</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(code_body)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
        elif a["code"]:
            # Earlier attempts: collapsed
            lines.append("")
            lines.append("<details>")
            lines.append(f"<summary>Code (attempt {a['num']})</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(code_body)
            lines.append("```")
            lines.append("")
            lines.append("</details>")

        # State diff
        if a["state_diff"]:
            lines.append("")
            lines.append(f"```")
            lines.append(a["state_diff"])
            lines.append(f"```")

        lines.append("")

    # Final metrics
    metrics = t.get("final_metrics", {})
    extra = metrics.get("extra", {})
    if extra:
        result = 'SUCCESS' if extra.get('task_success') else 'FAIL'
        lines.append(
            f"**Result:** {result} | "
            f"**Attempts:** {extra.get('sim_attempts', '?')} | "
            f"**Time:** {extra.get('elapsed_seconds', '?')}s"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_trajectory.py <trajectory.json>")
        sys.exit(1)
    print(render(sys.argv[1]))
