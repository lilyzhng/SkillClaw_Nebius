"""Render ATIF trajectory JSON into readable markdown for PR descriptions."""

import json
import sys


def render(trajectory_path: str) -> str:
    t = json.load(open(trajectory_path))
    lines = []
    lines.append("### Agent Trajectory")
    lines.append("")

    # Collect all actions into phases
    study_cmds = []
    attempts = []  # list of {"num", "code_len", "result", "detail", "reasoning"}
    sim_count = 0
    pending_reasoning = None  # reasoning text before a sim_exec

    for step in t["steps"]:
        source = step["source"]
        msg = step.get("message", "")

        if source == "agent":
            has_sim = any(tc["function_name"] == "sim_exec" for tc in step.get("tool_calls", []))

            # Capture reasoning (agent thinking before/between attempts)
            if msg and len(msg) > 20:
                # Extract key reasoning — look for numbered plans or key sentences
                clean_lines = []
                for line in msg.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Keep numbered steps, key insights, strategy descriptions
                    if (line.startswith(("1.", "2.", "3.", "- "))
                        or "plan" in line.lower()
                        or "strategy" in line.lower()
                        or "issue" in line.lower()
                        or "try" in line.lower()
                        or "approach" in line.lower()
                        or "didn't" in line.lower()
                        or "failed" in line.lower()
                        or "instead" in line.lower()):
                        # Clean markdown formatting
                        cleaned = line.lstrip("#*- ").strip("*")
                        if len(cleaned) > 10:
                            clean_lines.append(cleaned[:120])
                    if len(clean_lines) >= 4:
                        break
                if clean_lines:
                    pending_reasoning = clean_lines  # keep as list
                else:
                    # Fallback: first sentence
                    fallback = msg.replace("\n", " ").strip()[:150]
                    pending_reasoning = [fallback]

            for tc in step.get("tool_calls", []):
                name = tc["function_name"]
                args = tc["arguments"]
                if name == "bash" and sim_count == 0:
                    cmd = args.get("command", "")
                    if "cat " in cmd:
                        file = cmd.split("cat ")[-1].split(" ")[0].strip("\"'")
                        study_cmds.append(f"Read `{file}`")
                    elif "ls " in cmd:
                        dir_path = cmd.split("ls ")[-1].split(" ")[0].strip("\"'")
                        study_cmds.append(f"List `{dir_path}`")
                    elif "find " in cmd or "grep " in cmd:
                        study_cmds.append(f"Search: `{cmd[:60]}`")
                elif name == "sim_exec":
                    sim_count += 1
                    current = {
                        "num": sim_count,
                        "code_len": len(args.get("code", "")),
                        "result": None,
                        "detail": None,
                        "reasoning": pending_reasoning,
                    }
                    pending_reasoning = None
                    attempts.append(current)

        elif source == "system" and step.get("observation"):
            for r in step["observation"]["results"]:
                content = r.get("content", "")
                if attempts and attempts[-1]["result"] is None:
                    if "TASK SUCCESS" in content:
                        attempts[-1]["result"] = "success"
                    elif "ERROR" in content:
                        for el in content.split("\n"):
                            if el.startswith("EXECUTION ERROR:"):
                                attempts[-1]["detail"] = el.replace("EXECUTION ERROR: ", "")
                            elif el.startswith("Message:") and attempts[-1]["detail"]:
                                attempts[-1]["detail"] += " — " + el.replace("Message: ", "")[:80]
                        attempts[-1]["result"] = "error"
                    elif "NOT completed" in content:
                        attempts[-1]["result"] = "not_completed"
                        attempts[-1]["detail"] = "Code ran but goal not reached"

    # Render: Study Phase
    lines.append("**📚 Study Phase**")
    lines.append("")
    for cmd in study_cmds:
        lines.append(f"- {cmd}")
    lines.append("")

    # Render: Execution Phase
    lines.append("**🔧 Execution Phase**")
    lines.append("")
    for a in attempts:
        # Show reasoning if available (agent's thinking before this attempt)
        if a["reasoning"]:
            lines.append(f"> 💭 **Agent's thinking:**")
            for r in a["reasoning"]:
                lines.append(f"> - {r}")
            lines.append("")

        if a["result"] == "success":
            lines.append(f"- **Attempt {a['num']}:** `sim_exec({a['code_len']} chars)` → ✅ **SUCCESS**")
        elif a["result"] == "error":
            lines.append(f"- **Attempt {a['num']}:** `sim_exec({a['code_len']} chars)` → ❌ {a['detail']}")
        elif a["result"] == "not_completed":
            lines.append(f"- **Attempt {a['num']}:** `sim_exec({a['code_len']} chars)` → ❌ {a['detail']}")
        else:
            lines.append(f"- **Attempt {a['num']}:** `sim_exec({a['code_len']} chars)` → ⏳ unknown")
        lines.append("")

    # Final metrics
    metrics = t.get("final_metrics", {})
    extra = metrics.get("extra", {})
    if extra:
        result = '✅ SUCCESS' if extra.get('task_success') else '❌ FAIL'
        lines.append(f"**Result:** {result} | **Attempts:** {extra.get('sim_attempts', '?')} | **Time:** {extra.get('elapsed_seconds', '?')}s")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_trajectory.py <trajectory.json>")
        sys.exit(1)
    print(render(sys.argv[1]))
