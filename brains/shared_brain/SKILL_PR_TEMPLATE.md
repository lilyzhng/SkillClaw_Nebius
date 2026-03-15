## New Skill: sc-{name}

**Author:** {agent_session_id}
**Task:** {env_id} (seed={seed})
**Result:** {SUCCESS/FAIL} (attempt {n})
**Date:** {timestamp}

### What this skill does

{one sentence description}

### Strategy

{what patterns were composed/adapted and why}

### Demo

![demo](demos/{task}_sess_{id}.gif)

### Agent Trajectory

```
Step 1: [agent] {what agent decided to do}
  ├─ bash("{command}")  → {result summary}

Step 2: [agent] {reasoning}
  ├─ bash("{command}")  → {result summary}

...

Step N: [agent] {final attempt}
  ├─ sim_exec(code)     → {SUCCESS/FAIL + key state change}
```

Render this from the ATIF trajectory JSON. Show each agent step with its tool calls and key observations. Keep it concise — summarize long outputs.

### Evidence

- **Seed:** {seed}
- **Attempts:** {n}
- **Time:** {elapsed}s
- **GIF:** `demos/{task}_sess_{id}.gif`
