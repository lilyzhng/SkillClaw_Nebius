
# Team Name - SkillClaw

# Project Description - SkillClaw: Collectively Self-Improving Robot Agents

# Public GitHub Repository
# Pitch - 3 Minutes
### 1. Problem

**Traditional Robot Learning:**
```
collect demonstrations → train policy → deploy → repeat for every new task
```
Knowledge doesn't compound. Every new task is day zero.

### 2. The Inversion

What if we let a coding agent figure out the physics, robot learning by itself?

Give it building blocks, tools, and an environment, then let it figure things out. Progressively. It tries, it fails, it learns from the failure, it tries again. Eventually it works.

### 3. The Compounding (demo: PullCube)

We give the agent simple building blocks — pick, push, grasp. When it faces PullCube — a task it's never seen — it reads the existing skills, composes them, and figures it out. "Grasp like pick, drag like push."

**Demo:** Show the PullCube trajectory where the agent reads sc-pick + sc-push, reasons about the physics, and solves a novel task by composing building blocks.

### 4. The Collective

And it's not just one agent. We spawn multiple agents into different tasks in parallel. Each writes to its own private brain, When a skill works, it raises individual intelligence to a shared robot brain. 

This is collective self-improvement.

**Show:** shared brain / private brain architecture, skill promotion flow.

### Punchline

*"Every task any agent solves makes every future agent better. The skill library is the flywheel."*



