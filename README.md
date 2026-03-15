# SkillClaw

An AI agent that teaches itself robot skills by writing code, running it in simulation, debugging failures, and saving what it learns.

## How it works

The agent has two tools:
- **bash** — read files, search code, write notes
- **sim_exec** — run Python code on a GPU robot simulator (ManiSkill)

With just these two tools, the agent solves robot manipulation tasks by writing `solve()` functions, executing them in simulation, and iterating on failures using a scratch pad for cross-attempt reasoning.

## Architecture

```
rlvr/
  agent_server_v3.py    — Agent loop (bash + sim_exec, ReAct pattern)
  resources_server.py   — FastAPI server managing ManiSkill simulation sessions
  primitives.py         — 12 atomic robot actions + task registry
  kitchen_helpers.py    — Kitchen/mobile robot integration (TidyVerse)
  trajectory.py         — ATIF trajectory recording

brains/
  shared_brain/         — Skill library (graduated skills available to all runs)
    skills/sc-pick/     — Pick objects using OBB grasp
    skills/sc-push/     — Push objects to goals
    skills/sc-pull/     — Pull objects by grasping and dragging
    skills/sc-insert/   — Peg insertion with pose algebra
    skills/sc-plug-charger/  — Charger insertion
    skills/sc-stack-cube/    — Stack objects
    skills/sc-lift-peg-upright/  — Reorient objects
  private_brain/        — Agent's working memory (scratch pad)

maniskill-tidyverse/    — Mobile robot: 3-DOF base + 7-DOF Panda arm + Robotiq gripper (submodule)
```

## Quick Start

```bash
# 1. Start resources server on GPU machine
python rlvr/resources_server.py --port 8100

# 2. Run agent locally
export OPENROUTER_API_KEY=...
python rlvr/agent_server_v3.py \
  --env-id PickCube-v1 \
  --seed 42 \
  --record-video \
  --resources-url http://<gpu-vm-ip>:8100
```

## Supported Tasks

14 ManiSkill benchmark tasks (easy → hard):
- **Easy:** PickCube, PushCube, PullCube
- **Medium:** StackCube, LiftPegUpright, PlaceSphere, PickSingleYCB, RollBall, PokeCube, TurnFaucet
- **Hard:** PegInsertionSide, PlugCharger, StackPyramid, AssemblingKits
- **Kitchen:** RoboCasaKitchen-v1 (mobile manipulation)

## Built with

- [ManiSkill](https://github.com/haosulab/ManiSkill) — GPU-parallel robot simulation
- [Nebius AI Cloud](https://nebius.ai) — GPU compute
- [OpenRouter](https://openrouter.ai) — LLM API (Claude Sonnet)
- [TidyVerse](https://github.com/shaoyifei96/maniskill-tidyverse) — Mobile robot by Simon Shao
