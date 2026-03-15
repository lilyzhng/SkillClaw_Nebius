# SkillClaw

**A self-evolving agent that learns to write robot skills — studies frameworks, writes solutions, debugs via state diff, and accumulates reusable skills.**

> Like Claude Code, but for robotics. Two tools: `bash()` + `sim_exec()`. Agent decides everything else.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │       Orchestrator (orchestrator.py) │
                    └──────┬──────────┬──────────┬────────┘
                           │          │          │
              ┌────────────┘     ┌────┘     ┌────┘
              ▼                  ▼          ▼
┌──────────────────────┐  ┌──────────┐  ┌────────────────┐
│ 🧪 Solver Agent(s)   │  │ 🤖 PR    │  │ 🔍 Oversight   │
│ (Claude Sonnet)      │  │  Agent   │  │  Agent         │
│                      │  │ (no LLM) │  │ (Claude Opus)  │
│ 1. UNDERSTAND task   │  │          │  │                │
│ 2. STUDY patterns    │  │ Renders  │  │ Reviews code:  │
│ 3. WRITE solve()     │  │ traj →   │  │ - no hardcoded │
│ 4. EXECUTE sim_exec  │  │ PR body  │  │   positions    │
│ 5. DEBUG via state   │  │ + GIF    │  │ - uses helpers │
│ 6. SAVE skill        │  │ + git    │  │ - error handle │
│                      │  │ + gh PR  │  │ - visual check │
└──────────┬───────────┘  └────┬─────┘  └────────┬───────┘
           │                   │                  │
     pr_queue ────────→   review_queue ──────→  gh approve
           │                   │                  │
    ┌──────┴──────┐     ┌──────┴──────┐          │
    │ bash()      │     │ git, gh,    │     Reads PR via
    │ sim_exec()  │     │ ffmpeg      │     gh pr diff
    ▼             ▼     └─────────────┘
  Local       GPU VM
  Files       ManiSkill
```

### Data Flow

```
Solver succeeds → saves skill to brains/private_brain/dev-sc-<name>/
                → pushes SkillSave to pr_queue

PR Agent picks up → renders trajectory as markdown (actual code, errors, state diffs)
                  → copies private → shared brain
                  → converts video to GIF
                  → git branch + commit + push + gh pr create
                  → pushes to review_queue

Oversight Agent   → reads SKILL.md + main.py from the PR (via gh)
                  → reads demo GIF for visual verification
                  → calls Claude Opus 4.6 to review
                  → gh pr review --approve or gh pr comment
```

## Getting Started

### Prerequisites

- Python 3.10+
- `httpx` (`pip install httpx`)
- OpenRouter API key (for Claude Sonnet + Opus)
- Nebius GPU VM (for ManiSkill simulation)
- `gh` CLI (authenticated with GitHub)
- `ffmpeg` (for video → GIF conversion)

### 1. Clone and install

```bash
git clone git@github.com:lilyzhng/SkillClaw.git
cd SkillClaw
pip install httpx
```

### 2. Set up environment variables

```bash
export OPENROUTER_API_KEY=sk-or-v1-...

# Optional: bot token for agent GitHub comments (separate account)
# export SKILLCLAW_BOT_TOKEN=github_pat_...
```

### 3. Start the Nebius GPU VM

```bash
# Start the VM
~/.nebius/bin/nebius compute instance start --id computeinstance-e00jppwj3yqa9ax8ve

# SSH in (current IP: 89.169.121.51, may change on restart)
ssh lily@89.169.121.51

# Start the resources server on the VM
cd ~/SkillClaw
nohup python3 rlvr/resources_server.py --port 8100 > /tmp/resources_server.log 2>&1 &

# Verify
curl http://localhost:8100/health
```

### 4. Run the multi-agent orchestrator

```bash
# Single task (solver + PR + oversight)
python -m rlvr.agents \
  --resources-url http://89.169.121.51:8100 \
  --tasks PullCube-v1:42 \
  --record-video

# Parallel: solve 3 tasks at once
python -m rlvr.agents \
  --resources-url http://89.169.121.51:8100 \
  --tasks PullCube-v1:42 LiftPegUpright-v1:42 PickSingleYCB-v1:42 \
  --record-video

# Flywheel (sequential, skills accumulate between tasks)
python -m rlvr.agents \
  --resources-url http://89.169.121.51:8100 \
  --tasks PullCube-v1:42 LiftPegUpright-v1:42 \
  --flywheel \
  --record-video
```

### 5. Run single agent (backwards compatible)

```bash
python rlvr/agent_server.py \
  --resources-url http://89.169.121.51:8100 \
  --env-id PickSingleYCB-v1 \
  --seed 42 \
  --record-video

# Study mode (agent reads ManiSkill repo, writes framework.md)
python rlvr/agent_server.py --study
```

### 6. Stop the VM (avoid GPU charges!)

```bash
~/.nebius/bin/nebius compute instance stop --id computeinstance-e00jppwj3yqa9ax8ve
```

## Project Structure

```
SkillClaw/
├── rlvr/
│   ├── agent_server.py         # Single-agent mode (bash + sim_exec ReAct loop)
│   ├── resources_server.py     # FastAPI server on GPU VM (ManiSkill execution)
│   ├── trajectory.py           # ATIF v1.6 trajectory recording
│   ├── primitives.py           # v1 primitives + task registry
│   ├── kitchen_helpers.py      # Kitchen/mobile robot helpers
│   └── agents/                 # Multi-agent orchestration
│       ├── orchestrator.py     # Main CLI — launches all agents
│       ├── solver_agent.py     # Solver: bash + sim_exec ReAct loop
│       ├── pr_agent.py         # PR agent: trajectory render + git + gh pr
│       ├── oversight_agent.py  # Oversight: code review + visual check (Opus 4.6)
│       ├── __init__.py
│       └── __main__.py         # Entry point for python -m rlvr.agents
├── brains/
│   ├── shared_brain/           # Shared across all agents (git repo)
│   │   ├── README.md           # Task analysis + API reference + skill index
│   │   ├── debug_patterns.md   # Lessons from failures
│   │   └── skills/sc-*/        # Published skills (via PR)
│   │       ├── SKILL.md        # Metadata + description
│   │       └── scripts/main.py # The solve() function
│   └── private_brain/          # Per-agent workspace (not shared)
│       ├── dev-sc-*/           # Skills in development
│       └── scratch.md          # Agent's thinking notebook
├── trajectories/               # ATIF v1.6 trajectory logs
│   └── solver_0/               # Per-agent trajectory dirs
├── demos/                      # Recorded videos + GIFs
├── scripts/                    # Test scripts
│   └── test_oversight.py       # Open-loop oversight agent test
└── ManiSkill/                  # ManiSkill repo (reference code)
```

## Available Tasks

14 ManiSkill tasks across three difficulty levels:

| Difficulty | Tasks |
|---|---|
| Easy | PickCube, PushCube, PullCube |
| Medium | StackCube, LiftPegUpright, PokeCube, RollBall, PlaceSphere, PickSingleYCB, TurnFaucet |
| Hard | StackPyramid, PegInsertionSide, PlugCharger, AssemblingKits |
| Kitchen | RoboCasaKitchen (mobile robot) |

## Key Files for Simon

| What you need | File |
|---|---|
| Run multi-agent | `python -m rlvr.agents --tasks EnvId:seed --resources-url http://<vm>:8100` |
| Run single agent | `python rlvr/agent_server.py --env-id EnvId --seed 42` |
| Resources server | `rlvr/resources_server.py` — endpoints: `/seed_session`, `/execute_code`, `/verify`, `/video/{filename}` |
| Orchestrator code | `rlvr/agents/orchestrator.py` |
| Solver agent | `rlvr/agents/solver_agent.py` |
| Available tasks | `TASK_DESCRIPTIONS` in `rlvr/agent_server.py` |

## Inspired By

- **[TidyBot Universe](https://github.com/jimmyyhwu/tidybot-universe)** — skill format (SKILL.md + scripts/main.py), OpenClaw-style memory
- **[Harbor](https://github.com/harbor-ai/harbor)** — bash-as-agent-interface, ATIF trajectory format for RL training
- **[OpenClaw](https://openclaw.org)** — agentic feedback loop + persistent memory architecture
- **[Voyager](https://voyager.minedojo.org/)** — LLM skill library concept (Minecraft → robotics)

Built for Nebius.Build SF 2026.
