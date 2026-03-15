# Private Brain

Per-agent workspace. Each agent has its own private brain in its sandbox/VM. Not shared with other agents.

## Structure

```
private_brain/
├── dev-sc-*/          ← Skills in development (not yet proven)
├── logs/              ← Debug trajectories and attempt history
└── notes.md           ← Agent's own observations and hypotheses
```

## What goes here vs shared_brain

| Private Brain | Shared Brain |
|---|---|
| Skills in development (`dev-sc-*`) | Proven skills (`sc-*`) |
| Debug logs and failed attempts | Debug patterns (distilled lessons) |
| Agent-specific notes | Framework knowledge |
| Experimental code | Building blocks |

## Promotion flow

```
private_brain/dev-sc-new-skill/     ← agent develops here
    │
    │ passes multi-seed test
    │ agent creates PR
    ▼
shared_brain/skills/sc-new-skill/   ← promoted after review
```
