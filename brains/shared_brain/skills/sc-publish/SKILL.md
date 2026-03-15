---
name: sc-publish
description: Publish a learned skill to the shared brain via PR. Use after successfully solving a task and saving to private brain.
---

# Publish Skill

Promotes a skill from private brain to shared brain and opens a PR.

## When to use

After task success + saving to `brains/private_brain/dev-sc-<name>/`.

## Usage

```bash
bash brains/shared_brain/skills/sc-publish/scripts/main.sh <name> <video_file> "<description>"
```

## Example

```bash
bash brains/shared_brain/skills/sc-publish/scripts/main.sh pull PullCube-v1_sess_abc.mp4 "Pull object to goal by grasping and dragging"
```

## What it does

1. Copies `dev-sc-<name>` from private to shared brain as `sc-<name>`
2. Deletes the private brain copy
3. Converts video to GIF (for PR preview)
4. Creates branch `feat/skill-add-<name>`
5. Commits shared brain skill + GIF
6. Pushes and opens PR
