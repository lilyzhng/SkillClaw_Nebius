"""
SkillClaw PR Agent — creates rich PRs from solver output.

Consumes SkillSave messages from a queue and:
1. Renders the trajectory into a rich PR body (no LLM needed)
2. Copies skill from private → shared brain
3. Converts video to GIF
4. Creates branch, commits, pushes, creates PR via gh
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pr_agent")

PROJECT_ROOT = Path(__file__).parent.parent.parent  # rlvr/agents/ → SkillClaw/

# Import the enhanced render function
sys.path.insert(0, str(PROJECT_ROOT / "brains" / "shared_brain" / "skills" / "sc-publish" / "scripts"))
from render_trajectory import render


def _bot_env() -> dict:
    """Return env with bot token for gh commands, if available."""
    env = os.environ.copy()
    bot_token = os.environ.get("SKILLCLAW_BOT_TOKEN")
    if bot_token:
        env["GH_TOKEN"] = bot_token
    return env


def _run_git(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git/shell command from project root. Uses bot token for gh commands."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), timeout=60,
        env=_bot_env() if cmd.startswith("gh ") else None,
    )
    if check and result.returncode != 0:
        logger.error(f"Command failed: {cmd}\n{result.stderr}")
    return result


async def run_pr_agent(
    skill_queue: asyncio.Queue,
    review_queue: Optional[asyncio.Queue] = None,
    timeout: float = 600,
):
    """Consume SkillSave messages and create PRs.

    Args:
        skill_queue: Queue of SkillSave objects from solver agents.
        review_queue: Queue to push completed PRs for oversight review.
        timeout: Seconds to wait for next skill before exiting.
    """
    logger.info("PR Agent started, waiting for skills...")

    while True:
        try:
            skill = await asyncio.wait_for(skill_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.info("PR Agent: no more skills in queue, exiting.")
            break

        logger.info(f"PR Agent: processing {skill.skill_name} from {skill.agent_id}")

        try:
            await _create_pr(skill, review_queue)
        except Exception as e:
            logger.error(f"PR Agent: failed to create PR for {skill.skill_name}: {e}")

    logger.info("PR Agent finished.")


async def _create_pr(skill, review_queue):
    """Create a PR for a single skill."""
    import time
    name = skill.skill_name
    timestamp = time.strftime("%m%d-%H%M")
    branch = f"feat/skill-add-{name}-{timestamp}"
    private_dir = PROJECT_ROOT / "brains" / "private_brain" / f"dev-sc-{name}"
    shared_dir = PROJECT_ROOT / "brains" / "shared_brain" / "skills" / f"sc-{name}"

    # 1. Verify private brain has the skill
    if not private_dir.exists():
        logger.error(f"PR Agent: private brain dir not found: {private_dir}")
        return

    # 2. Render trajectory BEFORE git operations (files are still on disk)
    traj_section = ""
    if skill.trajectory_path and Path(skill.trajectory_path).exists():
        try:
            traj_section = render(skill.trajectory_path)
        except Exception as e:
            logger.warning(f"PR Agent: trajectory render failed: {e}")

    # 3. Convert video to GIF BEFORE git operations
    gif_file = ""
    if skill.video_file:
        mp4_path = PROJECT_ROOT / "demos" / skill.video_file
        gif_name = skill.video_file.replace(".mp4", ".gif")
        gif_path = PROJECT_ROOT / "demos" / gif_name
        if mp4_path.exists():
            logger.info(f"PR Agent: converting video to GIF")
            _run_git(
                f'ffmpeg -i "demos/{skill.video_file}" -vf "fps=10,scale=320:-1" -y "demos/{gif_name}" 2>/dev/null',
                check=False,
            )
            if gif_path.exists():
                gif_file = gif_name

    # 4. Git: create branch FIRST, then copy files onto it
    _run_git("git checkout main", check=False)
    _run_git(f"git branch -D {branch}", check=False)
    _run_git(f"git push origin --delete {branch}", check=False)  # clean remote too
    _run_git(f"git checkout -b {branch}")

    # 5. Copy private → shared brain (on the new branch)
    logger.info(f"PR Agent: copying {private_dir.name} → {shared_dir.name}")
    _run_git(f"cp -r '{private_dir}' '{shared_dir}'")

    # 6. Clean private brain
    _run_git(f"rm -rf '{private_dir}'")

    # 7. Stage, commit, push
    _run_git(f"git add 'brains/shared_brain/skills/sc-{name}'")
    if gif_file:
        _run_git(f"git add 'demos/{gif_file}'")

    _run_git(f'git commit -m "skill: sc-{name} — {skill.description}"')
    _run_git(f"git push -u origin {branch}")

    # 7. Build PR body
    repo_result = _run_git("gh repo view --json nameWithOwner --jq '.nameWithOwner'", check=False)
    repo = repo_result.stdout.strip() or "lilyzhng/SkillClaw"
    commit_sha = _run_git("git rev-parse HEAD").stdout.strip()

    gif_section = ""
    if gif_file:
        gif_url = f"https://raw.githubusercontent.com/{repo}/{commit_sha}/demos/{gif_file}"
        gif_section = f"### Demo\n\n![demo]({gif_url})\n"

    pr_body = f"""> **🤖 PR Agent** · Automated by SkillClaw

## I learned a new skill: sc-{name}

**Description:** {skill.description}

{gif_section}
{traj_section}

### Skill Files

- `brains/shared_brain/skills/sc-{name}/SKILL.md`
- `brains/shared_brain/skills/sc-{name}/scripts/main.py`
"""

    # 8. Create PR
    pr_body_file = f"/tmp/skillclaw_pr_{name}.md"
    Path(pr_body_file).write_text(pr_body)

    result = _run_git(
        f'gh pr create --title "I learned a new skill: {skill.description}" --body-file "{pr_body_file}"'
    )
    pr_url = result.stdout.strip()
    logger.info(f"PR Agent: PR created — {pr_url}")

    # Cleanup temp file
    Path(pr_body_file).unlink(missing_ok=True)

    # Switch back to main for next PR
    _run_git("git checkout main", check=False)

    # Push to review queue
    if review_queue is not None:
        skill.pr_url = pr_url
        await review_queue.put(skill)
        logger.info(f"PR Agent: pushed to review_queue: {name}")
