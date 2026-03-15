"""
SkillClaw Oversight Agent — reviews solver code quality before PR approval.

Uses Claude Opus 4.6 with vision to review:
- Code quality (no hardcoded values, uses framework helpers, error handling)
- Demo GIF verification (visually confirm the robot completed the task)
- SKILL.md accuracy
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger("oversight")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OVERSIGHT_MODEL = "anthropic/claude-opus-4.6"
PROJECT_ROOT = Path(__file__).parent.parent.parent  # rlvr/agents/ → SkillClaw/

REVIEW_PROMPT = """You are a senior code reviewer for SkillClaw, a robot skill learning system that teaches robots manipulation tasks through code generation.

Your job: review a robot manipulation skill and decide if it's APPROVED or needs CHANGES.

## Review Criteria

### Code Quality
1. **No hardcoded positions/seeds** — The solve() function should work for ANY seed, not just the one it was tested on. Positions must come from state observation (get_actor_obb, initial_state), not literal coordinates. Small constant offsets (e.g. 0.02 for clearance) are acceptable if they're physics-based, not position-based.

2. **Uses framework helpers** — Should use get_actor_obb(), compute_grasp_info_by_obb(), planner.move_to_pose_with_screw(), etc. where appropriate. Raw coordinate manipulation is a red flag.

3. **Error handling** — Checks planner return values (res == -1 means failure). Raises RuntimeError on planning failures instead of silently returning.

4. **Reusable pattern** — The approach should generalize. A skill that only works because of a specific object arrangement is not reusable.

5. **SKILL.md accuracy** — The description and strategy in SKILL.md should match what the code actually does.

### Visual Verification (if demo GIF provided)
6. **Task completion** — Does the GIF show the robot actually completing the task? (e.g., for "pull cube", does the cube end up at the goal position?)
7. **Motion quality** — Is the robot motion smooth and reasonable? Any collisions, drops, or erratic behavior?

## Response Format

Start your response with either:
- `APPROVED` — if the skill meets all criteria
- `CHANGES NEEDED` — if there are issues

Then explain your reasoning briefly. For CHANGES NEEDED, list the specific issues.

Keep your review concise (under 400 words).
"""


async def run_oversight_agent(
    api_key: str,
    review_queue: asyncio.Queue,
    model_id: str = DEFAULT_OVERSIGHT_MODEL,
    timeout: float = 600,
):
    """Consume PRs from review_queue and review them.

    Args:
        api_key: OpenRouter API key.
        review_queue: Queue of SkillSave objects with pr_url attached.
        model_id: LLM model for review (default: Opus 4.6 for vision).
        timeout: Seconds to wait for next PR before exiting.
    """
    logger.info("Oversight Agent started, waiting for PRs to review...")

    async with httpx.AsyncClient(timeout=120) as client:
        while True:
            try:
                skill = await asyncio.wait_for(review_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.info("Oversight Agent: no more PRs to review, exiting.")
                break

            pr_url = getattr(skill, "pr_url", None)
            logger.info(f"Oversight Agent: reviewing {skill.skill_name} (PR: {pr_url})")

            try:
                await _review_skill(client, api_key, skill, model_id)
            except Exception as e:
                logger.error(f"Oversight Agent: review failed for {skill.skill_name}: {e}")

    logger.info("Oversight Agent finished.")


def _load_gif_as_base64(skill) -> str | None:
    """Try to load the demo GIF for visual review."""
    if not skill.video_file:
        return None

    gif_name = skill.video_file.replace(".mp4", ".gif")
    gif_path = PROJECT_ROOT / "demos" / gif_name
    if gif_path.exists():
        data = gif_path.read_bytes()
        return base64.standard_b64encode(data).decode("utf-8")

    # Also check for mp4 directly (some models handle video)
    mp4_path = PROJECT_ROOT / "demos" / skill.video_file
    if mp4_path.exists():
        data = mp4_path.read_bytes()
        return base64.standard_b64encode(data).decode("utf-8")

    return None


def _read_pr_files(pr_url: str) -> dict[str, str]:
    """Read PR files directly from GitHub via gh CLI. Returns {filename: content}."""
    files = {}
    if not pr_url:
        return files

    # Get the diff to find which files were added/changed
    result = subprocess.run(
        f'gh pr diff "{pr_url}"',
        shell=True, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), env=_bot_env(),
    )
    if result.returncode != 0:
        logger.warning(f"Failed to read PR diff: {result.stderr[:200]}")
        return files

    # Get list of files in PR
    files_result = subprocess.run(
        f'gh pr view "{pr_url}" --json files --jq ".files[].path"',
        shell=True, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), env=_bot_env(),
    )
    if files_result.returncode != 0:
        return files

    # Read each file from the PR branch
    # First get the branch name
    branch_result = subprocess.run(
        f'gh pr view "{pr_url}" --json headRefName --jq ".headRefName"',
        shell=True, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), env=_bot_env(),
    )
    branch = branch_result.stdout.strip()
    if not branch:
        return files

    # Only read text files we care about (skip binaries like GIFs)
    TEXT_EXTENSIONS = {".md", ".py", ".txt", ".yaml", ".yml", ".json", ".sh"}

    for filepath in files_result.stdout.strip().split("\n"):
        filepath = filepath.strip()
        if not filepath:
            continue
        # Skip binary files
        ext = Path(filepath).suffix.lower()
        if ext not in TEXT_EXTENSIONS:
            continue
        # Read file content from the PR branch via git
        content_result = subprocess.run(
            f'git show "origin/{branch}:{filepath}"',
            shell=True, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        if content_result.returncode == 0:
            files[filepath] = content_result.stdout

    return files


async def _review_skill(
    client: httpx.AsyncClient,
    api_key: str,
    skill,
    model_id: str,
):
    """Review a single skill's code quality + visual verification."""
    name = skill.skill_name
    pr_url = getattr(skill, "pr_url", "")

    # Read skill files from the PR itself (not local disk)
    skill_md = "(SKILL.md not found)"
    main_py = "(main.py not found)"

    if pr_url:
        logger.info(f"Oversight Agent: reading files from PR {pr_url}")
        # Fetch remote so we have the PR branch
        subprocess.run(
            "git fetch origin", shell=True, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        pr_files = _read_pr_files(pr_url)
        for filepath, content in pr_files.items():
            if filepath.endswith("SKILL.md"):
                skill_md = content
            elif filepath.endswith("main.py"):
                main_py = content
        logger.info(f"Oversight Agent: read {len(pr_files)} files from PR")
    else:
        # Fallback: read from local disk (for open-loop testing)
        logger.info(f"Oversight Agent: no PR URL, reading from local disk")
        skill_dir = PROJECT_ROOT / "brains" / "shared_brain" / "skills" / f"sc-{name}"
        skill_md_path = skill_dir / "SKILL.md"
        main_py_path = skill_dir / "scripts" / "main.py"
        skill_md = skill_md_path.read_text() if skill_md_path.exists() else skill_md
        main_py = main_py_path.read_text() if main_py_path.exists() else main_py

    # Build user message content (text + optional image)
    task_desc = skill.description
    text_content = (
        f"## Task\n{task_desc}\n\n"
        f"## SKILL.md\n\n{skill_md}\n\n"
        f"## scripts/main.py\n\n```python\n{main_py}\n```"
    )

    # Try to include trajectory context
    if skill.trajectory_path and Path(skill.trajectory_path).exists():
        try:
            traj = json.loads(Path(skill.trajectory_path).read_text())
            extra = traj.get("final_metrics", {}).get("extra", {})
            text_content += (
                f"\n\n## Trajectory Summary\n"
                f"- Task success: {extra.get('task_success')}\n"
                f"- Sim attempts: {extra.get('sim_attempts')}\n"
                f"- Time: {extra.get('elapsed_seconds')}s\n"
            )
        except Exception:
            pass

    # Build message parts
    user_content = []

    # Add GIF for visual verification
    gif_b64 = _load_gif_as_base64(skill)
    if gif_b64:
        gif_name = skill.video_file.replace(".mp4", ".gif")
        media_type = "image/gif"
        if not gif_name.endswith(".gif"):
            media_type = "video/mp4"

        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": gif_b64,
            },
        })
        text_content = "**Demo GIF is attached above.** Please verify the robot completes the task visually.\n\n" + text_content
        logger.info(f"Oversight Agent: attached demo GIF for visual review")

    user_content.append({"type": "text", "text": text_content})

    # Call LLM (Opus 4.6 with vision)
    response = await client.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "max_tokens": 1500,
            "messages": [
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": user_content},
            ],
        },
    )

    if response.status_code != 200:
        logger.error(f"Oversight Agent: LLM error {response.status_code}: {response.text[:200]}")
        return

    review = response.json()["choices"][0]["message"]["content"]
    pr_url = getattr(skill, "pr_url", "")
    logger.info(f"Oversight Agent review for {name}:\n{review}")

    if "APPROVED" in review.upper().split("\n")[0]:
        logger.info(f"Oversight Agent: APPROVED {name}")
        if pr_url:
            _gh_review(pr_url, approve=True, body=f"> **🔍 Oversight Agent** · Powered by Claude Opus 4.6\n\n✅ **APPROVED**\n\n{review}")
            logger.info(f"Oversight Agent: approved PR {pr_url}")
    else:
        logger.info(f"Oversight Agent: CHANGES NEEDED for {name}")
        logger.info(f"  Review: {review[:200]}")
        if pr_url:
            _gh_review(pr_url, approve=False, body=f"> **🔍 Oversight Agent** · Powered by Claude Opus 4.6\n\n❌ **CHANGES NEEDED**\n\n{review}")
            logger.info(f"Oversight Agent: commented on PR {pr_url}")


def _bot_env() -> dict:
    """Return env with bot token for gh commands, if available."""
    env = os.environ.copy()
    bot_token = os.environ.get("SKILLCLAW_BOT_TOKEN")
    if bot_token:
        env["GH_TOKEN"] = bot_token
    return env


def _gh_review(pr_url: str, approve: bool, body: str):
    """Post a review or comment on a GitHub PR. Uses bot token if available."""
    review_file = f"/tmp/skillclaw_review_{hash(pr_url)}.md"
    Path(review_file).write_text(body)

    if approve:
        cmd = f'gh pr review "{pr_url}" --approve --body-file "{review_file}"'
    else:
        cmd = f'gh pr comment "{pr_url}" --body-file "{review_file}"'

    subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
        env=_bot_env(),
    )
    Path(review_file).unlink(missing_ok=True)
