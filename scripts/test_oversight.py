"""Open-loop test for the oversight agent.

Tests two modes:
1. With PR URL — reads files from GitHub PR (the real path)
2. Without PR URL — reads files from local disk (fallback)

Usage:
    # Test with existing PR #2
    python scripts/test_oversight.py --pr https://github.com/lilyzhng/SkillClaw/pull/2

    # Test with local files (fallback)
    python scripts/test_oversight.py
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rlvr"))

from agents.oversight_agent import _review_skill, DEFAULT_OVERSIGHT_MODEL
from agents.solver_agent import SkillSave

import httpx
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", default="", help="PR URL to review (reads files from GitHub)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY")
        sys.exit(1)

    skill = SkillSave(
        skill_name="pull-cube",
        env_id="PullCube-v1",
        seed=42,
        description="Pull a cube onto a target.",
        trajectory_path="trajectories/solver_0/PullCube-v1_seed42_sc_aa8a80d9ad6d.json",
        video_file="PullCube-v1_sess_1e0ec912.mp4",
        agent_id="solver_0",
    )
    skill.pr_url = args.pr

    print(f"Testing oversight agent:")
    print(f"  Skill: sc-{skill.skill_name}")
    print(f"  PR URL: {skill.pr_url or '(none — local fallback)'}")
    print(f"  Model: {DEFAULT_OVERSIGHT_MODEL}")
    print()

    async with httpx.AsyncClient(timeout=120) as client:
        await _review_skill(client, api_key, skill, DEFAULT_OVERSIGHT_MODEL)


if __name__ == "__main__":
    asyncio.run(main())
