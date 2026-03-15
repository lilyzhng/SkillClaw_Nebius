"""
SkillClaw Orchestrator — multi-agent coordination.

Launches solver agents in parallel, a PR agent, and an oversight agent.
Replaces agent_server_v3.py as the main CLI for multi-task runs.

Usage:
    # Parallel: solve 3 tasks at once
    python -m rlvr.agents --tasks PullCube-v1:42 LiftPegUpright-v1:42 PickSingleYCB-v1:42

    # Single task (backwards compatible)
    python -m rlvr.agents --tasks PullCube-v1:42

    # Flywheel (sequential, skill library accumulates)
    python -m rlvr.agents --flywheel --tasks PullCube-v1:42 LiftPegUpright-v1:42
"""

import argparse
import asyncio
import logging
import os
from typing import List, Tuple

from .solver_agent import run_solver, DEFAULT_MODEL
from .pr_agent import run_pr_agent
from .oversight_agent import run_oversight_agent, DEFAULT_OVERSIGHT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("orchestrator")


def parse_task_spec(spec: str) -> Tuple[str, int]:
    """Parse 'EnvId-v1:seed' or 'EnvId-v1' (default seed=42)."""
    if ":" in spec:
        env_id, seed_str = spec.rsplit(":", 1)
        return env_id, int(seed_str)
    return spec, 42


async def run_orchestrator(
    resources_url: str,
    api_key: str,
    tasks: List[Tuple[str, int]],
    model_id: str = DEFAULT_MODEL,
    record_video: bool = False,
    queue_timeout: float = 120,
):
    """Run the full multi-agent pipeline.

    Args:
        resources_url: URL of the ManiSkill resources server.
        api_key: OpenRouter API key.
        tasks: List of (env_id, seed) tuples.
        model_id: LLM model ID.
        record_video: Record simulation videos.
        queue_timeout: Seconds PR/oversight agents wait before exiting.
    """
    pr_queue = asyncio.Queue()       # solver → PR agent
    review_queue = asyncio.Queue()   # PR agent → oversight

    logger.info(f"Orchestrator: launching {len(tasks)} solver(s) + PR agent + oversight agent")
    for i, (env_id, seed) in enumerate(tasks):
        logger.info(f"  Solver {i}: {env_id} seed={seed}")

    # Launch solver agents in parallel
    solver_coros = [
        run_solver(
            resources_url=resources_url,
            api_key=api_key,
            env_id=env_id,
            seed=seed,
            agent_id=f"solver_{i}",
            pr_queue=pr_queue,
            model_id=model_id,
            record_video=record_video,
        )
        for i, (env_id, seed) in enumerate(tasks)
    ]

    # Launch PR agent (creates PRs from solver output)
    pr_coro = run_pr_agent(pr_queue, review_queue, timeout=queue_timeout)

    # Launch oversight agent (reviews PRs — uses Opus 4.6 for vision)
    oversight_coro = run_oversight_agent(
        api_key=api_key,
        review_queue=review_queue,
        model_id=DEFAULT_OVERSIGHT_MODEL,
        timeout=queue_timeout,
    )

    # Run all concurrently
    results = await asyncio.gather(
        *solver_coros,
        pr_coro,
        oversight_coro,
        return_exceptions=True,
    )

    # Separate solver results from agent results
    solver_results = results[:len(tasks)]
    logger.info(f"\n{'='*60}")
    logger.info("ORCHESTRATOR RESULTS")
    logger.info(f"{'='*60}")
    for r in solver_results:
        if isinstance(r, Exception):
            logger.error(f"  ❌ Solver error: {r}")
        else:
            status = "✅" if r["task_success"] else "❌"
            logger.info(
                f"  {status} {r['env_id']} seed={r['seed']} — "
                f"{r['sim_attempts']} attempts, {r['total_steps']} steps, "
                f"{r['elapsed_seconds']}s"
            )

    successes = sum(
        1 for r in solver_results
        if not isinstance(r, Exception) and r.get("task_success")
    )
    logger.info(f"  Total: {successes}/{len(tasks)} succeeded")
    logger.info(f"{'='*60}")

    return solver_results


async def run_flywheel(
    resources_url: str,
    api_key: str,
    tasks: List[Tuple[str, int]],
    model_id: str = DEFAULT_MODEL,
    record_video: bool = False,
    queue_timeout: float = 120,
):
    """Run tasks sequentially (flywheel mode).

    Skills accumulate in the shared brain between tasks,
    so later tasks can reference earlier skills.
    """
    logger.info(f"Flywheel: {len(tasks)} tasks, sequential")

    all_results = []
    for i, (env_id, seed) in enumerate(tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"Flywheel task {i+1}/{len(tasks)}: {env_id} seed={seed}")
        logger.info(f"{'='*60}")

        results = await run_orchestrator(
            resources_url=resources_url,
            api_key=api_key,
            tasks=[(env_id, seed)],
            model_id=model_id,
            record_video=record_video,
            queue_timeout=queue_timeout,
        )
        all_results.extend(results)

        successes = sum(
            1 for r in all_results
            if not isinstance(r, Exception) and r.get("task_success")
        )
        logger.info(f"Flywheel progress: {successes}/{len(all_results)}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="SkillClaw Orchestrator — multi-agent task solving",
    )
    parser.add_argument("--resources-url", default="http://localhost:8100")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help="Tasks as EnvId-v1:seed (e.g. PullCube-v1:42). Seed defaults to 42.",
    )
    parser.add_argument(
        "--flywheel", action="store_true",
        help="Run tasks sequentially (skill library accumulates between tasks)",
    )
    parser.add_argument(
        "--queue-timeout", type=float, default=120,
        help="Seconds PR/oversight agents wait for work before exiting",
    )

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        exit(1)

    tasks = [parse_task_spec(t) for t in args.tasks]

    if args.flywheel:
        asyncio.run(run_flywheel(
            resources_url=args.resources_url,
            api_key=api_key,
            tasks=tasks,
            model_id=args.model,
            record_video=args.record_video,
            queue_timeout=args.queue_timeout,
        ))
    else:
        asyncio.run(run_orchestrator(
            resources_url=args.resources_url,
            api_key=api_key,
            tasks=tasks,
            model_id=args.model,
            record_video=args.record_video,
            queue_timeout=args.queue_timeout,
        ))


if __name__ == "__main__":
    main()
