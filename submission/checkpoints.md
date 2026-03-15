# SkillClaw — Pitch (3 min + 2 min Q&A)

## Judging: Live Demo 45%, Creativity 35%, Impact 20%

---

## One-liner

"An AI agent that teaches itself robot skills by writing code, failing, debugging, and trying again — then saves what it learns so the next task is easier."

---

## Pitch Flow (3 min)

### Hook (0:00 – 0:30)

"How do you teach a robot a new skill today? You write a program. It fails. You debug it. You rewrite it. Repeat for weeks.

What if the robot's AI could do that loop itself? Write code, run it in simulation, read the error, fix it, try again — and when it succeeds, save the skill for next time.

That's SkillClaw."

### Live Demo (0:30 – 2:00) — THIS IS 45% OF THE SCORE

**Show the agent solving a task it's never seen live.**

Option A (safest): Run the agent on a NEW task (e.g. PickSingleYCB or a kitchen task) and show:
1. Agent reads the task description + initial state
2. Agent studies the skill library (bash: `cat brains/shared_brain/README.md`)
3. Agent writes solve() code
4. sim_exec runs on GPU VM → robot moves in simulation → video appears
5. If it fails: agent reads error, updates scratch pad, writes new code
6. If it succeeds: agent saves skill to library, creates PR

Show the scratch pad updating in real time — the audience sees the agent THINKING.

Option B (pre-recorded + live): Play the 1-minute video showing the progression, then do a live agent run.

**Key demo moments:**
- Scratch pad updating live (audience sees reasoning)
- Robot moving in simulation (video from sim_exec)
- Skill library growing (show the PR / shared_brain)
- Kitchen: mobile robot navigating a full kitchen and picking up objects

### How It Works (2:00 – 2:30)

"Two tools. That's it.

The agent has `bash` — to read code, search docs, write files. And `sim_exec` — to run code on a GPU robot simulator.

With just these two tools, the agent:
- Reads the task description
- Studies reference code and its own skill library
- Writes a solve() function
- Runs it in ManiSkill simulation
- If it fails: reads the error + state diff, updates its scratch pad, tries again
- If it succeeds: saves the skill, publishes a PR

Each solved task becomes a reusable skill. The 7th task is easier than the 1st because the agent has 6 skills to learn from.

We call this the flywheel: solve → save → share → solve faster."

### Results (2:30 – 2:50)

"In 4 flywheel runs, the agent went from solving 1 out of 4 tasks to 4 out of 4.

It learned to pick objects, stack cubes, insert pegs, and plug chargers — all by writing code and self-debugging.

Then we plugged in a completely different robot — a mobile base with a Panda arm in a full kitchen — same agent loop, same two tools. It navigated to a counter and picked up a cube. No retraining. No new code. Just a new environment."

### Close (2:50 – 3:00)

"Robots don't need more training data. They need an agent that can debug itself. SkillClaw is that agent."

---

## 1-Minute Video (submission)

Storyboard:
1. (5s) Title card: "SkillClaw — Self-Debugging Robot Skill Agent"
2. (15s) Flywheel montage: PickCube → StackCube → PegInsertion → PlugCharger. Show score 1/4 → 4/4
3. (10s) Scratch pad scrolling — agent reasoning about physics ("gripper pushes ball when positioning!")
4. (15s) Kitchen: mobile robot navigating, reaching counter, grasping cube, lifting
5. (10s) Skill library growing: PRs being created, GIFs of solved tasks
6. (5s) End card: "bash + sim_exec. Two tools. Any robot skill."

---

## Q&A Prep

**Q: How is this different from just prompting GPT-4 to write robot code?**
A: Three things. First, the self-debug loop — it doesn't just write code once, it iterates with state diffs that show exactly what the robot did wrong. Second, the skill library — solved tasks become reusable patterns. Third, the scratch pad — the agent tracks patterns across attempts, like a human engineer would.

**Q: Does it work on real robots?**
A: ManiSkill has sim-to-real transfer. The code the agent writes calls the same API a real robot uses. Today it's simulation; the gap to hardware is one API swap, not a rewrite.

**Q: What's the success rate?**
A: On 14 benchmark tasks: 7 fully solved (50%), 2 partial (made real progress), 5 not yet attempted. The solved tasks include hard ones — peg insertion, charger plugging — that require sub-millimeter alignment.

**Q: Why not just use RL/imitation learning?**
A: RL needs thousands of rollouts to learn one task. Our agent solves tasks in 1-4 attempts because it can READ code, READ errors, and REASON about physics. RL is great for calibrating force — code gen is great for strategy. They're complementary.

**Q: What about the kitchen demo?**
A: Different robot (mobile base + arm), different environment (RoboCasa kitchen with 70+ fixtures), same agent loop. The agent had never seen this robot before. It figured out the action space, the planner API, and wrote grasp code in 2 attempts. That's the transferability story.

**Q: What sponsors did you use?**
A: Nebius GPU cloud for the simulation server. OpenRouter for the LLM API (Claude Sonnet). The agent runs locally, sim runs on Nebius.

---

## Demo Logistics

- **Resources server** running on Nebius VM (89.169.121.51:8100)
- **Agent** runs locally on laptop (Python script)
- **Scratch pad** visible in VS Code (brains/private_brain/scratch.md)
- **Videos** download automatically to demos/ folder
- **Backup**: pre-recorded videos if live demo fails
