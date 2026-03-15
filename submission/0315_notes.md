---
date: 2026-03-15
time: 11:02
---

### Project Overview: Collectively Self-Improving Robot Agents

- Traditional robot learning starts from zero for each new task - knowledge doesn’t compound
- Proposed solution: Give robots building blocks and environment to progressively learn through trials and failures
- Core concept: Robots compose existing skills (grasp, pick, push, drag) to tackle new tasks
- Demo shows robot learning to grasp objects from different orientations when initial approach fails

### Technical Implementation

- Working demo includes basic skills: pick, push, grasp, drag
- Agent attempts task, documents failure, tries different approach until success
- Example: Robot learns to grasp peg from side instead of top when initial orientation fails
- System includes planner component for task sequencing and skill composition
- Uses simulation environment for rapid iteration (20+ attempts per session)

### Development Strategy: Parallel Processing

- Two-layer parallelization approach:
  1. High-level: Different agents running in parallel environments
  2. Low-level: Varying parameters within skills for faster development
- Multiple environment setups (kitchen, robot arm scenarios)
- Strategy parameters include motion direction, simulation speed, initial conditions
- Enables rapid skill development and testing across scenarios

### Next Steps

### Simon
**Fix RoboCasa Environment:**
- Fix collision issue in the kitchen environment
- Continue development
**Interesting demo cases to try:**
- Chain tasks (compound skills)
- Water bottle in the RoboCasa kitchen environment

### Lily
**Lily:**
- Codebase cleanup → merge into one repo
- Prepare submission details (repo, pitch, demo video)
- Finalize pitch materials

**Parallelization (two layers):**
1. High-level: multiple agents running in parallel across different environments
2. Low-level: show variance across parameters, same skill with different parameters, run in parallel → faster iteration
    - Different initial conditions: speed, direction (push forward vs backward)
    - Variations within skills: different parameters per skill
