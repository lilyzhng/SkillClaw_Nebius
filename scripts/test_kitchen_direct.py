"""Direct kitchen grasp test — no agent, just session + execute_code."""
import httpx
import json
import sys

URL = sys.argv[1] if len(sys.argv) > 1 else "http://89.169.121.51:8100"
client = httpx.Client(timeout=300)

# Create session
r = client.post(f"{URL}/seed_session", json={
    "env_id": "RoboCasaKitchen-v1",
    "seed": 0,
    "robot_uid": "tidyverse",
})
d = r.json()
sid = d["session_id"]
cube = d["initial_state"].get("spawned_objects", {}).get("target_cube", {})
print(f"Session: {sid}")
print(f"Cube at: {cube.get('position')}")

# Execute pick_up directly (no navigate_to — let whole-body planner handle base)
code = '''
def solve(env, planner):
    import numpy as np

    def step_fn(action):
        wrapped_env.step(action)

    # Find target cube
    target_pos = None
    for e in scene.entities:
        if e.name == "target_cube":
            target_pos = list(e.pose.p)
            break
    if target_pos is None:
        raise RuntimeError("No target cube found")

    print(f"Target cube at {target_pos}")
    print(f"Arm base at {[round(x,3) for x in next(l for l in robot.get_links() if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()]}")

    # Call pick_up directly — whole-body planner handles base movement
    result = pick_up(env, planner, planning_world, target_pos, step_fn)
    print(f"pick_up result: {result}")

    # Check cube final position
    for e in scene.entities:
        if e.name == "target_cube":
            final = list(e.pose.p)
            z_delta = final[2] - target_pos[2]
            print(f"Cube final: {[round(x,4) for x in final]}")
            print(f"Cube Z delta: {z_delta:.4f}m")
            if z_delta > 0.05:
                print("SUCCESS — cube lifted!")
            elif z_delta > 0.01:
                print("PARTIAL — cube moved slightly")
            else:
                print("FAIL — cube didn't move")
            break
'''

r = client.post(f"{URL}/execute_code", json={
    "session_id": sid,
    "record_video": True,
    "code": code,
})
d = r.json()
print(f"\nsuccess: {d['success']}")
if d.get("error"):
    print(f"error: {str(d['error'])[:300]}")
if d.get("error_traceback"):
    print(f"traceback: {d['error_traceback'][-400:]}")

sa = d.get("state_after", {})
cube_after = sa.get("spawned_objects", {}).get("target_cube")
print(f"cube after: {cube_after}")
print(f"video: {d.get('video_path')}")

# Cleanup
client.post(f"{URL}/cleanup_session", json={"session_id": sid})
