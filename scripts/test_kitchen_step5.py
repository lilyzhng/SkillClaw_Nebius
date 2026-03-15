"""Step 5 test: hardcoded kitchen solve via resources_server API."""
import httpx
import json

URL = "http://localhost:8100"
client = httpx.Client(timeout=300)

# Create session
r = client.post(f"{URL}/seed_session", json={
    "env_id": "RoboCasaKitchen-v1",
    "seed": 0,
    "robot_uid": "tidyverse",
})
sid = r.json()["session_id"]
print(f"Session: {sid}")

code = '''
def solve(env, planner):
    import numpy as np
    pw = planning_world
    def step_fn(action):
        wrapped_env.step(action)

    target_pos = None
    for e in scene.entities:
        if e.name == "target_cube":
            target_pos = e.pose.p.tolist()
            break
    print(f"Cube at {target_pos}")

    nav_ok = navigate_to(env, planner, pw, target_pos, step_fn)
    print(f"navigate_to: {nav_ok}")

    for l in robot.get_links():
        if l.get_name() == "panda_link0":
            arm_base = l.pose.p[0].cpu().numpy()
            break
    dist = np.linalg.norm(arm_base - np.array(target_pos))
    print(f"Arm base to cube: {dist:.3f}m")

    for e in scene.entities:
        if e.name == "target_cube":
            target_pos = e.pose.p.tolist()
            break
    result = pick_up(env, planner, pw, target_pos, step_fn)
    print(f"pick_up: {result}")

    for e in scene.entities:
        if e.name == "target_cube":
            final = e.pose.p.tolist()
            print(f"Cube final Z: {final[2]:.4f} (delta: {final[2] - 0.94:.4f}m)")
            break
'''

r = client.post(f"{URL}/execute_code", json={
    "session_id": sid,
    "record_video": True,
    "code": code,
})
d = r.json()
print(f"success: {d['success']}")
if d.get("error"):
    print(f"error: {str(d['error'])[:300]}")
if d.get("error_traceback"):
    print(f"traceback: {d['error_traceback'][-400:]}")
sa = d.get("state_after", {})
cube = sa.get("spawned_objects", {}).get("target_cube")
print(f"cube after: {cube}")
print(f"gripper after: {sa.get('gripper_position')}")

# Cleanup
client.post(f"{URL}/cleanup_session", json={"session_id": sid})
