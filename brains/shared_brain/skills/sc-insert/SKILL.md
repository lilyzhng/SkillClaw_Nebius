---
name: sc-insert
description: Pick up a peg and insert into a hole. Uses pose algebra for alignment and iterative refinement for precision.
---

# Insert

Grasp peg → compute alignment pose via pose algebra → iterative refinement → insert.
Pattern: `goal_pose * offset * obj.pose.inv() * tcp_pose`
