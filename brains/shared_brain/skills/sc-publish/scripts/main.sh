#!/bin/bash
# Usage: bash brains/shared_brain/skills/sc-publish/scripts/main.sh <skill-name> <video-file> <description>
# Example: bash brains/shared_brain/skills/sc-publish/scripts/main.sh pull PullCube-v1_sess_abc.mp4 "Pull object to goal by grasping and dragging"

set -e

SKILL_NAME=$1
VIDEO_FILE=$2
DESCRIPTION=$3

if [ -z "$SKILL_NAME" ] || [ -z "$VIDEO_FILE" ] || [ -z "$DESCRIPTION" ]; then
    echo "Usage: bash brains/shared_brain/skills/sc-publish/scripts/main.sh <skill-name> <video-file> <description>"
    echo "Example: bash brains/shared_brain/skills/sc-publish/scripts/main.sh pull PullCube-v1_sess_abc.mp4 \"Pull object to goal\""
    exit 1
fi

GIF_FILE="${VIDEO_FILE%.mp4}.gif"
BRANCH_NAME="feat/skill-add-$SKILL_NAME"
PR_BODY_FILE="/tmp/skillclaw_pr_body.md"

# Get repo info for raw URL
REPO=$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || echo "lilyzhng/SkillClaw")

echo "=== Publishing skill: sc-$SKILL_NAME ==="

# 1. Copy to shared brain
echo "1. Copying to shared brain..."
cp -r "brains/private_brain/dev-sc-$SKILL_NAME" "brains/shared_brain/skills/sc-$SKILL_NAME"

# 2. Clean private brain
echo "2. Cleaning private brain..."
rm -rf "brains/private_brain/dev-sc-$SKILL_NAME"

# 3. Convert video to GIF
echo "3. Converting video to GIF..."
ffmpeg -i "demos/$VIDEO_FILE" -vf "fps=10,scale=320:-1" -y "demos/$GIF_FILE" 2>/dev/null

# 4. Create branch
echo "4. Creating branch..."
git checkout -b "$BRANCH_NAME"

# 5. Add files (GIF only, not mp4)
echo "5. Staging files..."
git add "brains/shared_brain/skills/sc-$SKILL_NAME" "demos/$GIF_FILE"

# 6. Commit
echo "6. Committing..."
git commit -m "skill: sc-$SKILL_NAME — $DESCRIPTION"

# 7. Push (GIF must be on GitHub BEFORE we reference it in PR body)
echo "7. Pushing..."
git push -u origin "$BRANCH_NAME"

# 8. Get the commit SHA for raw URL (now that it's pushed)
COMMIT_SHA=$(git rev-parse HEAD)
GIF_URL="https://raw.githubusercontent.com/$REPO/$COMMIT_SHA/demos/$GIF_FILE"

# 9. Render trajectory (find latest trajectory file)
echo "8. Rendering trajectory..."
SCRIPT_DIR="$(dirname "$0")"
LATEST_TRAJ=$(ls -t trajectories/*.json 2>/dev/null | head -1)
TRAJ_SECTION=""
if [ -n "$LATEST_TRAJ" ]; then
    TRAJ_SECTION=$(python3 "$SCRIPT_DIR/render_trajectory.py" "$LATEST_TRAJ" 2>/dev/null || echo "")
fi

# 10. Generate PR description markdown (AFTER push, so GIF URL works)
echo "9. Generating PR description..."
cat > "$PR_BODY_FILE" << EOF
## I learned a new skill: sc-$SKILL_NAME

**Description:** $DESCRIPTION

### Demo

![demo]($GIF_URL)

$TRAJ_SECTION

### Skill Files

- \`brains/shared_brain/skills/sc-$SKILL_NAME/SKILL.md\`
- \`brains/shared_brain/skills/sc-$SKILL_NAME/scripts/main.py\`
EOF

echo "--- PR description ---"
cat "$PR_BODY_FILE"
echo "----------------------"

# 10. Create PR using body file
echo "9. Creating PR..."
gh pr create \
    --title "I learned a new skill: $DESCRIPTION" \
    --body-file "$PR_BODY_FILE"

# 11. Cleanup temp file
rm -f "$PR_BODY_FILE"

echo "=== Done! PR created for sc-$SKILL_NAME ==="
