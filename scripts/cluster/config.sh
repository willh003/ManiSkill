# Shared settings for the local <-> klone workflow. Sourced by the other scripts.
REMOTE_HOST=${REMOTE_HOST:-klone}
REMOTE_DIR=${REMOTE_DIR:-/mmfs1/gscratch/weirdlab/will/ManiSkill}
LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Never pushed (and never deleted on the remote): results and local-only junk.
PUSH_EXCLUDES=(--exclude=.git --exclude=logs/ --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.egg-info'
  --exclude=.venv --exclude=wandb/ --exclude=/data --exclude=/demos --exclude=/videos
  --exclude=/tmp --exclude=.claude/settings.local.json)

# Fail fast (no hanging on a password/Duo prompt) if the ssh control socket is gone.
check_ssh() {
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" true 2>/dev/null; then
    echo "ERROR: cannot reach $REMOTE_HOST non-interactively. Ask the user to run '! ssh $REMOTE_HOST' (Duo) to refresh the control socket." >&2
    exit 2
  fi
}
