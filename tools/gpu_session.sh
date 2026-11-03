#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lock_file="${BREEZE_GPU_LOCK_FILE:-$repo_root/state/gpu-session.lock}"
mkdir -p -- "$(dirname -- "$lock_file")"
exec 9>"$lock_file"
flock -n 9 || { echo 'Another GPU session holds this project lock.' >&2; exit 1; }
service="${BREEZE_TTS_SERVICE:-}"
restore_tts=0
cleanup() {
    result=$?
    trap - EXIT
    if [ "$restore_tts" = 1 ]; then
        if ! systemctl --user start "$service"; then
            echo 'Could not restore the configured user service.' >&2
            [ "$result" = 0 ] && result=1
        fi
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -n "$service" ] && systemctl --user is-active --quiet "$service"; then
    systemctl --user stop "$service"
    restore_tts=1
fi
"$@"
