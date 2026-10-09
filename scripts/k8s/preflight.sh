#!/bin/bash
# Pre-launch checks for our Nautilus Jobs. Usage:
#   scripts/k8s/preflight.sh <rendered.yaml> [pod-with-pvc]
# Exits non-zero when: the rendered file still has __PLACEHOLDERS__, we already
# have >= MAX_OURS active pods, or the PVC (read through the given pod, or the
# first running pod of ours that mounts it) has < MIN_FREE_GB free.
set -u

NS="${NS:-dsc-capstone-25-26}"
MAX_OURS="${MAX_OURS:-20}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
OUR_PREFIX="${OUR_PREFIX:-sh-}"

rendered="${1:-}"
pvc_pod="${2:-}"
status=0

if [ -z "$rendered" ] || [ ! -f "$rendered" ]; then
    echo "usage: $0 <rendered.yaml> [pod-with-pvc]" >&2
    exit 1
fi

# 1. placeholders (comment lines excluded)
n_ph=$(grep -v '^[[:space:]]*#' "$rendered" | grep -c '__[A-Z_]*__' || true)
if [ "$n_ph" -gt 0 ]; then
    echo "FAIL: $n_ph unresolved placeholder line(s) in $rendered:"
    grep -v '^[[:space:]]*#' "$rendered" | grep -n '__[A-Z_]*__' | sed 's/\(TOKEN\|KEY\)__.*/\1__ .../'
    status=1
else
    echo "ok: no placeholders"
fi

# 2. pod counts
all_pods=$(kubectl get pods -n "$NS" --no-headers 2>/dev/null)
total=$(printf '%s\n' "$all_pods" | grep -c . || true)
ours_active=$(printf '%s\n' "$all_pods" | grep "^$OUR_PREFIX" | grep -Ev 'Completed|Succeeded|Failed|Error' | grep -c . || true)
echo "pods: $total in namespace, $ours_active active of ours (max $MAX_OURS)"
if [ "$ours_active" -ge "$MAX_OURS" ]; then
    echo "FAIL: at or above our pod cap"
    status=1
fi

# 3. PVC free space
if [ -z "$pvc_pod" ]; then
    pvc_pod=$(printf '%s\n' "$all_pods" | grep "^$OUR_PREFIX" | grep -w Running | awk '{print $1}' | head -1)
fi
if [ -n "$pvc_pod" ]; then
    # busybox wraps long device names onto their own line; take the free
    # column relative to the end of the last line (Avail Use% Mounted).
    free_line=$(kubectl exec "$pvc_pod" -n "$NS" -- df -k /mnt/data 2>/dev/null | tail -1)
    if [ -n "$free_line" ]; then
        free_kb=$(printf '%s' "$free_line" | awk '{print $(NF-2)}')
        free_gb=$((free_kb / 1024 / 1024))
        echo "pvc (via $pvc_pod): ${free_gb} GB free (min $MIN_FREE_GB)"
        if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
            echo "FAIL: PVC free space below threshold"
            status=1
        fi
    else
        echo "WARN: could not read df through $pvc_pod"
    fi
else
    echo "WARN: no running pod of ours to read the PVC through; pass one as \$2 (see pvc_inspect_pod.yaml)"
fi

exit $status
