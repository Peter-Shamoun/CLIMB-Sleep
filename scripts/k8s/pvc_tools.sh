#!/bin/sh
# PVC hygiene helpers. Run INSIDE a pod that mounts the PVC at /mnt/data
# (a training pod of ours, or scripts/k8s/pvc_inspect_pod.yaml):
#   kubectl exec <pod> -n dsc-capstone-25-26 -- sh /mnt/data/pvc_tools.sh df
#   ... usage            du per run dir under /mnt/data, largest first
#   ... prune <prefix>   delete optimizer.pt and non-final lm_model/ under
#                        checkpoint-* of FINISHED runs whose dir name starts
#                        with <prefix>. Only our own prefixes are accepted.
#   ... prune-dry <prefix>  same, print only.
# The script is POSIX sh so it runs in busybox.
set -eu

ROOT="${PVC_ROOT:-/mnt/data}"
# train.py writes runs to <output_dir>/checkpoints/<wandb project>/<run name>
RUNS_GLOB="$ROOT/checkpoints/*"
ALLOWED_PREFIXES="sh_expmt_ baseline_clm_ rr_expmt_ sh-"

usage_() {
    sed -n '2,11p' "$0"
    exit 1
}

allowed() {
    for p in $ALLOWED_PREFIXES; do
        case "$1" in
            "$p"*) return 0 ;;
        esac
    done
    return 1
}

cmd="${1:-}"
case "$cmd" in
    df)
        df -h "$ROOT"
        ;;
    usage)
        # one line per run directory, MB, largest first
        for d in $RUNS_GLOB/*/; do
            [ -d "$d" ] || continue
            du -sm "$d" 2>/dev/null
        done | sort -rn
        ;;
    prune|prune-dry)
        prefix="${2:-}"
        [ -n "$prefix" ] || usage_
        if ! allowed "$prefix"; then
            echo "refusing: prefix '$prefix' is not one of ours ($ALLOWED_PREFIXES)" >&2
            exit 2
        fi
        dry=0
        [ "$cmd" = "prune-dry" ] && dry=1
        for run in $RUNS_GLOB/"$prefix"*/; do
            [ -d "$run" ] || continue
            # a run is finished when its final lm_model/ export exists at the top
            # level; unfinished runs keep everything so they can resume.
            if [ ! -d "$run/lm_model" ]; then
                echo "skip (not finished): $run"
                continue
            fi
            # highest-numbered checkpoint is the final one; keep it intact
            last=$(ls -d "$run"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
            for ck in "$run"/checkpoint-*/; do
                [ -d "$ck" ] || continue
                step=$(basename "$ck" | sed 's/checkpoint-//')
                [ "$step" = "$last" ] && continue
                for victim in "$ck/optimizer.pt" "$ck/lm_model"; do
                    [ -e "$victim" ] || continue
                    if [ "$dry" = 1 ]; then
                        echo "would delete: $victim"
                    else
                        rm -rf "$victim"
                        echo "deleted: $victim"
                    fi
                done
            done
        done
        df -h "$ROOT"
        ;;
    *)
        usage_
        ;;
esac
