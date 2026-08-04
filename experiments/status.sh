#!/bin/bash
# Show what is queued/running and where its log is.
#
#   experiments/status.sh          queue + the 10 most recent logs
#   experiments/status.sh -f       follow the newest log
#   experiments/status.sh <jobid>  show one job's detail and follow its log
#
# Logs live under experiments/outputs/{download,convert,training/<run-name>}/.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT="${SCRIPT_DIR}/outputs"

newest_log() { find "${OUT}" -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

case "${1:-}" in
    -f)
        log=$(newest_log)
        [[ -z "${log}" ]] && { echo "no logs under ${OUT}"; exit 1; }
        echo "==> ${log}"
        exec tail -f "${log}"
        ;;
    [0-9]*)
        jobid="$1"
        scontrol show job "${jobid}" 2>/dev/null | head -20
        log=$(find "${OUT}" -name "*-${jobid}.log" | head -1)
        [[ -z "${log}" ]] && { echo "no log for job ${jobid} under ${OUT}"; exit 1; }
        echo "==> ${log}"
        exec tail -f "${log}"
        ;;
esac

echo "=== queue (${USER}) ==="
squeue -u "${USER}" -o "%.10i %.34j %.14P %.2t %.10M %.10L %R" | head -30
n=$(squeue -u "${USER}" -h | wc -l)
echo "(${n} jobs; states: $(squeue -u "${USER}" -h -o '%t' | sort | uniq -c | tr '\n' ' '))"

echo
echo "=== recent logs ==="
find "${OUT}" -name '*.log' -printf '%T@ %TY-%Tm-%Td %TH:%TM  %s  %p\n' 2>/dev/null \
    | sort -rn | head -10 | cut -d' ' -f2- || echo "(none yet)"

echo
echo "=== training runs ==="
for d in "${OUT}"/training/*/; do
    [[ -d "$d" ]] || continue
    printf "  %-40s %s log(s)\n" "$(basename "$d")" "$(ls "$d"/*.log 2>/dev/null | wc -l)"
done

echo
echo "tail a job:  experiments/status.sh <jobid>     newest:  experiments/status.sh -f"
