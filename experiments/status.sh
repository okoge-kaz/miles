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
: "${QSTAT_BIN:=qstat}"
: "${QSELECT_BIN:=qselect}"

find_logs() {
    find "${OUT}" -type f \( \
        -name '*.log' -o -name '*.o[0-9]*' -o -name '*.e[0-9]*' \
        -o -name '*.OU' -o -name '*.ER' \
    \) "$@" 2>/dev/null
}

newest_log() {
    find_logs -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-
}

case "${1:-}" in
    -f)
        log=$(newest_log)
        [[ -z "${log}" ]] && { echo "no logs under ${OUT}"; exit 1; }
        echo "==> ${log}"
        exec tail -f "${log}"
        ;;
    [0-9]*)
        jobid="$1"
        "${QSTAT_BIN}" -f "${jobid}" 2>/dev/null | head -30
        sequence="${jobid%%.*}"
        log=$(find "${OUT}" -type f \
            \( -name "*-${sequence}.log" -o -name "*.o${sequence}" -o -name "*.e${sequence}" \
                -o -name "${sequence}*.OU" -o -name "${sequence}*.ER" \) \
            -print -quit 2>/dev/null)
        [[ -z "${log}" ]] && { echo "no log for job ${jobid} under ${OUT}"; exit 1; }
        echo "==> ${log}"
        exec tail -f "${log}"
        ;;
esac

echo "=== queue (${USER}) ==="
queue_output="$("${QSTAT_BIN}" -u "${USER}" 2>/dev/null || true)"
printf '%s\n' "${queue_output}" | head -30
n=$("${QSELECT_BIN}" -u "${USER}" 2>/dev/null | wc -l)
states=$(printf '%s\n' "${queue_output}" | awk 'NR > 2 && NF >= 5 { print $5 }' \
    | sort | uniq -c | tr '\n' ' ')
echo "(${n} jobs; states: ${states:-none})"

echo
echo "=== recent logs ==="
find_logs -printf '%T@ %TY-%Tm-%Td %TH:%TM  %s  %p\n' \
    | sort -rn | head -10 | cut -d' ' -f2- || echo "(none yet)"

echo
echo "=== training runs ==="
for d in "${OUT}"/training/*/; do
    [[ -d "$d" ]] || continue
    count=$(find "$d" -maxdepth 1 -type f \
        \( -name '*.log' -o -name '*.o[0-9]*' -o -name '*.e[0-9]*' \
            -o -name '*.OU' -o -name '*.ER' \) 2>/dev/null | wc -l)
    printf "  %-40s %s log(s)\n" "$(basename "$d")" "${count}"
done

echo
echo "tail a job:  experiments/status.sh <jobid>     newest:  experiments/status.sh -f"
