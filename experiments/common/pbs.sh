#!/bin/bash
# PBS compatibility helpers shared by submission and allocation-side scripts.
# This file is safe to source; execute it directly to call pbs_submit.

: "${PBS_GPU_QUEUE:=R9920261300}"
: "${PBS_GPU_RESOURCE_TYPE:=rt_HF}"
: "${PBS_CPU_QUEUE:=${PBS_GPU_QUEUE}}"
: "${PBS_CPU_RESOURCE_TYPE:=rt_HC}"
: "${PBS_GPU_CPUS_PER_NODE:=192}"
: "${PBS_GPU_GPUS_PER_NODE:=8}"
: "${PBS_CPU_CPUS_PER_NODE:=32}"
: "${PBS_DEFAULT_WALLTIME:=24:00:00}"
: "${PBS_CONTAINER_WALLTIME:=00:30:00}"
: "${PBS_PREP_WALLTIME:=08:00:00}"
: "${PBS_DOWNLOAD_WALLTIME:=24:00:00}"
: "${PBS_GPU_PLACE:=scatter:excl}"
: "${PBS_CPU_PLACE:=scatter}"
if [[ -z "${PBS_QSUB_BIN:-}" ]]; then
    _miles_qsub_candidate="${PBS_EXEC:-/opt/pbs}/bin/qsub"
    if [[ -x "${_miles_qsub_candidate}" ]]; then
        PBS_QSUB_BIN="${_miles_qsub_candidate}"
    else
        PBS_QSUB_BIN=qsub
    fi
    unset _miles_qsub_candidate
fi

export PBS_GPU_QUEUE PBS_GPU_RESOURCE_TYPE PBS_CPU_QUEUE PBS_CPU_RESOURCE_TYPE
export PBS_GPU_CPUS_PER_NODE PBS_GPU_GPUS_PER_NODE PBS_CPU_CPUS_PER_NODE
export PBS_DEFAULT_WALLTIME PBS_CONTAINER_WALLTIME PBS_PREP_WALLTIME PBS_DOWNLOAD_WALLTIME

_miles_short_hostname() {
    local host="${1:-}"
    host="${host%%.*}"
    printf '%s\n' "${host}"
}

_miles_pbs_refresh_context() {
    local previous_job_id="${MILES_JOB_ID:-}"
    local native_job_id="${PBS_JOBID:-}"
    local current_host host short_host rank
    local -a unique_nodes=()
    local -A seen_nodes=()

    if [[ -n "${native_job_id}" ]]; then
        # Discard context inherited from the submit host, but retain the rank
        # explicitly assigned to an MPI child in this same PBS allocation.
        if [[ "${previous_job_id}" != "${native_job_id}" ]]; then
            unset MILES_JOB_NUM_NODES MILES_NODE_RANK
        fi
        MILES_JOB_ID="${native_job_id}"
    else
        MILES_JOB_ID="${previous_job_id:-local}"
    fi
    MILES_JOB_ID_SHORT="${MILES_JOB_ID%%.*}"
    MILES_SUBMIT_DIR="${PBS_O_WORKDIR:-${MILES_SUBMIT_DIR:-${PWD}}}"
    MILES_JOB_TMPDIR="${PBS_LOCALDIR:-${TMPDIR:-${MILES_JOB_TMPDIR:-/tmp}}}"

    if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
        while IFS= read -r host || [[ -n "${host}" ]]; do
            short_host="$(_miles_short_hostname "${host}")"
            [[ -n "${short_host}" ]] || continue
            if [[ -z "${seen_nodes[${short_host}]+x}" ]]; then
                seen_nodes[${short_host}]=1
                unique_nodes+=("${short_host}")
            fi
        done < "${PBS_NODEFILE}"
    fi

    if (( ${#unique_nodes[@]} > 0 )); then
        MILES_JOB_NUM_NODES="${MILES_JOB_NUM_NODES:-${#unique_nodes[@]}}"
        if [[ -z "${MILES_NODE_RANK:-}" ]]; then
            current_host="$(_miles_short_hostname "${MILES_PBS_HOSTNAME:-$(hostname -s)}")"
            rank=0
            for host in "${unique_nodes[@]}"; do
                if [[ "${host}" == "${current_host}" ]]; then
                    MILES_NODE_RANK="${rank}"
                    break
                fi
                rank=$(( rank + 1 ))
            done
            : "${MILES_NODE_RANK:=0}"
        fi
    else
        : "${MILES_JOB_NUM_NODES:=1}"
        : "${MILES_NODE_RANK:=0}"
    fi

    export MILES_JOB_ID MILES_JOB_ID_SHORT MILES_JOB_NUM_NODES MILES_NODE_RANK
    export MILES_SUBMIT_DIR MILES_JOB_TMPDIR
}

_miles_sbatch_header_value() {
    local script="$1"
    local key="$2"
    local line value

    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        [[ "${line}" == \#SBATCH* ]] || continue
        line="${line#\#SBATCH}"
        line="${line#"${line%%[![:space:]]*}"}"
        if [[ "${line}" == "--${key}="* ]]; then
            value="${line#*=}"
        elif [[ "${line}" == "--${key} "* ]]; then
            value="${line#* }"
        else
            continue
        fi
        value="${value%"${value##*[![:space:]]}"}"
        if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        printf '%s\n' "${value}"
        return 0
    done < "${script}"
    return 1
}

_miles_pbs_header_value() {
    local script="$1"
    local key="$2"
    local line resource value

    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        [[ "${line}" == \#PBS* ]] || continue
        line="${line#\#PBS}"
        line="${line#"${line%%[![:space:]]*}"}"
        case "${key}" in
            queue)
                [[ "${line}" =~ ^-q[[:space:]]+(.+)$ ]] && value="${BASH_REMATCH[1]}"
                ;;
            job-name)
                [[ "${line}" =~ ^-N[[:space:]]+(.+)$ ]] && value="${BASH_REMATCH[1]}"
                ;;
            output)
                [[ "${line}" =~ ^-o[[:space:]]+(.+)$ ]] && value="${BASH_REMATCH[1]}"
                ;;
            export)
                [[ "${line}" =~ ^-v[[:space:]]+(.+)$ ]] && value="${BASH_REMATCH[1]}"
                ;;
            nodes)
                if [[ "${line}" =~ ^-l[[:space:]]+select=([0-9]+)(:.*)?$ ]]; then
                    value="${BASH_REMATCH[1]}"
                fi
                ;;
            select)
                if [[ "${line}" =~ ^-l[[:space:]]+(select=[^,[:space:]]+) ]]; then
                    value="${BASH_REMATCH[1]}"
                fi
                ;;
            place)
                if [[ "${line}" =~ ^-l[[:space:]]+(place=[^,[:space:]]+) ]]; then
                    value="${BASH_REMATCH[1]#place=}"
                fi
                ;;
            walltime)
                if [[ "${line}" =~ ^-l[[:space:]]+walltime=([^,[:space:]]+) ]]; then
                    value="${BASH_REMATCH[1]}"
                elif [[ "${line}" =~ ^-l[[:space:]]+(.+,)?walltime=([^,[:space:]]+) ]]; then
                    value="${BASH_REMATCH[2]}"
                fi
                ;;
        esac
        if [[ -n "${value:-}" ]]; then
            value="${value%"${value##*[![:space:]]}"}"
            printf '%s\n' "${value}"
            return 0
        fi
    done < "${script}"
    return 1
}

_miles_pbs_walltime() {
    local value="$1"
    local days hours minutes seconds

    if [[ "${value}" =~ ^([0-9]+)-([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        days="${BASH_REMATCH[1]}"
        hours="${BASH_REMATCH[2]}"
        minutes="${BASH_REMATCH[3]}"
        seconds="${BASH_REMATCH[4]}"
        printf '%02d:%s:%s\n' "$(( 10#${days} * 24 + 10#${hours} ))" "${minutes}" "${seconds}"
    elif [[ "${value}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        printf '%02d:00:00\n' "$(( 10#${BASH_REMATCH[1]} * 24 + 10#${BASH_REMATCH[2]} ))"
    elif [[ "${value}" =~ ^[0-9]+$ ]]; then
        hours=$(( 10#${value} / 60 ))
        minutes=$(( 10#${value} % 60 ))
        printf '%02d:%02d:00\n' "${hours}" "${minutes}"
    else
        printf '%s\n' "${value}"
    fi
}

_miles_pbs_usage() {
    cat >&2 <<'EOF'
usage: pbs_submit [options] script [script arguments...]

Recognized options include -N/--job-name, --nodes, --profile,
--time, --output, --export, --dependency, and --parsable.
Legacy -A/--account, -p/--partition, and --qos values are accepted but are not
sent as PBS project/account options.
EOF
}

pbs_submit() {
    local account="" partition="" profile="" job_kind="" nodes="" walltime=""
    local job_name="" output="" export_spec="" dependency="" chdir=""
    local script="" script_absolute job_output job_id output_path output_dir
    local resource_export="" qsub_export=""
    local select_spec place native_select="" native_place=""
    local parsable=0 test_only=0 profile_set=0 nodes_set=0 option value
    local -a script_args=() qsub_args=()

    while (( $# > 0 )); do
        option="$1"
        case "${option}" in
            --)
                shift
                [[ $# -gt 0 ]] || { _miles_pbs_usage; return 2; }
                script="$1"
                shift
                script_args=("$@")
                break
                ;;
            -A|--account|-p|--partition|-N|--nodes|--time|--job-name|--output|-o|--export|--dependency|--profile|--job-kind|--chdir)
                [[ $# -ge 2 ]] || { echo "pbs_submit: ${option} needs a value" >&2; return 2; }
                value="$2"
                shift 2
                case "${option}" in
                    -A|--account) account="${value}" ;;
                    -p|--partition) partition="${value}" ;;
                    -N|--job-name) job_name="${value}" ;;
                    --nodes) nodes="${value}"; nodes_set=1 ;;
                    --time) walltime="${value}" ;;
                    --output|-o) output="${value}" ;;
                    --export) export_spec="${value}" ;;
                    --dependency) dependency="${value}" ;;
                    --profile) profile="${value}"; profile_set=1 ;;
                    --job-kind) job_kind="${value}" ;;
                    --chdir) chdir="${value}" ;;
                esac
                ;;
            -A?*) account="${option#-A}"; shift ;;
            -p?*) partition="${option#-p}"; shift ;;
            -N?*) job_name="${option#-N}"; shift ;;
            -o?*) output="${option#-o}"; shift ;;
            --account=*) account="${option#*=}"; shift ;;
            --partition=*) partition="${option#*=}"; shift ;;
            --nodes=*) nodes="${option#*=}"; nodes_set=1; shift ;;
            --time=*) walltime="${option#*=}"; shift ;;
            --job-name=*) job_name="${option#*=}"; shift ;;
            --output=*) output="${option#*=}"; shift ;;
            --export=*) export_spec="${option#*=}"; shift ;;
            --dependency=*) dependency="${option#*=}"; shift ;;
            --profile=*) profile="${option#*=}"; profile_set=1; shift ;;
            --job-kind=*) job_kind="${option#*=}"; shift ;;
            --chdir=*) chdir="${option#*=}"; shift ;;
            --qos)
                [[ $# -ge 2 ]] || { echo "pbs_submit: --qos needs a value" >&2; return 2; }
                shift 2
                ;;
            --qos=*) shift ;;
            --parsable) parsable=1; shift ;;
            --test-only) test_only=1; shift ;;
            --exclusive|--requeue|--no-requeue)
                shift
                ;;
            --cpus-per-task|--mem|--mem-per-cpu|--gres|--gpus|--gpus-per-node|--gpus-per-task|--ntasks|--ntasks-per-node|--signal)
                [[ $# -ge 2 ]] || { echo "pbs_submit: ${option} needs a value" >&2; return 2; }
                shift 2
                ;;
            --cpus-per-task=*|--mem=*|--mem-per-cpu=*|--gres=*|--gpus=*|--gpus-per-node=*|--gpus-per-task=*|--ntasks=*|--ntasks-per-node=*|--signal=*)
                shift
                ;;
            --*) echo "pbs_submit: unsupported option: ${option}" >&2; return 2 ;;
            -*) echo "pbs_submit: unsupported option: ${option}" >&2; return 2 ;;
            *)
                script="${option}"
                shift
                script_args=("$@")
                break
                ;;
        esac
    done

    [[ -n "${script}" ]] || { _miles_pbs_usage; return 2; }
    if [[ ! -r "${script}" ]]; then
        echo "pbs_submit: script is not readable: ${script}" >&2
        return 2
    fi

    script_absolute="$(cd -- "$(dirname -- "${script}")" && pwd)/$(basename -- "${script}")"
    native_select="$(_miles_pbs_header_value "${script}" select || true)"
    native_place="$(_miles_pbs_header_value "${script}" place || true)"
    : "${partition:=$(_miles_sbatch_header_value "${script}" partition || _miles_pbs_header_value "${script}" queue || true)}"
    : "${nodes:=$(_miles_sbatch_header_value "${script}" nodes || _miles_pbs_header_value "${script}" nodes || true)}"
    : "${walltime:=$(_miles_sbatch_header_value "${script}" time || _miles_pbs_header_value "${script}" walltime || true)}"
    : "${job_name:=$(_miles_sbatch_header_value "${script}" job-name || _miles_pbs_header_value "${script}" job-name || true)}"
    : "${output:=$(_miles_sbatch_header_value "${script}" output || _miles_pbs_header_value "${script}" output || true)}"
    : "${export_spec:=$(_miles_sbatch_header_value "${script}" export || _miles_pbs_header_value "${script}" export || true)}"
    : "${nodes:=1}"
    : "${job_name:=$(basename "${script}")}"
    job_name="${job_name%.sbatch}"

    if [[ -z "${job_kind}" ]]; then
        case "${script}" in
            */container/*) job_kind=container ;;
            */download/*) job_kind=download ;;
            */setup/*) job_kind=prep ;;
            *) job_kind=training ;;
        esac
    fi
    case "${job_kind}" in
        container) : "${walltime:=${PBS_CONTAINER_WALLTIME}}" ;;
        prep) : "${walltime:=${PBS_PREP_WALLTIME}}" ;;
        download) : "${walltime:=${PBS_DOWNLOAD_WALLTIME}}" ;;
        training) : "${walltime:=${PBS_DEFAULT_WALLTIME}}" ;;
        *) echo "pbs_submit: --job-kind must be container, prep, download, or training, got: ${job_kind}" >&2; return 2 ;;
    esac

    if [[ -z "${profile}" ]]; then
        if [[ "${native_select}" =~ (^|:)ngpus=([0-9]+) ]]; then
            if (( 10#${BASH_REMATCH[2]} > 0 )); then
                profile=gpu
            else
                profile=cpu
            fi
        elif [[ -n "${native_select}" ]]; then
            profile=cpu
        else
            case "${partition}" in
                cpu|cpu_*|cpu-*|rt_HC) profile=cpu ;;
                *) profile=gpu ;;
            esac
        fi
    fi

    case "${profile}" in
        gpu)
            if [[ -n "${native_select}" && "${nodes_set}" == 0 && "${profile_set}" == 0 ]]; then
                select_spec="${native_select}"
            else
                select_spec="select=${nodes}:ncpus=${PBS_GPU_CPUS_PER_NODE}:ngpus=${PBS_GPU_GPUS_PER_NODE}:mpiprocs=1"
            fi
            place="${native_place:-${PBS_GPU_PLACE}}"
            qsub_args=(-q "${PBS_GPU_QUEUE}")
            [[ -z "${PBS_GPU_RESOURCE_TYPE}" ]] || \
                resource_export="RTYPE=${PBS_GPU_RESOURCE_TYPE}"
            ;;
        cpu)
            if [[ -n "${native_select}" && "${nodes_set}" == 0 && "${profile_set}" == 0 ]]; then
                select_spec="${native_select}"
            else
                select_spec="select=${nodes}:ncpus=${PBS_CPU_CPUS_PER_NODE}:mpiprocs=1"
            fi
            place="${native_place:-${PBS_CPU_PLACE}}"
            qsub_args=(-q "${PBS_CPU_QUEUE}")
            if [[ "${PBS_CPU_QUEUE}" == "${PBS_GPU_QUEUE}" \
                && -n "${PBS_CPU_RESOURCE_TYPE}" ]]; then
                resource_export="RTYPE=${PBS_CPU_RESOURCE_TYPE}"
            fi
            ;;
        *) echo "pbs_submit: --profile must be cpu or gpu, got: ${profile}" >&2; return 2 ;;
    esac

    [[ "${nodes}" =~ ^[1-9][0-9]*$ ]] || { echo "pbs_submit: invalid node count: ${nodes}" >&2; return 2; }
    walltime="$(_miles_pbs_walltime "${walltime}")"
    chdir="${chdir:-${MILES_SUBMIT_DIR:-${PWD}}}"
    [[ "${chdir}" == /* ]] || chdir="${PWD}/${chdir}"

    qsub_args+=(
        -l "${select_spec}"
        -l "place=${place}"
        -l "walltime=${walltime}"
        -N "${job_name}"
        -j oe
    )

    if [[ -z "${output}" && -n "${OUTPUT_DIR:-}" ]]; then
        output="${OUTPUT_DIR%/}/"
    fi
    if [[ -n "${output}" ]]; then
        if [[ "${output}" == *%* ]]; then
            output_dir="$(dirname -- "${output}")"
            [[ "${output_dir}" == /* ]] || output_dir="${chdir}/${output_dir}"
            mkdir -p -- "${output_dir}"
            output_path="${output_dir%/}/"
        else
            output_path="${output}"
            [[ "${output_path}" == /* ]] || output_path="${chdir}/${output_path}"
            mkdir -p -- "$(dirname -- "${output_path}")"
        fi
        qsub_args+=(-o "${output_path}")
    fi

    if [[ -n "${resource_export}" && ",${export_spec}," == *,RTYPE,* ]] || \
        [[ -n "${resource_export}" && ",${export_spec}," == *,RTYPE=* ]]; then
        echo "pbs_submit: RTYPE is managed by the selected PBS resource profile" >&2
        return 2
    fi
    case "${export_spec}" in
        ""|NIL|NONE) ;;
        ALL) qsub_args+=(-V) ;;
        ALL,*) qsub_args+=(-V); qsub_export="${export_spec#ALL,}" ;;
        *) qsub_export="${export_spec}" ;;
    esac
    if [[ -n "${resource_export}" ]]; then
        qsub_export="${resource_export}${qsub_export:+,${qsub_export}}"
    fi
    [[ -z "${qsub_export}" ]] || qsub_args+=(-v "${qsub_export}")
    [[ -z "${dependency}" ]] || qsub_args+=(-W "depend=${dependency}")

    if (( test_only != 0 )); then
        printf 'cd %q && ' "${chdir}"
        printf '%q ' "${PBS_QSUB_BIN}" "${qsub_args[@]}" -- \
            /bin/bash -c 'cd -- "$1" && shift && exec /bin/bash "$@"' \
            miles-pbs-launch "${chdir}" "${script_absolute}" "${script_args[@]}"
        printf '\n'
        return 0
    fi

    job_output="$(
        cd -- "${chdir}" && \
            "${PBS_QSUB_BIN}" "${qsub_args[@]}" -- \
                /bin/bash -c 'cd -- "$1" && shift && exec /bin/bash "$@"' \
                miles-pbs-launch "${chdir}" "${script_absolute}" "${script_args[@]}"
    )" || return
    job_id="${job_output##*$'\n'}"
    job_id="${job_id#"${job_id%%[![:space:]]*}"}"
    job_id="${job_id%"${job_id##*[![:space:]]}"}"
    if [[ ! "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]]; then
        echo "pbs_submit: qsub returned an invalid job ID: ${job_output}" >&2
        return 1
    fi

    if (( parsable != 0 )); then
        printf '%s\n' "${job_id}"
    else
        printf 'Submitted PBS job %s\n' "${job_id}"
    fi
}

_miles_pbs_refresh_context

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pbs_submit "$@"
fi
