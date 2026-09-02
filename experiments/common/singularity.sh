#!/bin/bash
# Singularity execution helpers for local and one-task-per-PBS-node launches.

_miles_singularity_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ! declare -F _miles_pbs_refresh_context >/dev/null; then
    # shellcheck source=experiments/common/pbs.sh
    source "${_miles_singularity_dir}/pbs.sh"
fi
unset _miles_singularity_dir

: "${SINGULARITY_BIN:=singularity}"
: "${ABCI_HPCX_MODULE:=hpcx/2.20}"

_miles_load_mpi_runtime() {
    local resolved

    if [[ -n "${MPIRUN_BIN:-}" ]]; then
        if [[ "${MPIRUN_BIN}" == */* ]]; then
            [[ -x "${MPIRUN_BIN}" ]] || {
                echo "miles_container_exec_all: MPIRUN_BIN is not executable: ${MPIRUN_BIN}" >&2
                return 2
            }
        else
            resolved="$(command -v "${MPIRUN_BIN}" 2>/dev/null || true)"
            [[ -n "${resolved}" ]] || {
                echo "miles_container_exec_all: MPIRUN_BIN was not found: ${MPIRUN_BIN}" >&2
                return 2
            }
            MPIRUN_BIN="${resolved}"
        fi
        return 0
    fi

    if [[ -n "${ABCI_HPCX_MODULE}" ]]; then
        if ! command -v module >/dev/null 2>&1; then
            [[ -r /etc/profile.d/modules.sh ]] || {
                echo "miles_container_exec_all: environment modules are unavailable" >&2
                return 2
            }
            # ABCI does not initialize modules in every non-interactive PBS shell.
            # shellcheck source=/dev/null
            source /etc/profile.d/modules.sh
        fi
        module load "${ABCI_HPCX_MODULE}" || return
    fi

    MPIRUN_BIN="$(command -v mpirun 2>/dev/null || true)"
    [[ -n "${MPIRUN_BIN}" ]] || {
        echo "miles_container_exec_all: mpirun is unavailable after loading ${ABCI_HPCX_MODULE:-no MPI module}" >&2
        return 2
    }
}

_miles_container_add_env() {
    local spec="$1"
    local item name
    local -a items=()

    case "${spec}" in
        ALL) return 0 ;;
        NIL|NONE)
            MILES_CONTAINER_COMMAND+=(--cleanenv)
            return 0
            ;;
        ALL,*) spec="${spec#ALL,}" ;;
        *) MILES_CONTAINER_COMMAND+=(--cleanenv) ;;
    esac

    IFS=',' read -r -a items <<< "${spec}"
    for item in "${items[@]}"; do
        [[ -n "${item}" ]] || continue
        if [[ "${item}" == *=* ]]; then
            MILES_CONTAINER_COMMAND+=(--env "${item}")
            continue
        fi
        name="${item}"
        if [[ -v "${name}" ]]; then
            MILES_CONTAINER_COMMAND+=(--env "${name}=${!name}")
        fi
    done
}

_miles_build_container_command() {
    local image="${MILES_CONTAINER_IMAGE:-${SINGULARITY_IMAGE:-${CONTAINER_IMAGE:-}}}"
    local cwd="${MILES_CONTAINER_CWD:-}"
    local nv_mode="${MILES_CONTAINER_NV:-auto}"
    local writable_tmpfs=1 no_home=1 cleanenv=0 fakeroot=0
    local option value export_spec=""
    local -a binds=() payload=()

    [[ -z "${MILES_CONTAINER_BINDS:-}" ]] || binds+=("${MILES_CONTAINER_BINDS}")

    while (( $# > 0 )); do
        option="$1"
        case "${option}" in
            --)
                shift
                payload=("$@")
                break
                ;;
            --image)
                [[ $# -ge 2 ]] || { echo "miles_container_exec: ${option} needs a value" >&2; return 2; }
                image="$2"
                shift 2
                ;;
            --image=*) image="${option#*=}"; shift ;;
            --bind)
                [[ $# -ge 2 ]] || { echo "miles_container_exec: ${option} needs a value" >&2; return 2; }
                binds+=("$2")
                shift 2
                ;;
            --bind=*) binds+=("${option#*=}"); shift ;;
            --cwd)
                [[ $# -ge 2 ]] || { echo "miles_container_exec: ${option} needs a value" >&2; return 2; }
                cwd="$2"
                shift 2
                ;;
            --cwd=*) cwd="${option#*=}"; shift ;;
            --env|--export)
                [[ $# -ge 2 ]] || { echo "miles_container_exec: ${option} needs a value" >&2; return 2; }
                export_spec="$2"
                shift 2
                ;;
            --env=*|--export=*) export_spec="${option#*=}"; shift ;;
            --nv) nv_mode=1; shift ;;
            --no-nv) nv_mode=0; shift ;;
            --writable-tmpfs) writable_tmpfs=1; shift ;;
            --read-only|--readonly) writable_tmpfs=0; shift ;;
            --no-home) no_home=1; shift ;;
            --mount-home) no_home=0; shift ;;
            --cleanenv) cleanenv=1; shift ;;
            --fakeroot) fakeroot=1; shift ;;
            --*) echo "miles_container_exec: unsupported option: ${option}" >&2; return 2 ;;
            *) payload=("$@"); break ;;
        esac
    done

    [[ -n "${image}" ]] || { echo "miles_container_exec: --image is required" >&2; return 2; }
    (( ${#payload[@]} > 0 )) || { echo "miles_container_exec: a command is required" >&2; return 2; }

    # Prevent Singularity from evaluating environment values as shell source.
    MILES_CONTAINER_COMMAND=("${SINGULARITY_BIN}" exec --no-eval)
    (( fakeroot == 0 )) || MILES_CONTAINER_COMMAND+=(--fakeroot)
    case "${nv_mode}" in
        1|yes|true|on) MILES_CONTAINER_COMMAND+=(--nv) ;;
        0|no|false|off) ;;
        auto)
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
                MILES_CONTAINER_COMMAND+=(--nv)
            fi
            ;;
        *) echo "miles_container_exec: MILES_CONTAINER_NV must be auto, 1, or 0" >&2; return 2 ;;
    esac
    (( writable_tmpfs == 0 )) || MILES_CONTAINER_COMMAND+=(--writable-tmpfs)
    (( no_home == 0 )) || MILES_CONTAINER_COMMAND+=(--no-home)
    (( cleanenv == 0 )) || MILES_CONTAINER_COMMAND+=(--cleanenv)
    [[ -z "${cwd}" ]] || MILES_CONTAINER_COMMAND+=(--pwd "${cwd}")
    for value in "${binds[@]}"; do
        [[ -z "${value}" ]] || MILES_CONTAINER_COMMAND+=(--bind "${value}")
    done
    [[ -z "${export_spec}" ]] || _miles_container_add_env "${export_spec}"
    MILES_CONTAINER_COMMAND+=("${image}" "${payload[@]}")
}

miles_container_exec() {
    _miles_build_container_command "$@" || return
    env \
        SINGULARITYENV_MILES_JOB_ID="${MILES_JOB_ID}" \
        SINGULARITYENV_MILES_JOB_NUM_NODES="${MILES_JOB_NUM_NODES}" \
        SINGULARITYENV_MILES_NODE_RANK="${MILES_NODE_RANK}" \
        SINGULARITYENV_MILES_SUBMIT_DIR="${MILES_SUBMIT_DIR}" \
        SINGULARITYENV_MILES_JOB_TMPDIR="${MILES_JOB_TMPDIR}" \
        SINGULARITYENV_PBS_JOBID="${PBS_JOBID:-${MILES_JOB_ID}}" \
        SINGULARITYENV_PBS_O_WORKDIR="${PBS_O_WORKDIR:-${MILES_SUBMIT_DIR}}" \
        "${MILES_CONTAINER_COMMAND[@]}"
}

_miles_unique_pbs_hosts() {
    local host short_host
    local -A seen_nodes=()

    [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]] || {
        _miles_short_hostname "$(hostname -s)"
        return
    }
    while IFS= read -r host || [[ -n "${host}" ]]; do
        short_host="$(_miles_short_hostname "${host}")"
        if [[ -n "${short_host}" && -z "${seen_nodes[${short_host}]+x}" ]]; then
            seen_nodes[${short_host}]=1
            printf '%s\n' "${host}"
        fi
    done < "${PBS_NODEFILE}"
}

miles_container_exec_all() {
    local rank name task_status status=0
    local status_root status_dir status_file env_file task_file mpi_task_file host_file
    local -a command=() hosts=()

    if [[ -z "${PBS_NODEFILE:-}" || ! -r "${PBS_NODEFILE}" ]]; then
        if (( MILES_JOB_NUM_NODES == 1 )); then
            miles_container_exec "$@"
            return
        fi
        echo "miles_container_exec_all: PBS_NODEFILE is required for a multi-node allocation" >&2
        return 2
    fi
    _miles_build_container_command "$@" || return
    command=("${MILES_CONTAINER_COMMAND[@]}")

    # Keep per-rank status files even though Open MPI normally propagates a
    # failure: they distinguish a payload failure from a launcher failure.
    status_root="${MILES_NODE_STATUS_ROOT:-${MILES_PBSDH_STATUS_ROOT:-${OUTPUT_DIR:-${MILES_SUBMIT_DIR}/experiments/outputs}/.pbs-task-status}}"
    status_dir="${status_root%/}/${MILES_JOB_ID}"
    mkdir -p -- "${status_dir}"
    chmod 0700 "${status_dir}"
    rm -f -- "${status_dir}"/*.status "${status_dir}"/*.partial-* 2>/dev/null || true

    # Snapshot the payload environment before loading host HPC-X. Its Open MPI
    # runtime variables (including FI_PROVIDER=mlx on ABCI) are needed only by
    # mpirun and must not leak through Singularity to Ray, torch, or NCCL. The
    # per-node wrapper enters `env -i` before sourcing this file.
    env_file="${status_dir}/environment.sh"
    (
        umask 0077
        : > "${env_file}"
        while IFS= read -r name; do
            [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
            printf 'export %s=%q\n' "${name}" "${!name}" >> "${env_file}"
        done < <(compgen -e | sort -u)
    )

    mapfile -t hosts < <(_miles_unique_pbs_hosts)
    if (( ${#hosts[@]} != MILES_JOB_NUM_NODES )); then
        echo "miles_container_exec_all: nodefile yielded ${#hosts[@]} unique nodes, expected ${MILES_JOB_NUM_NODES}" >&2
        rm -f -- "${env_file}"
        return 2
    fi
    host_file="${status_dir}/hosts"
    printf '%s\n' "${hosts[@]}" > "${host_file}"
    chmod 0600 "${host_file}"

    # Keep the MPI argv small by publishing the fixed wrapper and quoted
    # Singularity command beside the environment file.
    task_file="${status_dir}/task.sh"
    (
        umask 0077
        {
            printf '%s\n' \
                '#!/bin/bash' \
                'rank="${1:?missing node rank}"' \
                'status_dir="$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)"' \
                'env_file="${status_dir}/environment.sh"' \
                'source "${env_file}"' \
                'export MILES_NODE_RANK="${rank}"' \
                'export SINGULARITYENV_MILES_JOB_ID="${MILES_JOB_ID}"' \
                'export SINGULARITYENV_MILES_JOB_NUM_NODES="${MILES_JOB_NUM_NODES}"' \
                'export SINGULARITYENV_MILES_NODE_RANK="${rank}"' \
                'export SINGULARITYENV_MILES_SUBMIT_DIR="${MILES_SUBMIT_DIR}"' \
                'export SINGULARITYENV_MILES_JOB_TMPDIR="${MILES_JOB_TMPDIR}"' \
                'export SINGULARITYENV_PBS_JOBID="${PBS_JOBID:-${MILES_JOB_ID}}"' \
                'export SINGULARITYENV_PBS_O_WORKDIR="${PBS_O_WORKDIR:-${MILES_SUBMIT_DIR}}"'
            printf 'command=('
            printf ' %q' "${command[@]}"
            printf ' )\n'
            printf '%s\n' \
                '"${command[@]}"' \
                'status=$?' \
                'partial="${status_dir}/${rank}.partial-$$"' \
                'printf "%s\n" "${status}" > "${partial}"' \
                'mv -f -- "${partial}" "${status_dir}/${rank}.status"' \
                'exit "${status}"'
        } > "${task_file}"
        chmod 0700 "${task_file}"
    )

    mpi_task_file="${status_dir}/mpi-task.sh"
    (
        umask 0077
        {
            printf '%s\n' '#!/bin/bash' 'set -u'
            printf 'task_file=%q\n' "${task_file}"
            cat <<'EOF'
rank="${OMPI_COMM_WORLD_RANK:?HPC-X did not provide OMPI_COMM_WORLD_RANK}"
allowed="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
[[ -n "${allowed}" ]] || {
    echo "Miles MPI launcher rank=${rank}: Cpus_allowed_list is unavailable" >&2
    exit 70
}
cpu_count=0
IFS=',' read -r -a cpu_ranges <<< "${allowed}"
for cpu_range in "${cpu_ranges[@]}"; do
    if [[ "${cpu_range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        cpu_count=$(( cpu_count + BASH_REMATCH[2] - BASH_REMATCH[1] + 1 ))
    elif [[ "${cpu_range}" =~ ^[0-9]+$ ]]; then
        cpu_count=$(( cpu_count + 1 ))
    else
        echo "Miles MPI launcher rank=${rank}: invalid Cpus_allowed_list=${allowed}" >&2
        exit 70
    fi
done
printf 'Miles MPI launcher rank=%s host=%s Cpus_allowed_list=%s cpu_count=%s\n' \
    "${rank}" "$(hostname -s)" "${allowed}" "${cpu_count}"
expected="${MILES_NODE_CPUS_PER_TASK:-}"
if [[ -n "${expected}" && "${cpu_count}" != "${expected}" ]]; then
    echo "Miles MPI launcher rank=${rank}: expected ${expected} CPUs, got ${cpu_count} (${allowed})" >&2
    exit 70
fi
exec /usr/bin/env -i PATH=/usr/bin:/bin /bin/bash "${task_file}" "${rank}"
EOF
        } > "${mpi_task_file}"
        chmod 0700 "${mpi_task_file}"
    )

    _miles_load_mpi_runtime || {
        status=$?
        rm -f -- "${env_file}" "${task_file}" "${mpi_task_file}" "${host_file}"
        return "${status}"
    }

    # Node-level launchers must be unbound. Otherwise every Ray/torch worker
    # inherits the single-core mask selected for its parent MPI process.
    "${MPIRUN_BIN}" \
        --display-allocation \
        --display-map \
        --report-bindings \
        -hostfile "${host_file}" \
        -np "${MILES_JOB_NUM_NODES}" \
        -map-by ppr:1:node \
        -bind-to none \
        /bin/bash "${mpi_task_file}" || status=$?

    for (( rank = 0; rank < MILES_JOB_NUM_NODES; rank++ )); do
        status_file="${status_dir}/${rank}.status"
        if [[ ! -s "${status_file}" ]]; then
            echo "miles_container_exec_all: node rank ${rank} did not publish a task status" >&2
            (( status != 0 )) || status=1
            continue
        fi
        task_status="$(< "${status_file}")"
        if [[ ! "${task_status}" =~ ^[0-9]+$ ]] || (( task_status != 0 )); then
            echo "miles_container_exec_all: node rank ${rank} exited ${task_status}" >&2
            [[ "${task_status}" =~ ^[0-9]+$ ]] && status="${task_status}" || status=1
        fi
    done
    rm -f -- "${env_file}" "${task_file}" "${mpi_task_file}" "${host_file}"
    return "${status}"
}

miles_srun() {
    local nodes="" ntasks="" ntasks_per_node=1 cpus_per_task="" option value
    local nodes_set=0 ntasks_set=0
    local -a container_args=() payload=()

    while (( $# > 0 )); do
        option="$1"
        case "${option}" in
            --)
                shift
                payload=("$@")
                break
                ;;
            --nodes|--ntasks|--ntasks-per-node)
                [[ $# -ge 2 ]] || { echo "miles_srun: ${option} needs a value" >&2; return 2; }
                value="$2"
                shift 2
                case "${option}" in
                    --nodes) nodes="${value}"; nodes_set=1 ;;
                    --ntasks) ntasks="${value}"; ntasks_set=1 ;;
                    --ntasks-per-node) ntasks_per_node="${value}" ;;
                esac
                ;;
            --nodes=*) nodes="${option#*=}"; nodes_set=1; shift ;;
            --ntasks=*) ntasks="${option#*=}"; ntasks_set=1; shift ;;
            --ntasks-per-node=*) ntasks_per_node="${option#*=}"; shift ;;
            --image|--bind|--cwd|--env|--export)
                [[ $# -ge 2 ]] || { echo "miles_srun: ${option} needs a value" >&2; return 2; }
                container_args+=("${option}" "$2")
                shift 2
                ;;
            --image=*|--bind=*|--cwd=*|--env=*|--export=*)
                container_args+=("${option}")
                shift
                ;;
            --writable-tmpfs|--read-only|--readonly|--no-home|--mount-home|--nv|--no-nv|--cleanenv|--fakeroot)
                container_args+=("${option}")
                shift
                ;;
            --overlap|--exact|--exclusive|--label|--unbuffered|--kill-on-bad-exit)
                shift
                ;;
            --cpus-per-task)
                [[ $# -ge 2 ]] || { echo "miles_srun: ${option} needs a value" >&2; return 2; }
                cpus_per_task="$2"
                shift 2
                ;;
            --mem|--mem-per-cpu|--gpus|--gpus-per-node|--gpus-per-task|--gres|--cpu-bind|--gpu-bind|--mpi|--distribution)
                [[ $# -ge 2 ]] || { echo "miles_srun: ${option} needs a value" >&2; return 2; }
                shift 2
                ;;
            --cpus-per-task=*) cpus_per_task="${option#*=}"; shift ;;
            --mem=*|--mem-per-cpu=*|--gpus=*|--gpus-per-node=*|--gpus-per-task=*|--gres=*|--cpu-bind=*|--gpu-bind=*|--mpi=*|--distribution=*|--kill-on-bad-exit=*)
                shift
                ;;
            --*) echo "miles_srun: unsupported srun option: ${option}" >&2; return 2 ;;
            *) payload=("$@"); break ;;
        esac
    done

    if (( nodes_set == 0 && ntasks_set != 0 )); then
        nodes="${ntasks}"
    elif (( nodes_set != 0 && ntasks_set == 0 )); then
        ntasks="${nodes}"
    fi
    : "${nodes:=1}"
    : "${ntasks:=1}"
    [[ "${nodes}" =~ ^[1-9][0-9]*$ ]] || { echo "miles_srun: invalid --nodes=${nodes}" >&2; return 2; }
    [[ "${ntasks}" =~ ^[1-9][0-9]*$ ]] || { echo "miles_srun: invalid --ntasks=${ntasks}" >&2; return 2; }
    [[ -z "${cpus_per_task}" || "${cpus_per_task}" =~ ^[1-9][0-9]*$ ]] || {
        echo "miles_srun: invalid --cpus-per-task=${cpus_per_task}" >&2
        return 2
    }
    [[ "${ntasks_per_node}" == 1 ]] || {
        echo "miles_srun: PBS compatibility supports exactly one task per node" >&2
        return 2
    }
    (( ${#payload[@]} > 0 )) || { echo "miles_srun: a command is required" >&2; return 2; }

    container_args+=(-- "${payload[@]}")
    if (( nodes > 1 || ntasks > 1 )); then
        if (( nodes != MILES_JOB_NUM_NODES || ntasks != nodes )); then
            echo "miles_srun: multi-node execution must use the full ${MILES_JOB_NUM_NODES}-node PBS allocation" >&2
            return 2
        fi
        if [[ -n "${cpus_per_task}" ]]; then
            export MILES_NODE_CPUS_PER_TASK="${cpus_per_task}"
        fi
        miles_container_exec_all "${container_args[@]}"
    else
        miles_container_exec "${container_args[@]}"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    command="${1:-}"
    [[ $# -eq 0 ]] || shift
    case "${command}" in
        exec) miles_container_exec "$@" ;;
        exec-all) miles_container_exec_all "$@" ;;
        srun) miles_srun "$@" ;;
        *) echo "usage: singularity.sh {exec|exec-all|srun} [options] -- command" >&2; exit 2 ;;
    esac
fi
