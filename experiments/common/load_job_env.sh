#!/bin/bash

# Load an explicit allowlist from a dotenv file without evaluating shell code.
# This is intended for scheduler job wrappers: credentials stay out of submit
# arguments and Ray runtime metadata while still being inherited from the job
# process. Values are never printed.

load_job_env() {
    local env_file=$1
    shift
    [[ -r "${env_file}" ]] || return 0

    local line key raw_value value allowed
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line=${line%$'\r'}
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "${line}" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]] || {
            echo "invalid dotenv assignment in ${env_file}" >&2
            return 1
        }
        key=${BASH_REMATCH[2]}
        raw_value=${BASH_REMATCH[3]}
        allowed=0
        for candidate in "$@"; do
            if [[ "${key}" == "${candidate}" ]]; then
                allowed=1
                break
            fi
        done
        [[ "${allowed}" == 1 ]] || continue
        # Explicit scheduler/job values take precedence over the repository
        # fallback. This also prevents an empty dotenv placeholder from
        # erasing a secret injected by the scheduler.
        [[ -v "${key}" ]] && continue

        value=${raw_value#"${raw_value%%[![:space:]]*}"}
        value=${value%"${value##*[![:space:]]}"}
        if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
            value=${value:1:${#value}-2}
        elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
            value=${value:1:${#value}-2}
        elif [[ "${value}" == *[[:space:]]* ]]; then
            echo "unquoted whitespace is not supported for ${key} in ${env_file}" >&2
            return 1
        fi
        printf -v "${key}" '%s' "${value}"
        export "${key}"
    done < "${env_file}"
}
