#!/bin/bash

# Build-time hardening for repository task images.  The trusted materializer
# supplies only validated schema/commit values.  This script is never invoked
# in the verifier image, which intentionally retains its hidden test assets.

set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/opt/miles-swe/root-home
export XDG_CONFIG_HOME=/opt/miles-swe/root-home/xdg
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0='*'
export GIT_CONFIG_KEY_1=core.fsmonitor
export GIT_CONFIG_VALUE_1=false
export GIT_CONFIG_KEY_2=core.hooksPath
export GIT_CONFIG_VALUE_2=/dev/null
export GIT_ATTR_NOSYSTEM=1
unset BASH_ENV CDPATH ENV GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR \
    GIT_DIR GIT_EXTERNAL_DIFF GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
    GIT_SSH GIT_SSH_COMMAND GIT_WORK_TREE SSH_ASKPASS

for command in awk dirname du find git getent grep readlink sed setpriv sha256sum sort stat wc; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "SWE agent image lacks required command: ${command}" >&2
        exit 1
    }
done
for command in /bin/bash /usr/bin/env /usr/bin/setpriv; do
    [[ -x "${command}" ]] || {
        echo "SWE agent image lacks required hardening executable: ${command}" >&2
        exit 1
    }
done

schema="${MILES_SWE_SCHEMA:?}"
base_commit="${MILES_SWE_BASE_COMMIT:-}"
runtime_policy="${MILES_SWE_RUNTIME_POLICY:-none}"
gold_commit="${MILES_SWE_GOLD_COMMIT:-}"
fresh_git=""
runtime_root=/opt/miles-swe/runtime
runtime_links=""
npm_runtime_paths=""
r2e_venv=""
r2e_python_runtime=""
r2e_runtime_links=""
r2e_runtime_imports=""
r2e_gold_blobs=""
case "${runtime_policy}" in
    none|npm-node-modules-v2|python-editable-metadata-v1) ;;
    *)
        echo "unsupported SWE agent runtime policy: ${runtime_policy}" >&2
        exit 1
        ;;
esac
cleanup_fresh_git() {
    if [[ -n "${fresh_git}" && "${fresh_git}" == /tmp/miles-swe-agent-git.* ]]; then
        rm -rf -- "${fresh_git}"
    fi
}
trap cleanup_fresh_git EXIT

if [[ -d /testbed/.git && ! -L /testbed/.git ]]; then
    repo=/testbed
else
    repo="$(pwd -P)"
    while [[ "${repo}" != / && ! -d "${repo}/.git" ]]; do
        repo="${repo%/*}"
        [[ -n "${repo}" ]] || repo=/
    done
fi
[[ -d "${repo}/.git" && ! -L "${repo}/.git" ]] || {
    echo "SWE source repository must use a real .git directory" >&2
    exit 1
}
repo="$(cd "${repo}" && pwd -P)"
source_gitdir="$(cd "${repo}/.git" && pwd -P)"
[[ "${source_gitdir}" == "${repo}/.git" ]] || {
    echo "SWE source Git directory escapes the repository" >&2
    exit 1
}
# Source-image local config is not authoritative. Remove every local config
# and attribute override before root invokes checkout/add, preventing filters,
# fsmonitor, hooks, includes, or external drivers from running as root.
install -o root -g root -m 0600 /dev/null "${source_gitdir}/config"
rm -f -- "${source_gitdir}/config.worktree" "${source_gitdir}/info/attributes"
export GIT_CONFIG_VALUE_0="${repo}"

if [[ "${schema}" == r2e-gym-v1 ]]; then
    [[ "${gold_commit}" =~ ^[0-9a-f]{40}$ ]] || {
        echo "R2E agent image requires the private gold-commit binding" >&2
        exit 1
    }
    git -C "${repo}" cat-file -e "${gold_commit}^{commit}" || {
        echo "R2E source image does not contain the bound gold commit" >&2
        exit 1
    }
    gold_parent_line="$(git -C "${repo}" show -s --format='%P' "${gold_commit}")"
    gold_parents=()
    if [[ -n "${gold_parent_line}" ]]; then
        read -r -a gold_parents <<<"${gold_parent_line}"
    fi
    [[ "${#gold_parents[@]}" == 1 ]] || {
        echo "R2E gold commit must have exactly one parent" >&2
        exit 1
    }
    if [[ -n "${base_commit}" && "${base_commit}" != "${gold_parents[0]}" ]]; then
        echo "published R2E base_commit does not match the gold parent" >&2
        exit 1
    fi
    base_commit="${gold_parents[0]}"
    [[ "$(git -C "${repo}" rev-parse HEAD)" == "${base_commit}" ]] || {
        echo "R2E source image HEAD is not the bound gold parent" >&2
        exit 1
    }
    install -d -o root -g root -m 0700 /opt/miles-swe
    r2e_gold_blobs=/opt/miles-swe/.r2e-gold-blobs
    : >"${r2e_gold_blobs}"
    r2e_changed_count=0
    while IFS= read -r -d '' relative; do
        [[ "${relative}" != *$'\r'* && "${relative}" != *$'\n'* ]] || {
            echo "R2E gold commit contains an unsupported path" >&2
            exit 1
        }
        r2e_changed_count=$((r2e_changed_count + 1))
        (( r2e_changed_count <= 10000 )) || {
            echo "R2E gold commit changes too many paths" >&2
            exit 1
        }
        blob="$(git -C "${repo}" rev-parse "${gold_commit}:${relative}" 2>/dev/null)" || continue
        [[ "$(git -C "${repo}" cat-file -t "${blob}")" == blob ]] || continue
        gold_file="/tmp/miles-r2e-gold-blob-${r2e_changed_count}"
        git -C "${repo}" cat-file blob "${blob}" >"${gold_file}"
        size="$(stat -c %s "${gold_file}")"
        digest="$(sha256sum "${gold_file}" | awk '{print $1}')"
        rm -f -- "${gold_file}"
        # Empty files carry no recoverable answer content and would match
        # unrelated filesystem placeholders, so they are not leak signatures.
        (( size == 0 )) || printf '%s %s\n' "${digest}" "${size}" >>"${r2e_gold_blobs}"
    done < <(git -C "${repo}" diff --name-only -z "${base_commit}" "${gold_commit}" --)
    (( r2e_changed_count > 0 )) || {
        echo "R2E gold commit has no changed paths" >&2
        exit 1
    }
    sort -u -o "${r2e_gold_blobs}" "${r2e_gold_blobs}"
fi
if [[ -n "${base_commit}" ]]; then
    base_tree="$(git -C "${repo}" rev-parse "${base_commit}^{tree}")"
    git -C "${repo}" checkout --detach "${base_commit}"
    git -C "${repo}" reset --hard "${base_commit}"
    [[ "$(git -C "${repo}" rev-parse HEAD)" == "${base_commit}" ]] || {
        echo "agent image did not reach the exact task base" >&2
        exit 1
    }
fi

if [[ "${runtime_policy}" == npm-node-modules-v2 ]]; then
    # SWE-ReBench publishes locked, task-specific base images with the npm
    # dependency tree preinstalled. Keep only this narrowly classified runtime
    # tree. The immutable swerebenchv2 image digest is the trust boundary; all
    # other ignored build products are removed below.
    [[ "${schema}" == swe-rebench-v2 ]] || {
        echo "npm runtime preservation is restricted to SWE-ReBench" >&2
        exit 1
    }
    [[ -f "${repo}/package-lock.json" && ! -L "${repo}/package-lock.json" ]] || {
        echo "npm runtime preservation requires a tracked package-lock.json" >&2
        exit 1
    }
    git -C "${repo}" ls-files --error-unmatch -- package-lock.json >/dev/null || {
        echo "npm lockfile is not part of the exact base tree" >&2
        exit 1
    }
    [[ -d "${repo}/node_modules" && ! -L "${repo}/node_modules" ]] || {
        echo "published npm runtime tree is missing or unsafe" >&2
        exit 1
    }
    [[ -z "$(git -C "${repo}" ls-files -- node_modules)" ]] || {
        echo "node_modules unexpectedly overlaps tracked source" >&2
        exit 1
    }
    runtime_bytes="$(du -sb -- "${repo}/node_modules" | awk '{print $1}')"
    runtime_entries="$(find "${repo}/node_modules" -xdev -printf . | wc -c)"
    [[ "${runtime_bytes}" =~ ^[0-9]+$ && "${runtime_entries}" =~ ^[0-9]+$ ]] || exit 1
    (( runtime_bytes > 0 && runtime_bytes <= 12884901888 )) || {
        echo "published npm runtime tree exceeds the 12 GiB bound" >&2
        exit 1
    }
    (( runtime_entries > 0 && runtime_entries <= 1000000 )) || {
        echo "published npm runtime tree exceeds the entry-count bound" >&2
        exit 1
    }
    if find "${repo}/node_modules" -xdev \( \
        -type b -o -type c -o -type p -o -type s \
    \) -print -quit | grep -q .; then
        echo "published npm runtime tree contains a special file" >&2
        exit 1
    fi
    if find "${repo}/node_modules" -xdev -type f -links +1 -print -quit | grep -q .; then
        echo "published npm runtime tree contains a hard-linked file" >&2
        exit 1
    fi
    install -d -o root -g root -m 0755 "${runtime_root}"
    mv -- "${repo}/node_modules" "${runtime_root}/node_modules"
    npm_runtime_paths=/tmp/miles-swe-npm-runtime-paths
    : >"${npm_runtime_paths}"
    if [[ -e "${repo}/dist" || -L "${repo}/dist" ]]; then
        [[ -d "${repo}/dist" && ! -L "${repo}/dist" ]] || {
            echo "published npm dist runtime is unsafe" >&2
            exit 1
        }
        [[ -z "$(git -C "${repo}" ls-files -- dist)" ]] || {
            echo "npm dist runtime overlaps tracked source" >&2
            exit 1
        }
        npm_dist_bytes="$(du -sb -- "${repo}/dist" | awk '{print $1}')"
        npm_dist_entries="$(find "${repo}/dist" -xdev -printf . | wc -c)"
        [[ "${npm_dist_bytes}" =~ ^[0-9]+$ && \
            "${npm_dist_entries}" =~ ^[0-9]+$ ]] || exit 1
        (( npm_dist_bytes > 0 && npm_dist_bytes <= 2147483648 )) || {
            echo "published npm dist runtime exceeds the 2 GiB bound" >&2
            exit 1
        }
        (( npm_dist_entries > 0 && npm_dist_entries <= 100000 )) || {
            echo "published npm dist runtime exceeds the entry-count bound" >&2
            exit 1
        }
        if find "${repo}/dist" -xdev \( \
            -type b -o -type c -o -type l -o -type p -o -type s \
        \) -print -quit | grep -q .; then
            echo "published npm dist runtime contains a symlink or special file" >&2
            exit 1
        fi
        if find "${repo}/dist" -xdev -type f -links +1 \
            -print -quit | grep -q .; then
            echo "published npm dist runtime contains a hard-linked file" >&2
            exit 1
        fi
        install -d -o root -g root -m 0755 "${runtime_root}/repo-overlay"
        mv -- "${repo}/dist" "${runtime_root}/repo-overlay/dist"
        chown -R root:root "${runtime_root}/repo-overlay/dist"
        chmod -R a+rX "${runtime_root}/repo-overlay/dist"
        chmod -R a-w "${runtime_root}/repo-overlay/dist"
        printf 'dist\n' >"${npm_runtime_paths}"
    fi
    npm_inventory=/opt/miles-swe/npm-repo-runtime.inventory
    : >"${npm_inventory}"
    if [[ -d "${runtime_root}/repo-overlay" ]]; then
        while IFS= read -r -d '' npm_runtime_file; do
            relative="${npm_runtime_file#"${runtime_root}/repo-overlay/"}"
            [[ "${relative}" != "${npm_runtime_file}" ]] || exit 1
            size="$(stat -c %s "${npm_runtime_file}")"
            digest="$(sha256sum "${npm_runtime_file}" | awk '{print $1}')"
            printf '%s\0%s\0%s\0' "${relative}" "${size}" "${digest}" \
                >>"${npm_inventory}"
        done < <(find "${runtime_root}/repo-overlay" -xdev -type f \
            -print0 | sort -z)
    fi
    sha256sum "${npm_inventory}" | awk '{print $1}' \
        >/opt/miles-swe/npm-repo-runtime.inventory.sha256
    chmod 0444 "${npm_inventory}" \
        /opt/miles-swe/npm-repo-runtime.inventory.sha256
    [[ -x /opt/miles-swe/seal_playwright_runtime.sh ]] || {
        echo "Playwright runtime sealing helper is missing" >&2
        exit 1
    }
    /opt/miles-swe/seal_playwright_runtime.sh
fi

if [[ "${runtime_policy}" == python-editable-metadata-v1 ]]; then
    [[ "${schema}" == swe-gym ]] || {
        echo "Python editable-metadata preservation is restricted to SWE-Gym" >&2
        exit 1
    }
    runtime_links=/tmp/miles-swe-python-runtime-links
    : >"${runtime_links}"
    install -d -o root -g root -m 0755 "${runtime_root}/python-editable"
    runtime_entry_count=0
    runtime_total_bytes=0
    while IFS= read -r -d '' metadata_dir; do
        relative="${metadata_dir#"${repo}/"}"
        [[ -n "${relative}" && "${relative}" != "${metadata_dir}" && \
            "${relative}" != *$'\r'* && "${relative}" != *$'\n'* && \
            "${relative}" != *$'\t'* ]] || {
            echo "published Python metadata has an unsafe path" >&2
            exit 1
        }
        [[ -z "$(git -C "${repo}" ls-files -- "${relative}")" ]] || {
            echo "Python editable metadata overlaps tracked source" >&2
            exit 1
        }
        metadata_bytes="$(du -sb -- "${metadata_dir}" | awk '{print $1}')"
        metadata_entries="$(find "${metadata_dir}" -xdev -printf . | wc -c)"
        [[ "${metadata_bytes}" =~ ^[0-9]+$ && \
            "${metadata_entries}" =~ ^[0-9]+$ ]] || exit 1
        runtime_entry_count=$((runtime_entry_count + metadata_entries))
        runtime_total_bytes=$((runtime_total_bytes + metadata_bytes))
        (( runtime_entry_count <= 100000 && runtime_total_bytes <= 1073741824 )) || {
            echo "published Python editable metadata exceeds its bound" >&2
            exit 1
        }
        if find "${metadata_dir}" -xdev \( \
            -type b -o -type c -o -type p -o -type s \
        \) -print -quit | grep -q .; then
            echo "published Python metadata contains a special file" >&2
            exit 1
        fi
        if find "${metadata_dir}" -xdev -type f -links +1 \
            -print -quit | grep -q .; then
            echo "published Python metadata contains a hard-linked file" >&2
            exit 1
        fi
        target="${runtime_root}/python-editable/${relative}"
        install -d -o root -g root -m 0755 "$(dirname -- "${target}")"
        mv -- "${metadata_dir}" "${target}"
        printf '%s\n' "${relative}" >>"${runtime_links}"
    done < <(find "${repo}" -xdev -type d -name '*.egg-info' -prune -print0)
fi

if [[ "${schema}" == r2e-gym-v1 ]]; then
    [[ -d "${repo}/.venv" && ! -L "${repo}/.venv" ]] || {
        echo "R2E source image lacks its installed virtual environment" >&2
        exit 1
    }
    r2e_runtime_bytes="$(du -sb -- "${repo}/.venv" | awk '{print $1}')"
    r2e_runtime_entries="$(find "${repo}/.venv" -xdev -printf . | wc -c)"
    [[ "${r2e_runtime_bytes}" =~ ^[0-9]+$ && "${r2e_runtime_entries}" =~ ^[0-9]+$ ]] || exit 1
    (( r2e_runtime_bytes > 0 && r2e_runtime_bytes <= 12884901888 )) || {
        echo "R2E virtual environment exceeds the 12 GiB bound" >&2
        exit 1
    }
    (( r2e_runtime_entries > 0 && r2e_runtime_entries <= 1000000 )) || {
        echo "R2E virtual environment exceeds the entry-count bound" >&2
        exit 1
    }
    if find "${repo}/.venv" -xdev \( \
        -type b -o -type c -o -type p -o -type s \
    \) -print -quit | grep -q .; then
        echo "R2E virtual environment contains a special file" >&2
        exit 1
    fi
    r2e_python="$(readlink -f -- "${repo}/.venv/bin/python")" || {
        echo "R2E virtual environment lacks a resolvable Python interpreter" >&2
        exit 1
    }
    case "${r2e_python}" in
        /root/.local/share/uv/python/*/bin/python*) ;;
        *)
            echo "R2E Python interpreter is outside the supported uv runtime" >&2
            exit 1
            ;;
    esac
    r2e_python_source="$(dirname -- "$(dirname -- "${r2e_python}")")"
    r2e_python_name="${r2e_python_source#/root/.local/share/uv/python/}"
    [[ -n "${r2e_python_name}" && "${r2e_python_name}" != */* && \
        -d "${r2e_python_source}" && ! -L "${r2e_python_source}" ]] || {
        echo "R2E uv Python runtime has an unsafe layout" >&2
        exit 1
    }
    r2e_python_bytes="$(du -sb -- "${r2e_python_source}" | awk '{print $1}')"
    r2e_python_entries="$(find "${r2e_python_source}" -xdev -printf . | wc -c)"
    [[ "${r2e_python_bytes}" =~ ^[0-9]+$ && \
        "${r2e_python_entries}" =~ ^[0-9]+$ ]] || exit 1
    (( r2e_python_bytes > 0 && r2e_python_bytes <= 4294967296 )) || {
        echo "R2E uv Python runtime exceeds the 4 GiB bound" >&2
        exit 1
    }
    (( r2e_python_entries > 0 && r2e_python_entries <= 250000 )) || {
        echo "R2E uv Python runtime exceeds the entry-count bound" >&2
        exit 1
    }
    if find "${r2e_python_source}" -xdev \( \
        -type b -o -type c -o -type p -o -type s \
    \) -print -quit | grep -q .; then
        echo "R2E uv Python runtime contains a special file" >&2
        exit 1
    fi
    install -d -o root -g root -m 0755 "${runtime_root}"
    mv -- "${repo}/.venv" "${runtime_root}/venv"
    r2e_venv="${runtime_root}/venv"
    mv -- "${r2e_python_source}" "${runtime_root}/python"
    r2e_python_runtime="${runtime_root}/python"
    ln -sfn -- "${r2e_python_runtime}/bin/${r2e_python##*/}" \
        "${r2e_venv}/bin/python"
    if [[ -f "${r2e_venv}/pyvenv.cfg" && ! -L "${r2e_venv}/pyvenv.cfg" ]]; then
        sed -i "s|${r2e_python_source}|${r2e_python_runtime}|g" \
            "${r2e_venv}/pyvenv.cfg"
    fi
    # uv normally hard-links installed packages from its root cache. Remove
    # that publisher cache after relocating the two required runtime trees.
    # Hard-link containment is checked after the exact editable overlay below
    # has also been relocated.
    rm -rf -- /root/.cache /root/.local/share/uv
    rm -rf -- /r2e_tests "${repo}/r2e_tests"
    rm -f -- "${repo}/run_tests.sh"
    secret_names=(
        syn_issue.json
        expected_test_output.json
        execution_result.json
        parsed_commit.json
        modified_files.json
        modified_entities.json
    )
    for secret_name in "${secret_names[@]}"; do
        rm -f -- "/${secret_name}" "/root/${secret_name}" "/testbed/${secret_name}"
    done
    find "${repo}" -type f \( \
        -name syn_issue.json -o -name expected_test_output.json -o \
        -name execution_result.json -o -name parsed_commit.json -o \
        -name modified_files.json -o -name modified_entities.json \
    \) -delete
    # R2E Python images commonly use an editable install. Preserve only the
    # narrow publisher-built overlay required by that install: extension
    # modules, generated version modules, and distribution metadata. It is
    # inventoried under /opt and linked back without entering the synthetic
    # exact-base Git tree. The admitted immutable image digest remains the
    # provenance boundary for these generated bytes.
    r2e_overlay="${runtime_root}/repo-overlay"
    r2e_runtime_links=/tmp/miles-r2e-runtime-links
    r2e_runtime_imports=/tmp/miles-r2e-runtime-imports
    : >"${r2e_runtime_links}"
    : >"${r2e_runtime_imports}"
    install -d -o root -g root -m 0755 "${r2e_overlay}"
    while IFS= read -r -d '' metadata_dir; do
        relative="${metadata_dir#"${repo}/"}"
        [[ -n "${relative}" && "${relative}" != "${metadata_dir}" && \
            "${relative}" =~ ^[A-Za-z0-9._/@+=-]+$ ]] || {
            echo "R2E editable metadata has an unsafe path" >&2
            exit 1
        }
        if [[ -n "$(git -C "${repo}" ls-files -- "${relative}")" ]]; then
            continue
        fi
        if find "${metadata_dir}" -xdev \( \
            -type b -o -type c -o -type p -o -type s -o -type l \
        \) -print -quit | grep -q .; then
            echo "R2E editable metadata contains a special file" >&2
            exit 1
        fi
        target="${r2e_overlay}/${relative}"
        install -d -o root -g root -m 0755 "$(dirname -- "${target}")"
        mv -- "${metadata_dir}" "${target}"
        printf '%s\n' "${relative}" >>"${r2e_runtime_links}"
        if [[ -f "${target}/top_level.txt" && ! -L "${target}/top_level.txt" ]]; then
            while IFS= read -r module; do
                [[ -z "${module}" ]] && continue
                [[ "${module}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
                    echo "R2E editable metadata has an unsafe import name" >&2
                    exit 1
                }
                printf '%s\n' "${module}" >>"${r2e_runtime_imports}"
            done <"${target}/top_level.txt"
        fi
    done < <(find "${repo}" -xdev -path "${source_gitdir}" -prune -o \
        -type d -name '*.egg-info' -prune -print0)
    while IFS= read -r -d '' generated_file; do
        relative="${generated_file#"${repo}/"}"
        [[ -n "${relative}" && "${relative}" != "${generated_file}" && \
            "${relative}" =~ ^[A-Za-z0-9._/@+=-]+$ ]] || {
            echo "R2E generated runtime file has an unsafe path" >&2
            exit 1
        }
        if [[ -n "$(git -C "${repo}" ls-files -- "${relative}")" ]]; then
            continue
        fi
        [[ -f "${generated_file}" && ! -L "${generated_file}" && \
            "$(stat -c %h "${generated_file}")" == 1 ]] || {
            echo "R2E generated runtime file is not an isolated untracked file" >&2
            exit 1
        }
        target="${r2e_overlay}/${relative}"
        install -d -o root -g root -m 0755 "$(dirname -- "${target}")"
        mv -- "${generated_file}" "${target}"
        printf '%s\n' "${relative}" >>"${r2e_runtime_links}"
    done < <(find "${repo}" -xdev -path "${source_gitdir}" -prune -o \
        -type f \( -name '*.so' -o -name version.py \) -print0)
    sort -u -o "${r2e_runtime_links}" "${r2e_runtime_links}"
    sort -u -o "${r2e_runtime_imports}" "${r2e_runtime_imports}"
    [[ -s "${r2e_runtime_links}" && -s "${r2e_runtime_imports}" ]] || {
        echo "R2E source image lacks an admissible editable runtime inventory" >&2
        exit 1
    }
    r2e_overlay_bytes="$(du -sb -- "${r2e_overlay}" | awk '{print $1}')"
    r2e_overlay_entries="$(find "${r2e_overlay}" -xdev -printf . | wc -c)"
    [[ "${r2e_overlay_bytes}" =~ ^[0-9]+$ && \
        "${r2e_overlay_entries}" =~ ^[0-9]+$ ]] || exit 1
    (( r2e_overlay_bytes > 0 && r2e_overlay_bytes <= 2147483648 )) || {
        echo "R2E editable runtime overlay exceeds the 2 GiB bound" >&2
        exit 1
    }
    (( r2e_overlay_entries > 0 && r2e_overlay_entries <= 100000 )) || {
        echo "R2E editable runtime overlay exceeds the entry-count bound" >&2
        exit 1
    }
    r2e_inventory=/opt/miles-swe/r2e-runtime-inventory
    : >"${r2e_inventory}"
    while IFS= read -r -d '' runtime_file; do
        relative="${runtime_file#"${runtime_root}/"}"
        runtime_size="$(stat -c %s "${runtime_file}")"
        runtime_digest="$(sha256sum "${runtime_file}" | awk '{print $1}')"
        printf '%s %s %q\n' \
            "${runtime_digest}" "${runtime_size}" "${relative}" \
            >>"${r2e_inventory}"
    done < <(find "${runtime_root}" -xdev -type f -print0 | sort -z)
    [[ -s "${r2e_inventory}" ]] || exit 1
    sha256sum "${r2e_inventory}" | awk '{print $1}' \
        >/opt/miles-swe/r2e-runtime-inventory.sha256
    chmod 0444 "${r2e_inventory}" /opt/miles-swe/r2e-runtime-inventory.sha256
    while read -r inode link_count; do
        inside_count="$(find "${runtime_root}" -xdev -type f -inum "${inode}" -printf . | wc -c)"
        [[ "${inside_count}" == "${link_count}" ]] || {
            echo "R2E runtime hard link escapes its sealed /opt tree" >&2
            exit 1
        }
    done < <(find "${runtime_root}" -xdev -type f -links +1 \
        -printf '%i %n\n' | sort -u)
    # The published image checks out the bound gold parent, but retains the
    # complete Git history plus task-specific tests. Keep only the separately
    # validated base-installed virtualenv; every other ignored artifact goes.
    git -C "${repo}" clean -ffdx
    find "${repo}" -type f -name '*.pyc' -delete
    find "${repo}" -type d -name __pycache__ -prune -exec rm -rf -- {} +
else
    # Repository-SWE source images are untrusted task inputs. Untracked files
    # can contain hidden tests, expected output, or an oracle patch, so never
    # preserve them in the agent image, including files hidden by ignore or
    # info/exclude rules. Runtime dependencies must live outside the repository
    # or be represented by the exact tracked base tree.
    git -C "${repo}" clean -ffdx
    while IFS= read -r -d '' path; do
        echo "agent source repository contains forbidden untracked content" >&2
        exit 1
    done < <(git -C "${repo}" ls-files --others -z)
fi

[[ -n "${base_commit}" ]] || {
    echo "SWE agent image requires an exact base commit" >&2
    exit 1
}
git -C "${repo}" diff --quiet --no-ext-diff "${base_commit}" -- || {
    echo "agent worktree differs from the exact task base" >&2
    exit 1
}
git -C "${repo}" diff --cached --quiet --no-ext-diff "${base_commit}" -- || {
    echo "agent index differs from the exact task base" >&2
    exit 1
}
while IFS= read -r -d '' _untracked_path; do
    echo "agent source repository retains untracked content" >&2
    exit 1
done < <(git -C "${repo}" ls-files --others -z)

# Copy only the verified base tree and its reachable tree/blob objects into a
# new object database. Re-adding the worktree is not exact: ignore rules,
# attributes, filters, gitlinks, and file modes can change the resulting tree.
# Starting from a tree object preserves the byte-for-byte published base while
# excluding the source commit and all history.
fresh_git="$(mktemp -d /tmp/miles-swe-agent-git.XXXXXX)"
git init --bare -q --template= "${fresh_git}"
printf '%s\n' "${base_tree}" \
    | git -C "${repo}" pack-objects --stdout --revs \
    | git --git-dir="${fresh_git}" index-pack --stdin --fix-thin >/dev/null
git --git-dir="${fresh_git}" cat-file -e "${base_tree}^{tree}"
git --git-dir="${fresh_git}" read-tree "${base_tree}"
commit="$(printf 'Task base state\n' | git --git-dir="${fresh_git}" \
    -c user.email=env@miles.invalid -c user.name=Miles commit-tree "${base_tree}")"
git --git-dir="${fresh_git}" symbolic-ref HEAD refs/heads/__miles_swe_base
git --git-dir="${fresh_git}" update-ref refs/heads/__miles_swe_base "${commit}"
if git --git-dir="${fresh_git}" cat-file -e "${base_commit}^{commit}" 2>/dev/null; then
    echo "source-image base commit remains readable after repository reinitialization" >&2
    exit 1
fi
if [[ -n "${gold_commit}" ]] && \
    git --git-dir="${fresh_git}" cat-file -e "${gold_commit}^{commit}" 2>/dev/null; then
    echo "gold commit remains readable after repository reinitialization" >&2
    exit 1
fi
if [[ -n "$(git --git-dir="${fresh_git}" fsck \
    --no-reflogs --unreachable --no-progress 2>&1)" ]]; then
    echo "agent repository contains unreachable Git objects" >&2
    exit 1
fi

install -d -m 0755 /opt/miles-swe
rm -rf -- /opt/miles-swe/agent-git
rm -rf -- "${repo}/.git"
mv -- "${fresh_git}" /opt/miles-swe/agent-git
fresh_git=""
# The worktree gitfile is a convenience for agent-side read-only Git commands;
# the root collector never trusts it. The agent may replace this file because
# the worktree root is writable, but the external Git dir remains root-owned.
printf 'gitdir: /opt/miles-swe/agent-git\n' >"${repo}/.git"
printf '%s\n' "${repo}" >/opt/miles-swe/workdir
printf '%s\n' /opt/miles-swe/agent-git >/opt/miles-swe/gitdir
chmod 0444 /opt/miles-swe/workdir /opt/miles-swe/gitdir

printf '%s\n' "${runtime_policy}" >/opt/miles-swe/runtime-policy
chmod 0444 /opt/miles-swe/runtime-policy
if [[ "${runtime_policy}" == npm-node-modules-v2 ]]; then
    # Symlink resolution is checked after relocation because npm .bin links are
    # relative to node_modules. Workspace links may target the exact base tree;
    # every other escape is rejected.
    while IFS= read -r -d '' link; do
        resolved="$(readlink -f -- "${link}")" || {
            echo "published npm runtime contains a dangling symlink" >&2
            exit 1
        }
        case "${resolved}" in
            "${runtime_root}/node_modules"/*|"${repo}"/*) ;;
            *)
                echo "published npm runtime symlink escapes the sealed runtime" >&2
                exit 1
                ;;
        esac
    done < <(find "${runtime_root}/node_modules" -xdev -type l -print0)
    ln -s "${runtime_root}/node_modules" "${repo}/node_modules"
    install -d -o root -g root -m 0755 /opt/miles-swe/agent-git/info
    printf '/node_modules\n' >/opt/miles-swe/agent-git/info/exclude
    while IFS= read -r relative; do
        [[ "${relative}" == dist && ! -e "${repo}/${relative}" && \
            -d "${runtime_root}/repo-overlay/${relative}" ]] || {
            echo "npm repository runtime path cannot be restored safely" >&2
            exit 1
        }
        mv -- "${runtime_root}/repo-overlay/${relative}" "${repo}/${relative}"
        printf '/%s\n' "${relative}" >>/opt/miles-swe/agent-git/info/exclude
    done <"${npm_runtime_paths}"
    if [[ -d "${runtime_root}/repo-overlay" ]]; then
        rmdir -- "${runtime_root}/repo-overlay"
    fi
    install -o root -g root -m 0444 "${npm_runtime_paths}" \
        /opt/miles-swe/npm-repo-runtime-paths
    rm -f -- "${npm_runtime_paths}"
    npm_runtime_paths=""
    chmod 0444 /opt/miles-swe/agent-git/info/exclude
fi
if [[ "${runtime_policy}" == python-editable-metadata-v1 ]]; then
    install -d -o root -g root -m 0755 /opt/miles-swe/agent-git/info
    : >/opt/miles-swe/agent-git/info/exclude
    while IFS= read -r relative; do
        [[ -n "${relative}" ]] || continue
        target="${runtime_root}/python-editable/${relative}"
        while IFS= read -r -d '' link; do
            resolved="$(readlink -f -- "${link}")" || {
                echo "published Python metadata contains a dangling symlink" >&2
                exit 1
            }
            case "${resolved}" in
                "${runtime_root}/python-editable"/*|"${repo}"/*) ;;
                *)
                    echo "published Python metadata symlink escapes its runtime" >&2
                    exit 1
                    ;;
            esac
        done < <(find "${target}" -xdev -type l -print0)
        ln -s "${target}" "${repo}/${relative}"
        printf '/%s\n' "${relative}" >>/opt/miles-swe/agent-git/info/exclude
    done <"${runtime_links}"
    install -o root -g root -m 0444 "${runtime_links}" \
        /opt/miles-swe/runtime-links
    rm -f -- "${runtime_links}"
    runtime_links=""
    chmod 0444 /opt/miles-swe/agent-git/info/exclude
fi
if [[ -n "${r2e_venv}" ]]; then
    while IFS= read -r -d '' link; do
        resolved="$(readlink -f -- "${link}")" || {
            echo "R2E virtual environment contains a dangling symlink" >&2
            exit 1
        }
        case "${resolved}" in
            "${runtime_root}"/*|"${repo}"/*|/usr/bin/*|/bin/*|/lib/*|/lib64/*) ;;
            *)
                echo "R2E virtualenv symlink escapes its runtime allowlist" >&2
                exit 1
                ;;
        esac
    done < <(find "${runtime_root}" -xdev -type l -print0)
    ln -s "${r2e_venv}" "${repo}/.venv"
    install -d -o root -g root -m 0755 /opt/miles-swe/agent-git/info
    printf '/.venv\n' >/opt/miles-swe/agent-git/info/exclude
    while IFS= read -r relative; do
        [[ -n "${relative}" ]] || continue
        target="${runtime_root}/repo-overlay/${relative}"
        [[ -e "${target}" && ! -e "${repo}/${relative}" && \
            -d "$(dirname -- "${repo}/${relative}")" ]] || {
            echo "R2E editable runtime link cannot be restored safely" >&2
            exit 1
        }
        ln -s "${target}" "${repo}/${relative}"
        printf '/%s\n' "${relative}" >>/opt/miles-swe/agent-git/info/exclude
    done <"${r2e_runtime_links}"
    install -o root -g root -m 0444 "${r2e_runtime_links}" \
        /opt/miles-swe/r2e-runtime-links
    install -o root -g root -m 0444 "${r2e_runtime_imports}" \
        /opt/miles-swe/r2e-runtime-imports
    rm -f -- "${r2e_runtime_links}" "${r2e_runtime_imports}"
    r2e_runtime_links=""
    r2e_runtime_imports=""
    chmod 0444 /opt/miles-swe/agent-git/info/exclude
fi

if [[ "${schema}" == r2e-gym-v1 ]]; then
    # The immutable publisher image is the runtime dependency trust boundary.
    # Independently reject two concrete oracle-leak classes after removing its
    # tests and complete source Git database: exact gold-file content outside
    # the base worktree, and the gold commit text anywhere in the visible
    # rootfs. Arbitrarily transformed copies require a separately trusted
    # clean-room runtime rebuild and remain outside this attestation.
    # Purge publisher scratch content before scanning; /tmp is model-visible
    # and may otherwise contain an arbitrarily named answer artifact.
    find /tmp -xdev -mindepth 1 -delete
    chmod 1777 /tmp
    r2e_blob_leak=0
    while read -r expected_digest expected_size; do
        [[ "${expected_digest}" =~ ^[0-9a-f]{64}$ && "${expected_size}" =~ ^[0-9]+$ ]] || {
            echo "invalid internal R2E gold-blob signature" >&2
            exit 1
        }
        while IFS= read -r -d '' candidate; do
            case "${candidate}" in
                "${repo}"/*|/proc/*|/sys/*|/dev/*|/run/*) continue ;;
            esac
            candidate_digest="$(sha256sum "${candidate}" 2>/dev/null | awk '{print $1}')" || continue
            if [[ "${candidate_digest}" == "${expected_digest}" ]]; then
                echo "R2E agent rootfs retains exact gold-file content: ${candidate}" >&2
                r2e_blob_leak=1
            fi
        done < <(find / -xdev -type f -size "${expected_size}c" -readable -print0 2>/dev/null)
    done <"${r2e_gold_blobs}"
    (( r2e_blob_leak == 0 )) || exit 1
    rm -f -- "${r2e_gold_blobs}"

    r2e_commit_leak=0
    while IFS= read -r -d '' candidate; do
        case "${candidate}" in
            /proc/*|/sys/*|/dev/*|/run/*) continue ;;
        esac
        if grep -a -F -q -- "${gold_commit}" "${candidate}" 2>/dev/null; then
            echo "R2E agent rootfs retains the gold commit text: ${candidate}" >&2
            r2e_commit_leak=1
        fi
    done < <(find / -xdev -type f -readable -print0 2>/dev/null)
    (( r2e_commit_leak == 0 )) || exit 1
    printf '%s\n' miles-r2e-visible-rootfs-attestation-v1 \
        >/opt/miles-swe/r2e-rootfs-attestation
    chmod 0444 /opt/miles-swe/r2e-rootfs-attestation
fi

# Harbor runs the agent as UID 1000 while collect hooks keep the image default
# (root).  The agent owns the working tree but cannot rewrite .git/HEAD, refs,
# the index baseline, or the trusted workdir marker used during collection.
if ! getent passwd 1000 >/dev/null; then
    useradd --create-home --uid 1000 --shell /bin/bash miles-agent
fi
agent_user="$(getent passwd 1000 | cut -d: -f1)"
agent_group="$(getent passwd 1000 | cut -d: -f4)"
[[ "$(id -u "${agent_user}")" == 1000 && "${agent_group}" != 0 ]] || {
    echo "UID 1000 must be an unprivileged account with a non-root primary group" >&2
    exit 1
}
usermod -G "${agent_group}" "${agent_user}"
if command -v sudo >/dev/null 2>&1 && \
    su -s /bin/sh "${agent_user}" -c 'sudo -n true' >/dev/null 2>&1; then
    echo "UID 1000 retains passwordless sudo privileges" >&2
    exit 1
fi
find "${repo}" -exec chown -h "1000:${agent_group}" {} +
if [[ "${runtime_policy}" == npm-node-modules-v2 ]]; then
    chown -h root:root "${repo}/node_modules"
    while IFS= read -r relative; do
        [[ -z "${relative}" ]] || {
            chown -R root:root "${repo}/${relative}"
            chmod -R a+rX "${repo}/${relative}"
            chmod -R a-w "${repo}/${relative}"
        }
    done </opt/miles-swe/npm-repo-runtime-paths
elif [[ "${runtime_policy}" == python-editable-metadata-v1 ]]; then
    while IFS= read -r relative; do
        [[ -z "${relative}" ]] || chown -h root:root "${repo}/${relative}"
    done </opt/miles-swe/runtime-links
elif [[ "${schema}" == r2e-gym-v1 ]]; then
    chown -h root:root "${repo}/.venv"
    while IFS= read -r relative; do
        [[ -z "${relative}" ]] || chown -h root:root "${repo}/${relative}"
    done </opt/miles-swe/r2e-runtime-links
fi
chown -R root:root /opt/miles-swe
chmod -R a+rX /opt/miles-swe/agent-git
if [[ -d "${runtime_root}" ]]; then
    chmod -R a+rX "${runtime_root}"
fi
chmod -R go-w /opt/miles-swe
chmod 0700 /opt/miles-swe/root-home /opt/miles-swe/root-home/xdg
if [[ "${schema}" == r2e-gym-v1 ]]; then
    # Leave the committed image with an empty model-visible scratch directory,
    # including after account and ownership tooling has run.
    find /tmp -xdev -mindepth 1 -delete
    chmod 1777 /tmp
fi
