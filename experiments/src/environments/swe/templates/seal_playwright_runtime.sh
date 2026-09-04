#!/bin/bash

# Relocate one publisher-provided Playwright browser cache into a bounded,
# root-owned runtime. The immutable source-image digest remains the trust
# boundary; semantic admission still runs the task's exact public baseline.

set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset BASH_ENV CDPATH ENV

target=/opt/miles-swe/runtime/ms-playwright
marker=/opt/miles-swe/playwright-browsers-path
inventory=/opt/miles-swe/playwright-runtime.inventory
inventory_digest=/opt/miles-swe/playwright-runtime.inventory.sha256
maximum_bytes=8589934592
maximum_entries=200000

[[ "$(id -u)" == 0 ]] || {
    echo "Playwright runtime sealing must run as root" >&2
    exit 1
}
[[ ! -e "${target}" && ! -L "${target}" && ! -e "${marker}" && \
    ! -L "${marker}" && ! -e "${inventory}" && \
    ! -e "${inventory_digest}" ]] || {
    echo "Playwright runtime target already exists" >&2
    exit 1
}

candidates=()
for candidate in /root/.cache/ms-playwright /ms-playwright; do
    [[ ! -e "${candidate}" && ! -L "${candidate}" ]] || candidates+=("${candidate}")
done
(( ${#candidates[@]} <= 1 )) || {
    echo "source image contains ambiguous Playwright browser caches" >&2
    exit 1
}

install -d -o root -g root -m 0755 /opt/miles-swe/runtime
if (( ${#candidates[@]} == 0 )); then
    printf 'none\n' >"${marker}"
    chmod 0444 "${marker}"
    exit 0
fi

source_path="${candidates[0]}"
[[ -d "${source_path}" && ! -L "${source_path}" && \
    "$(stat -c %u "${source_path}")" == 0 ]] || {
    echo "published Playwright cache root is unsafe" >&2
    exit 1
}
runtime_bytes="$(du -sb -- "${source_path}" | awk '{print $1}')"
runtime_entries="$(find "${source_path}" -xdev -printf . | wc -c)"
[[ "${runtime_bytes}" =~ ^[0-9]+$ && "${runtime_entries}" =~ ^[0-9]+$ ]] || exit 1
(( runtime_bytes > 0 && runtime_bytes <= maximum_bytes )) || {
    echo "published Playwright runtime exceeds the 8 GiB bound" >&2
    exit 1
}
(( runtime_entries > 0 && runtime_entries <= maximum_entries )) || {
    echo "published Playwright runtime exceeds the entry-count bound" >&2
    exit 1
}
if find "${source_path}" -xdev \( \
    -type b -o -type c -o -type l -o -type p -o -type s \
\) -print -quit | grep -q .; then
    echo "published Playwright runtime contains a symlink or special file" >&2
    exit 1
fi
while read -r inode link_count; do
    inside_count="$(find "${source_path}" -xdev -type f -inum "${inode}" -printf . | wc -c)"
    [[ "${inside_count}" == "${link_count}" ]] || {
        echo "Playwright runtime hard link escapes its sealed tree" >&2
        exit 1
    }
done < <(find "${source_path}" -xdev -type f -links +1 -printf '%i %n\n' | sort -u)
mv -- "${source_path}" "${target}"
chown -R root:root "${target}"
chmod -R a+rX "${target}"
chmod -R a-w "${target}"
find "${target}" -xdev -type f -perm /6000 -exec chmod a-s {} +
[[ -z "$(find "${target}" -xdev -perm /222 -print -quit)" ]] || {
    echo "sealed Playwright runtime remains writable" >&2
    exit 1
}
: >"${inventory}"
while IFS= read -r -d '' runtime_file; do
    relative="${runtime_file#"${target}/"}"
    [[ "${relative}" != "${runtime_file}" ]] || exit 1
    size="$(stat -c %s "${runtime_file}")"
    digest="$(sha256sum "${runtime_file}" | awk '{print $1}')"
    printf '%s\0%s\0%s\0' "${relative}" "${size}" "${digest}" \
        >>"${inventory}"
done < <(find "${target}" -xdev -type f -print0 | sort -z)
printf '%s\n' "$(sha256sum "${inventory}" | awk '{print $1}')" \
    >"${inventory_digest}"
printf '%s\n' "${target}" >"${marker}"
chown root:root "${marker}" "${inventory}" "${inventory_digest}"
chmod 0444 "${marker}" "${inventory}" "${inventory_digest}"
