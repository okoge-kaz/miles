#!/bin/bash
#PBS -N difficulty-merge
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

[[ "${2:-}" == merge ]] || { echo 'merge.sh requires merge mode' >&2; exit 2; }
# Both files are snapshotted together; use the adjacent copy in a submitted job.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run.sh" "$@"
