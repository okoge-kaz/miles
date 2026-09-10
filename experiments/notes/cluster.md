# OCI cluster: oci-hsg-cs-001

Verified on 2026-09-09; resource limits and TCP ranges rechecked on 2026-09-10.
These are the current defaults; cw-dfw measurements in
`agents/cluster.md` are historical and do not describe this cluster.

## Account and resource discovery

The experiment default account is **nemotron_sw_post**. Override it with
`SLURM_ACCOUNT_NAME` for submission wrappers, or `sbatch -A <account>`.
An environment variable cannot expand inside a `#SBATCH` directive.

Run these commands on the login host to distinguish partitions from QoS:

```bash
sinfo -o '%P %a %l %D %G'
scontrol show partition
sacctmgr show qos format=Name,Priority,Flags,MaxWall,MaxTRES,MaxTRESPU -P
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS
```

**interactive is a QoS, not a partition.** The association for kfujii includes
nemotron_sw_post and the interactive/normal/short and cpu-* QoS values.

| Work | Partition | QoS | Observed limit |
|---|---|---|---|
| GPU validation | batch | interactive | 4 h partition limit; 4 nodes/job and user |
| GPU tuning | batch | short | 2 h; 64 nodes/job |
| GPU training | batch | normal | 4 h |
| Longer GPU training | batch_long | normal | 7 d partition limit; recheck account admission |
| CPU validation/import | cpu | cpu-interactive | 1 d; 2 nodes/job |
| CPU normal | cpu | cpu-normal | 1 d; 2 nodes/job |
| Long-lived CPU controller | cpu | cpu-long | 7 d; 1 node/job |

Verify scheduler acceptance without submitting work:

```bash
sbatch --test-only -A nemotron_sw_post -p batch --qos=interactive \
  -N 2 --ntasks-per-node=1 --gpus-per-node=4 --time=00:20:00 --wrap=true
```

For disposable validation, reserve two nodes, then run tests inside the allocation:

```bash
salloc --no-shell -A nemotron_sw_post -p batch --qos=interactive \
  -N 2 --ntasks-per-node=1 --gpus-per-node=4 --time=01:00:00
# Use the job ID printed by salloc:
srun --jobid=<job-id> -N 2 --ntasks-per-node=1 hostname
# Release only your own completed validation allocation:
scancel <job-id>
```

One-node CUDA success does not qualify cross-node Ray or NCCL paths. Keep
training/checkpoint/eval qualification separate from unit and communication tests.

The CPU nodes are also aarch64, but have no GPU driver. Importing Megatron from
the qualified training image can import Transformer Engine and require
`libcuda.so.1`, even when the intended computation is CPU-only (job 7060110).
Keep those checks inside a GPU interactive allocation, or restrict a CPU job to
GPU-driver-independent libraries such as PyTorch CPU tensor operations and
safetensors. A CPU-only test subprocess on a GPU node is not equivalent to a
`cpu / cpu-interactive` allocation.

## Managed worker ports

Inside the interactive allocation, inspect the actual node's TCP ranges:

```bash
srun --jobid=<job-id> -N 2 --ntasks-per-node=1 \
  sysctl net.ipv4.ip_local_port_range net.ipv4.ip_local_reserved_ports
```

OCI reported `ip_local_port_range = 9000 65000`, not the commonly assumed
`32768 60999`. Job 7058972 hit an HTTP bind failure at port 20001: another
SGLang scheduler had already acquired that port as an internal listener.
The managed-port probe and the later HTTP bind are separate operations, so
probing an ephemeral-range port does not reserve it against such a race.

`experiments/env.sh` now sets `MILES_WORKER_PORT_START=2000` and
`MILES_WORKER_PORT_END=8999`. The allocator rejects blocks outside this range.
This keeps managed engine/router/RPC listeners below the observed ephemeral
range and away from Ray's 10002–19999 worker ports, without changing system
sysctls. The core allocator's defaults remain unchanged for other clusters.
Recheck these limits on a different cluster rather than copying them blindly.

## Hardware and filesystems

The GPU probe on nvl72005-T11 reported aarch64, **4 NVIDIA GB200 GPUs**,
189471 MiB/GPU, and driver 580.126.20. Slurm reports 144 CPUs and 942080 MB
node memory. Do not reuse H100 x8 node ratios, tensor-parallel shapes, or
throughput measurements as GB200 results.

The user workspace defaults to `/lustre/fsw/portfolios/coreai/users/$USER`.
`WS`, `SHARED_WS`, asset directories and `SQSH_IMAGE` are overridable in
`experiments/env.sh` or `.env`. A similarly named path on cw-dfw does not
imply assets exist here. In particular, the Step4000 checkpoint and its
policy-filtered DAPO dataset must be staged explicitly before training.

On the probed GPU and CPU nodes, `/tmp` is node-local ext4, not tmpfs.
GPU root storage was about 6.5 TiB; CPU root storage about 503 GiB.
Check `df -hT /tmp` and `findmnt -T /tmp` on the actual allocated node before
large imports. Enroot extraction/overlay scratch belongs on node-local storage;
downloaded layer cache and final SQSH can live on Lustre. Use a unique
`mktemp -d` directory per import. Do not assume the cw-dfw `/raid` path exists.

Submit recipes from the repository root, and create the requested log parent
directory before sbatch. Editing a script after submission does not change the
copy already spooled by Slurm; source files read at runtime are different.
