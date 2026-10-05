"""Cap torch's CPU threads at the container's CPU quota.

torch sizes its intra-op pool by the HOST's cores, not the cgroup limit: on a
cluster CPU pod (256-core node, 8-core limit) every matmul ran 256 threads on
8 cores, and the suite took ~20 min there (test_router_train alone ~11 min)
vs ~25 s on a laptop. No quota (laptop, unlimited container) -> unchanged.
"""

import math
from pathlib import Path


def _cgroup_cpu_quota():
    v2 = Path("/sys/fs/cgroup/cpu.max")  # "max 100000" or "<quota> <period>"
    if v2.exists():
        quota, period = v2.read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    v1_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if v1_quota.exists():
        quota = int(v1_quota.read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    return None


def pytest_configure(config):
    quota = _cgroup_cpu_quota()
    if quota is None:
        return
    import torch

    torch.set_num_threads(max(1, math.floor(quota)))
