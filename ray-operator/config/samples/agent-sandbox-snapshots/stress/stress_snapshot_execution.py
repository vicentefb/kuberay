"""
Stress harness for Agent Sandbox suspend/resume with GKE Pod Snapshots.

NOT part of the sample / PR. Measures suspend/resume latency distribution and
failure rate across many concurrent cycles, with a configurable per-sandbox
working set so snapshot time can be plotted against snapshot size.

Tunables (env vars, set via the RayJob runtimeEnvYAML):
  NUM_EXECUTORS  concurrent sandboxes            (default 20)
  CYCLES         suspend/resume cycles each      (default 5)
  STATE_MB       random tmpfs state per sandbox  (default 64)
  MODEL_LATENCY  seconds suspended per cycle     (default 5)
  WARMPOOL       warm pool to claim from         (default python-snapshot-pool)
"""

import os
import statistics
import sys
import time
import ray

NUM_EXECUTORS = int(os.environ.get("NUM_EXECUTORS", "20"))
CYCLES = int(os.environ.get("CYCLES", "5"))
STATE_MB = int(os.environ.get("STATE_MB", "64"))
MODEL_LATENCY = float(os.environ.get("MODEL_LATENCY", "5"))
WARMPOOL = os.environ.get("WARMPOOL", "python-snapshot-pool")
# Deep-queue budgets for high-concurrency rungs.
SNAPSHOT_READY_RETRIES = int(os.environ.get("SNAPSHOT_READY_RETRIES", "150"))
RESUME_WAIT = int(os.environ.get("RESUME_WAIT", "300"))

# Writes STATE_MB of random data to tmpfs (counts as guest memory, defeats
# compression) and records a sampled digest for post-restore verification.
WRITE_STATE = (
    "import hashlib, os\n"
    f"mb = {STATE_MB}\n"
    "with open('/tmp/state.bin', 'wb') as f:\n"
    "    for _ in range(mb):\n"
    "        f.write(os.urandom(1024 * 1024))\n"
    "h = hashlib.md5()\n"
    "with open('/tmp/state.bin', 'rb') as f:\n"
    "    h.update(f.read(4 * 1024 * 1024))\n"
    "print(h.hexdigest(), os.path.getsize('/tmp/state.bin'))\n"
)

VERIFY_STATE = (
    "import hashlib, os\n"
    "h = hashlib.md5()\n"
    "with open('/tmp/state.bin', 'rb') as f:\n"
    "    h.update(f.read(4 * 1024 * 1024))\n"
    "print(h.hexdigest(), os.path.getsize('/tmp/state.bin'))\n"
)


@ray.remote(num_cpus=0)
class StressExecutor:
    def __init__(self, worker_id: int):
        from k8s_agent_sandbox.gke_extensions.snapshots import PodSnapshotSandboxClient
        from k8s_agent_sandbox.models import SandboxInClusterConnectionConfig

        self.worker_id = worker_id
        self.client = PodSnapshotSandboxClient(
            connection_config=SandboxInClusterConnectionConfig(use_pod_ip=True, server_port=8888),
            cleanup=True,
        )
        t0 = time.time()
        self.sandbox = self.client.create_sandbox(warmpool=WARMPOOL)
        self.claim_seconds = time.time() - t0
        self.state_digest = None

    def setup_state(self) -> dict:
        try:
            self.sandbox.files.write("write_state.py", WRITE_STATE)
            self.sandbox.files.write("verify_state.py", VERIFY_STATE)
            res = self.sandbox.commands.run("python write_state.py", timeout=300)
            if res.exit_code != 0:
                return {"ok": False, "error": f"write_state: {res.stderr}"}
            self.state_digest = res.stdout.strip()
            return {"ok": True, "claim_seconds": self.claim_seconds}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def suspend(self) -> dict:
        t0 = time.time()
        try:
            res = self.sandbox.suspend(snapshot_before_suspend=True)
            if not res.success:
                return {"ok": False, "seconds": time.time() - t0, "error": res.error_reason}
            uid = res.snapshot_response.snapshot_uid
            # Poll until the snapshot is Ready (upload complete); generous
            # budget because large working sets upload slowly.
            for _ in range(SNAPSHOT_READY_RETRIES):
                listed = self.sandbox.snapshots.list()
                if listed.success and any(s.snapshot_uid == uid for s in listed.snapshots):
                    return {"ok": True, "seconds": time.time() - t0, "uid": uid}
                time.sleep(2)
            return {"ok": False, "seconds": time.time() - t0, "error": "snapshot never became Ready"}
        except Exception as e:
            return {"ok": False, "seconds": time.time() - t0, "error": str(e)}

    def resume_and_verify(self) -> dict:
        t0 = time.time()
        try:
            res = self.sandbox.resume(wait_timeout=RESUME_WAIT)
            seconds = time.time() - t0
            if not res.success:
                # The SDK reports a fresh-instance via success=False. Its
                # PodRestored check runs once, right after Ready, and can
                # race the condition write — adjudicate by checking whether
                # the state actually survived before calling it a loss.
                if "not restored from snapshot" in (res.error_reason or ""):
                    time.sleep(5)
                    check = self.sandbox.commands.run("python verify_state.py", timeout=120)
                    if check.exit_code == 0 and check.stdout.strip() == self.state_digest:
                        return {"ok": True, "seconds": seconds,
                                "note": "SDK false alarm: state intact, PodRestored condition lagged"}
                    return {"ok": False, "seconds": seconds, "error": "cold-started, state LOST"}
                return {"ok": False, "seconds": seconds, "error": res.error_reason}
            check = self.sandbox.commands.run("python verify_state.py", timeout=120)
            if check.exit_code != 0:
                return {"ok": False, "seconds": seconds, "error": f"verify exec: {check.stderr}"}
            if check.stdout.strip() != self.state_digest:
                return {"ok": False, "seconds": seconds,
                        "error": f"state mismatch: {check.stdout.strip()!r} != {self.state_digest!r}"}
            return {"ok": True, "seconds": seconds}
        except Exception as e:
            return {"ok": False, "seconds": time.time() - t0, "error": str(e)}

    def snapshot_count(self) -> int:
        listed = self.sandbox.snapshots.list()
        return len(listed.snapshots) if listed.success else -1

    def cleanup(self):
        self.sandbox.terminate()


def stats(samples):
    if not samples:
        return "n=0"
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(round(0.95 * len(s))) - 1)]
    return (f"n={len(s)} median={statistics.median(s):.1f}s "
            f"p95={p95:.1f}s max={s[-1]:.1f}s")


def gather(futures):
    """ray.get one by one so a single failure doesn't mask the others."""
    out = []
    for f in futures:
        try:
            out.append(ray.get(f))
        except Exception as e:
            out.append({"ok": False, "seconds": -1, "error": f"actor died: {e}"})
    return out


def main() -> int:
    ray.init()
    print(f"config: NUM_EXECUTORS={NUM_EXECUTORS} CYCLES={CYCLES} "
          f"STATE_MB={STATE_MB} MODEL_LATENCY={MODEL_LATENCY}")

    executors = [StressExecutor.remote(worker_id=i) for i in range(NUM_EXECUTORS)]
    suspend_times, resume_times, failures = [], [], []

    try:
        setups = gather([e.setup_state.remote() for e in executors])
        claim_times = [r["claim_seconds"] for r in setups if r.get("ok")]
        for i, r in enumerate(setups):
            if not r.get("ok"):
                failures.append(f"[setup][executor-{i}] {r.get('error')}")
        print(f"claims: {stats(claim_times)}  ({len(claim_times)}/{NUM_EXECUTORS} sandboxes ready, "
              f"{STATE_MB}MB state each)")

        for cycle in range(CYCLES):
            t_cycle = time.time()
            sus = gather([e.suspend.remote() for e in executors])
            for i, r in enumerate(sus):
                (suspend_times if r["ok"] else failures).append(
                    r["seconds"] if r["ok"] else f"[cycle {cycle}][suspend][executor-{i}] {r.get('error')}")

            time.sleep(MODEL_LATENCY)

            res = gather([e.resume_and_verify.remote() for e in executors])
            for i, r in enumerate(res):
                (resume_times if r["ok"] else failures).append(
                    r["seconds"] if r["ok"] else f"[cycle {cycle}][resume][executor-{i}] {r.get('error')}")

            ok = sum(1 for r in res if r["ok"])
            for i, r in enumerate(res):
                if r.get("note"):
                    print(f"  NOTE [cycle {cycle}][executor-{i}] {r['note']}")
            print(f"[cycle {cycle}] {ok}/{NUM_EXECUTORS} restored+verified "
                  f"(wall {time.time() - t_cycle:.0f}s)")

        counts = gather([e.snapshot_count.remote() for e in executors])
        print("\n=== RESULTS ===")
        print(f"suspend (snapshot Ready + pod gone): {stats(suspend_times)}")
        print(f"resume  (pod up + restore verified): {stats(resume_times)}")
        total_ops = NUM_EXECUTORS * CYCLES * 2
        print(f"failures: {len(failures)}/{total_ops} ops")
        for f in failures:
            print(f"  FAIL {f}")
        over = [c for c in counts if isinstance(c, int) and c > 3]
        print(f"retention (maxSnapshotCountPerGroup=3): per-sandbox snapshot counts {sorted(set(counts))} "
              f"-> {'VIOLATION: ' + str(over) if over else 'OK'}")
        return 1 if failures else 0

    finally:
        print("\nCleaning up...")
        gather([e.cleanup.remote() for e in executors])
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
