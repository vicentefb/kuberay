"""
Real-model RL-shaped suspend/resume harness (tier-1 realism).

Each Ray actor runs one self-paced multi-turn rollout: it fires suspend()
and a real Gemini inference CONCURRENTLY (snapshot uploads while the model
thinks, as a production orchestrator would), resumes when both complete,
executes the model-generated Python inside the gVisor sandbox, verifies the
transcript state survived the cycle, and repeats. No synchronized waves —
arrival stagger comes from genuine inference variance.

Key metric this adds over the synthetic harness:
  added_latency_per_turn = max(suspend, inference) + resume - inference
i.e. the true per-turn cost of suspension versus not suspending.

Env knobs:
  NUM_EXECUTORS (default 100)   TURNS (default 5)      STATE_MB (default 64)
  GEMINI_MODEL (default gemini-2.5-flash; use a pro-class model for
  Reflection-realistic 7-15s inference)   VERTEX_LOCATION (default us-central1)
  VERTEX_PROJECT (default gke-ai-eco-dev)
"""

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import ray

NUM_EXECUTORS = int(os.environ.get("NUM_EXECUTORS", "100"))
TURNS = int(os.environ.get("TURNS", "5"))
STATE_MB = int(os.environ.get("STATE_MB", "64"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "gke-ai-eco-dev")
WARMPOOL = os.environ.get("WARMPOOL", "python-snapshot-pool")
SNAPSHOT_READY_RETRIES = int(os.environ.get("SNAPSHOT_READY_RETRIES", "300"))
RESUME_WAIT = int(os.environ.get("RESUME_WAIT", "600"))

SETUP = (
    "import os\n"
    f"mb = {STATE_MB}\n"
    "with open('/tmp/ballast.bin', 'wb') as f:\n"
    "    for _ in range(mb):\n"
    "        f.write(os.urandom(1024 * 1024))\n"
    "open('/tmp/transcript.txt', 'w').write('turn 0: rollout started\\n')\n"
    "print('ready')\n"
)

PROMPT = (
    "You are an agent generating one small self-contained Python snippet per turn. "
    "Rules: output ONLY Python code, no markdown fences, no explanations. "
    "The snippet must: (1) do a small computation of your choice (math, string "
    "manipulation, a tiny algorithm — vary it), (2) append exactly one line "
    "describing the result to /tmp/transcript.txt, (3) finish in under 5 seconds. "
    "Recent transcript:\n{tail}\n"
)


@ray.remote(num_cpus=0, memory=200 * 1024 * 1024)
class RolloutActor:
    def __init__(self, worker_id: int):
        from google import genai
        from k8s_agent_sandbox.gke_extensions.snapshots import PodSnapshotSandboxClient
        from k8s_agent_sandbox.models import SandboxInClusterConnectionConfig

        self.worker_id = worker_id
        self.llm = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        self.client = PodSnapshotSandboxClient(
            connection_config=SandboxInClusterConnectionConfig(use_pod_ip=True, server_port=8888),
            cleanup=True,
        )
        self.sandbox = self.client.create_sandbox(warmpool=WARMPOOL)
        self.pool = ThreadPoolExecutor(max_workers=2)

    def _suspend(self) -> float:
        t0 = time.time()
        res = self.sandbox.suspend(snapshot_before_suspend=True)
        if not res.success:
            raise RuntimeError(f"suspend: {res.error_reason}")
        if res.snapshot_response is None:
            return time.time() - t0
        uid = res.snapshot_response.snapshot_uid
        for _ in range(SNAPSHOT_READY_RETRIES):
            listed = self.sandbox.snapshots.list()
            if listed.success and any(s.snapshot_uid == uid for s in listed.snapshots):
                return time.time() - t0
            time.sleep(2)
        raise RuntimeError("snapshot never Ready")

    def _retry(self, fn, tries=4, delay=3):
        last = None
        for _ in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(delay)
        raise last

    def _infer(self, tail: str) -> tuple[str, float]:
        t0 = time.time()
        resp = self.llm.models.generate_content(
            model=GEMINI_MODEL, contents=PROMPT.format(tail=tail))
        code = (resp.text or "").strip()
        if code.startswith("```"):
            code = code.strip("`\n")
            if code.startswith("python"):
                code = code[len("python"):]
        return code.strip(), time.time() - t0

    def run_rollout(self) -> dict:
        turns, errors = [], []
        try:
            self.sandbox.files.write("setup.py", SETUP)
            r = self.sandbox.commands.run("python setup.py", timeout=300)
            if r.exit_code != 0:
                return {"ok": False, "error": f"setup: {r.stderr}", "turns": [], "errors": []}
            tail = "turn 0: rollout started"

            for turn in range(1, TURNS + 1):
                try:
                    # Suspend and inference run CONCURRENTLY — the snapshot
                    # uploads while the model thinks.
                    f_sus = self.pool.submit(self._suspend)
                    f_inf = self.pool.submit(self._infer, tail)
                    code, infer_s = f_inf.result()
                    suspend_s = f_sus.result()

                    t0 = time.time()
                    res = self.sandbox.resume(wait_timeout=RESUME_WAIT)
                    resume_s = time.time() - t0
                    if not res.success:
                        if "not restored from snapshot" in (res.error_reason or ""):
                            time.sleep(3)
                            probe = self._retry(lambda: self.sandbox.commands.run(
                                "python -c \"print(sum(1 for _ in open('/tmp/transcript.txt')))\"",
                                timeout=15))
                            if probe.exit_code != 0:
                                raise RuntimeError("cold start CONFIRMED: transcript gone")
                            # false alarm: state intact — continue the turn
                        else:
                            raise RuntimeError(f"resume: {res.error_reason}")

                    self._retry(lambda: self.sandbox.files.write(f"turn_{turn}.py", code))
                    t0 = time.time()
                    r = self._retry(lambda: self.sandbox.commands.run(f"python turn_{turn}.py", timeout=20))
                    exec_s = time.time() - t0

                    chk = self._retry(lambda: self.sandbox.commands.run(
                        "python -c \"print(sum(1 for _ in open('/tmp/transcript.txt')))\"", timeout=15))
                    lines = int(chk.stdout.strip()) if chk.exit_code == 0 else -1
                    state_ok = lines >= turn  # model snippet may or may not have appended; >= turn-1 lines must survive
                    tail = self._retry(lambda: self.sandbox.commands.run(
                        "tail -3 /tmp/transcript.txt", timeout=15)).stdout

                    added = max(suspend_s, infer_s) + resume_s - infer_s
                    turns.append({"infer": infer_s, "suspend": suspend_s, "resume": resume_s,
                                  "exec": exec_s, "added": added, "model_code_ok": r.exit_code == 0,
                                  "state_ok": state_ok})
                except Exception as e:
                    errors.append(f"turn {turn}: {e}")
            return {"ok": len(errors) == 0, "turns": turns, "errors": errors}
        except Exception as e:
            return {"ok": False, "error": str(e), "turns": turns, "errors": errors}

    def cleanup(self):
        self.sandbox.terminate()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * len(xs))) - 1)] if xs else 0.0


def main() -> int:
    ray.init()
    print(f"config: NUM_EXECUTORS={NUM_EXECUTORS} TURNS={TURNS} STATE_MB={STATE_MB} "
          f"MODEL={GEMINI_MODEL} LOCATION={VERTEX_LOCATION}")

    actors = [RolloutActor.remote(i) for i in range(NUM_EXECUTORS)]
    results = []
    try:
        futures = [a.run_rollout.remote() for a in actors]
        for f in futures:
            try:
                results.append(ray.get(f))
            except Exception as e:
                results.append({"ok": False, "error": f"actor died: {e}", "turns": [], "errors": []})

        allt = [t for r in results for t in r["turns"]]
        errs = [e for r in results for e in r.get("errors", [])] + \
               [r["error"] for r in results if r.get("error")]
        print("\n=== RL-SHAPED RESULTS ===")
        print(f"rollouts: {sum(1 for r in results if r['ok'])}/{NUM_EXECUTORS} clean; "
              f"turns completed: {len(allt)}/{NUM_EXECUTORS * TURNS}")
        for k, label in [("infer", "inference"), ("suspend", "suspend"), ("resume", "resume"),
                         ("exec", "execute"), ("added", "ADDED LATENCY vs no-suspend")]:
            xs = [t[k] for t in allt]
            if xs:
                print(f"{label:28s} p50={statistics.median(xs):6.1f}s  "
                      f"p95={pct(xs, 0.95):6.1f}s  max={max(xs):6.1f}s")
        print(f"model snippets ran clean: {sum(1 for t in allt if t['model_code_ok'])}/{len(allt)}")
        print(f"state survived every verified cycle: "
              f"{sum(1 for t in allt if t['state_ok'])}/{len(allt)}")
        print(f"errors: {len(errs)}")
        for e in errs[:40]:
            print(f"  ERR {e}")
        return 1 if errs else 0
    finally:
        print("\nCleaning up...")
        for a in actors:
            try:
                ray.get(a.cleanup.remote())
            except Exception:
                pass
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
