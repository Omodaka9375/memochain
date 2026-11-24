"""Async SP1 proof queue for the MemoChain proxy.

The proxy folds evicted turns with the exact Q14 integer writer (instant,
CPU) and hands the client its capsule immediately. In the background, this
queue feeds the same raw Q14 docs to the SP1 IVC CLI
(``crates/sp1/script/src/bin/ivc.rs``) so the session's memory state also
carries a zkVM proof:

- mode ``execute``: guest execution only (fast, no cryptographic proof) —
  full replay of the session's cumulative docs from genesis each time.
  Produces an independent check that the capsule digest chain is correct.
- mode ``prove``: compressed recursive STARK proof. Genesis proof for the
  first batch, then one recursive step per batch via ``--prev-proof``.
  Verify in ~30 ms anywhere with ``ivc verify``.

All job state lives under ``work_root/<session_id>/``:
  docs_all.bin      cumulative raw Q14 docs (little-endian i16, 2560/doc)
  proof.bin         latest compressed proof (prove mode)
  proof_meta.json   latest proof/execution result + capsule match status
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

Runner = Callable[[list[str]], "subprocess.CompletedProcess[bytes]"]


def default_runner(cmd: list[str]) -> "subprocess.CompletedProcess[bytes]":
    # The ivc CLI resolves its guest ELF via ../ivc-program/... relative paths,
    # so run it from crates/sp1/script (one level above its release binary).
    cwd = None
    exe = Path(cmd[0])
    if exe.name == "ivc":
        script_dir = (exe.parent.parent.parent).resolve()  # .../script/target/release -> .../script
        if (script_dir.parent / "ivc-program").exists():
            cwd = script_dir
    return subprocess.run(cmd, capture_output=True, timeout=None, cwd=cwd)


class ProofQueue:
    """Single background worker thread; jobs are serialized (KISS, one GPU host)."""

    def __init__(
        self,
        mode: str,
        ivc_bin: str | Path,
        weights_path: str | Path,
        work_root: str | Path,
        session_lookup: Callable[[str], Any],
        expected_weights_digest: str | None = None,
        runner: Runner = default_runner,
    ) -> None:
        if mode not in ("execute", "prove"):
            raise ValueError(f"unknown proof mode {mode!r}")
        self.mode = mode
        self.ivc_bin = Path(ivc_bin).resolve()
        self.weights_path = Path(weights_path).resolve()
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.session_lookup = session_lookup
        self.expected_weights_digest = expected_weights_digest
        self._runner = runner
        self._jobs: queue.Queue[str | None] = queue.Queue()
        self._queued: set[str] = set()
        self._dirty: set[str] = set()
        self._proved_bytes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="memochain-proof")
        self._thread.start()

    # ---- public API -------------------------------------------------------

    def submit(self, session_id: str, docs_raw: bytes) -> bool:
        """Append newly folded docs and ensure a proof job runs for them.

        Docs are persisted to ``docs_all.bin`` immediately under the lock so no
        batch is ever dropped if a prior job is still running; when a job is
        already in flight, the session is marked dirty so the worker schedules
        one follow-up pass as soon as the current pass completes.
        Returns True if a new job was enqueued, False if merged into an active job.
        """
        if not docs_raw:
            return False
        d = self._session_dir(session_id)
        with self._lock:
            self._append(d / "docs_all.bin", docs_raw)
            if session_id in self._queued:
                self._dirty.add(session_id)
                return False
            self._queued.add(session_id)
        self._jobs.put(session_id)
        return True

    def status(self, session_id: str) -> dict[str, Any]:
        d = self.work_root / session_id
        meta_path = d / "proof_meta.json"
        with self._lock:
            queued = session_id in self._queued or session_id in self._dirty
        if not meta_path.exists():
            return {"enabled": True, "mode": self.mode, "queued": queued, "done": False}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["queued"] = queued
        meta["done"] = True
        return meta

    # ---- worker ------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            sid = self._jobs.get()
            if sid is None:
                break
            try:
                self._run_job(sid)
            except Exception as exc:  # noqa: BLE001 — report, never crash the worker
                self._write_meta(sid, {"error": f"{type(exc).__name__}: {exc}"})
            finally:
                reschedule = False
                with self._lock:
                    if sid in self._dirty:
                        self._dirty.discard(sid)
                        reschedule = True
                    else:
                        self._queued.discard(sid)
                if reschedule:
                    self._jobs.put(sid)

    def _run_job(self, sid: str) -> None:
        if self.mode == "prove":
            self._run_prove(sid)
        else:
            self._run_execute(sid)

    def _run_execute(self, sid: str) -> None:
        d = self._session_dir(sid)
        docs_all = d / "docs_all.bin"
        # Snapshot bytes under the lock so concurrent submit() appends never
        # tear a doc mid-read.
        with self._lock:
            raw = docs_all.read_bytes()
        tmp = d / "tmp"
        tmp.mkdir(exist_ok=True)
        snap = tmp / "docs_exec.bin"
        snap.write_bytes(raw)
        cmd = [
            str(self.ivc_bin), "execute",
            "--weights", str(self.weights_path),
            "--docs", str(snap),
        ]
        t0 = time.time()
        proc = self._runner(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"ivc execute failed: {proc.stderr.decode()[:500]}")
        out = proc.stdout.decode()
        secs = time.time() - t0
        meta = {
            "mode": "execute",
            "secs": round(secs, 2),
            "cycles": int(m.group(1)) if (m := re.search(r"total cycles: (\d+)", out)) else None,
            "step": int(m.group(1)) if (m := re.search(r"^step:\s+(\d+)", out, re.M)) else None,
            "weights_digest": (m.group(1) if (m := re.search(r"^weights_digest:\s+(\S+)", out, re.M)) else None),
            "state_digest": (m.group(1) if (m := re.search(r"^state_digest:\s+(\S+)", out, re.M)) else None),
            "accumulator": (m.group(1) if (m := re.search(r"^accumulator:\s+(\S+)", out, re.M)) else None),
        }
        self._check_and_write(sid, meta)

    def _run_prove(self, sid: str) -> None:
        d = self._session_dir(sid)
        docs_all = d / "docs_all.bin"
        with self._lock:
            all_bytes = docs_all.read_bytes()
            offset = self._proved_bytes.get(sid, 0)
            if offset == 0 and (d / "ivc_meta.json").exists():
                prev_meta = json.loads((d / "ivc_meta.json").read_text(encoding="utf-8"))
                offset = int(prev_meta.get("step", 0)) * 5120
            delta = all_bytes[offset:]
        if not delta:
            return
        tmp = d / "tmp"
        tmp.mkdir(exist_ok=True)
        batch = tmp / "docs_batch.bin"
        batch.write_bytes(delta)
        proof_bin = d / "proof.bin"
        cmd = [
            str(self.ivc_bin), "prove",
            "--weights", str(self.weights_path),
            "--docs", str(batch),
            "--out-proof", str(tmp / "proof.bin"),
            "--out-capsule", str(tmp / "capsule.bin"),
            "--out-meta", str(tmp / "meta.json"),
        ]
        if proof_bin.exists():  # recursive +batch step on top of the previous proof
            cmd += ["--prev-proof", str(proof_bin)]
        proc = self._runner(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"ivc prove failed: {proc.stderr.decode()[:500]}")
        ivc_meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
        os.replace(tmp / "proof.bin", proof_bin)
        os.replace(tmp / "meta.json", d / "ivc_meta.json")
        with self._lock:
            self._proved_bytes[sid] = offset + len(delta)
        meta = {"mode": "prove", **ivc_meta}
        self._check_and_write(sid, meta)

    # ---- helpers -----------------------------------------------------------

    def _session_dir(self, sid: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in sid)[:80]
        d = self.work_root / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _append(path: Path, data: bytes) -> None:
        with open(path, "ab") as fh:
            fh.write(data)

    def _write_meta(self, sid: str, meta: dict[str, Any]) -> None:
        d = self._session_dir(sid)
        (d / "proof_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _check_and_write(self, sid: str, meta: dict[str, Any]) -> None:
        if meta.get("weights_digest") and self.expected_weights_digest:
            if meta["weights_digest"] != self.expected_weights_digest:
                meta["error"] = (
                    f"proof weights_digest {meta['weights_digest']} != proxy writer "
                    f"{self.expected_weights_digest} (different student checkpoint?)"
                )
        sess = self.session_lookup(sid)
        head = getattr(sess, "head", None)
        if head is not None and meta.get("state_digest"):
            meta["matches_capsule"] = bool(
                meta.get("step") == head.step and meta.get("state_digest") == head.state_digest
            )
            if meta["mode"] == "execute":
                meta["matches_accumulator"] = bool(meta.get("accumulator") == head.accumulator)
        self._write_meta(sid, meta)

    def stop(self) -> None:
        self._jobs.put(None)
