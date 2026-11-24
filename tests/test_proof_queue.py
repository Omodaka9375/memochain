"""Tests for the async SP1 proof queue."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from memochain.server.proof import ProofQueue


class FakeSession:
    def __init__(self, step: int, state_digest: str) -> None:
        from memochain.verify import ChainHead

        self.head = ChainHead(step=step, accumulator=f"acc{step:061x}", state_digest=state_digest)


def fake_ivc_runner(calls: list[list[str]]):
    def run(cmd: list[str]) -> object:
        calls.append(cmd)
        mode = cmd[1]  # execute | prove
        weights = cmd[cmd.index("--weights") + 1]
        docs = cmd[cmd.index("--docs") + 1]
        n_docs = Path(docs).stat().st_size // 5120
        if mode == "prove":
            out_proof = Path(cmd[cmd.index("--out-proof") + 1])
            out_meta = Path(cmd[cmd.index("--out-meta") + 1])
            out_proof.write_bytes(b"proof")
            step = n_docs
            meta = {
                "mode": "prove",
                "n_docs_folded": n_docs,
                "step": step,
                "weights_digest": "626e2089fe38bc59",
                "state_digest": f"state{n_docs:064x}",
                "accumulator": f"acc{n_docs:061x}",
                "proof_bytes": 6,
                "prove_secs": 0.1,
                "verify_ms": 27.0,
            }
            out_meta.write_text(json.dumps(meta))

            class P:
                returncode = 0
                stdout = b""
                stderr = b""

            return P()
        else:

            class R:
                returncode = 0

                def __init__(self_inner):
                    self_inner.stdout = (
                        f"executed in 0.1s, total cycles: 100\n"
                        f"step:           {n_docs}\n"
                        f"weights_digest: 626e2089fe38bc59\n"
                        f"state_digest:   state{n_docs:064x}\n"
                        f"accumulator:    acc{n_docs:061x}\n"
                    ).encode()
                    self_inner.stderr = b""

            return R()

    return run


def test_execute_mode_matches_capsule(tmp_path) -> None:
    calls: list[list[str]] = []
    q = ProofQueue(
        mode="execute",
        ivc_bin="/fake/ivc",
        weights_path="/fake/weights.bin",
        work_root=tmp_path,
        session_lookup=lambda sid: FakeSession(6, f"state{6:064x}"),
        expected_weights_digest="626e2089fe38bc59",
        runner=fake_ivc_runner(calls),
    )
    docs = np.arange(3 * 2560, dtype="<i2").reshape(3, 2560)
    assert q.submit("s1", docs.tobytes()) is True
    # second submit while queued/running must still persist docs and catch up
    q.submit("s1", docs.tobytes())
    deadline = time.time() + 10
    while time.time() < deadline:
        st = q.status("s1")
        if st.get("done") and not st.get("queued") and st.get("step") == 6:
            break
        time.sleep(0.05)
    st = q.status("s1")
    assert st["done"], st
    assert st["queued"] is False
    assert st["step"] == 6
    assert st["matches_capsule"] is True
    assert st["matches_accumulator"] is True
    assert st["weights_digest"] == "626e2089fe38bc59"
    assert "error" not in st
    # cumulative docs file grew to all 6 docs (nothing dropped)
    assert (tmp_path / "s1" / "docs_all.bin").stat().st_size == 6 * 5120
    q.stop()


def test_prove_mode_recursive_chain(tmp_path) -> None:
    calls: list[list[str]] = []
    # track cumulative step across recursive batches (each call receives only the delta)
    cum_step = {"n": 0}

    def recursive_runner(cmd: list[str]) -> object:
        calls.append(cmd)
        docs = cmd[cmd.index("--docs") + 1]
        n_batch = Path(docs).stat().st_size // 5120
        cum_step["n"] += n_batch
        step = cum_step["n"]
        out_proof = Path(cmd[cmd.index("--out-proof") + 1])
        out_meta = Path(cmd[cmd.index("--out-meta") + 1])
        out_proof.write_bytes(b"proof")
        meta = {
            "mode": "prove",
            "n_docs_folded": n_batch,
            "step": step,
            "weights_digest": "626e2089fe38bc59",
            "state_digest": f"state{step:064x}",
            "accumulator": f"acc{step:061x}",
            "proof_bytes": 6,
            "prove_secs": 0.1,
            "verify_ms": 27.0,
        }
        out_meta.write_text(json.dumps(meta))

        class P:
            returncode = 0
            stdout = b""
            stderr = b""

        return P()

    q = ProofQueue(
        mode="prove",
        ivc_bin="/fake/ivc",
        weights_path="/fake/weights.bin",
        work_root=tmp_path,
        session_lookup=lambda sid: FakeSession(8, f"state{8:064x}"),
        expected_weights_digest="626e2089fe38bc59",
        runner=recursive_runner,
    )
    docs = np.arange(4 * 2560, dtype="<i2").reshape(4, 2560)
    q.submit("s2", docs.tobytes())
    deadline = time.time() + 10
    while time.time() < deadline:
        st = q.status("s2")
        if st.get("done") and not st.get("queued"):
            break
        time.sleep(0.05)
    st = q.status("s2")
    assert st["done"] and "error" not in st, st
    assert st["step"] == 4
    assert st["matches_capsule"] is False  # fake session head is ahead (step 8) — proving catches up below
    # genesis call: no --prev-proof
    assert not any("--prev-proof" in c for c in calls)
    # second batch must chain on top of the saved proof
    q.submit("s2", docs.tobytes())
    deadline = time.time() + 10
    seen_prev = False
    while time.time() < deadline:
        seen_prev = seen_prev or any("--prev-proof" in c for c in calls)
        st2 = q.status("s2")
        if st2.get("done") and not st2.get("queued") and st2.get("step") == 8:
            break
        time.sleep(0.05)
    assert seen_prev
    assert st2["step"] == 8
    assert st2["matches_capsule"] is True
    assert (tmp_path / "s2" / "proof.bin").exists()
    q.stop()


def test_weights_digest_mismatch_flagged(tmp_path) -> None:
    calls: list[list[str]] = []
    q = ProofQueue(
        mode="execute",
        ivc_bin="/fake/ivc",
        weights_path="/fake/weights.bin",
        work_root=tmp_path,
        session_lookup=lambda sid: FakeSession(3, f"state{3:064x}"),
        expected_weights_digest="deadbeef" * 8,
        runner=fake_ivc_runner(calls),
    )
    docs = np.zeros((3, 2560), dtype="<i2")
    q.submit("s3", docs.tobytes())
    deadline = time.time() + 10
    while time.time() < deadline:
        st = q.status("s3")
        if st.get("done"):
            break
        time.sleep(0.05)
    st = q.status("s3")
    assert "error" in st and "weights_digest" in st["error"]
    q.stop()


def test_status_before_any_job(tmp_path) -> None:
    q = ProofQueue(
        mode="execute",
        ivc_bin="/fake/ivc",
        weights_path="/fake/weights.bin",
        work_root=tmp_path,
        session_lookup=lambda sid: None,
        runner=fake_ivc_runner([]),
    )
    st = q.status("unknown")
    assert st == {"enabled": True, "mode": "execute", "queued": False, "done": False}
    q.stop()
