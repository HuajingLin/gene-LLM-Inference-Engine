#!/usr/bin/env python3
"""verify_kvpool.py - the paged KV cache bookkeeping (KvPool, Sequence) against an independent Python model.

gi_kvpool_check is driven with random scripts: allocate / retain / release blocks, create sequences, grow them
(reserve), set their length, truncate them, move them, destroy them, and write / read key and value rows. A small
model in this file predicts every answer: which block id comes out (blocks are reused most-recently-freed first),
reference counts, when the pool is full, which operations are errors, and the block table of every sequence. The
write / read commands also prove that different sequences, positions, layers and keys/values never overlap in
memory: every value read back must be the one written last to that exact (sequence, position, layer).

  python tests/verify_kvpool.py --exe build/gi_kvpool_check
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys


class Model:
    def __init__(self, layers, kv_dim, bs, nb):
        self.layers, self.kv_dim, self.bs, self.nb = layers, kv_dim, bs, nb
        self.free = list(range(nb - 1, -1, -1))        # stack, block 0 on top
        self.refs = [0] * nb
        self.seqs: dict[int, dict] = {}                 # id -> {"blocks": [...], "length": n}
        self.mem: dict[tuple, float] = {}               # (block, layer, row) -> key value

    def alloc(self):
        if not self.free:
            return "full"
        b = self.free.pop()
        self.refs[b] = 1
        return str(b)

    def retain(self, b):
        if not 0 <= b < self.nb or self.refs[b] < 1:
            return "error"
        self.refs[b] += 1
        return "ok"

    def _drop(self, b):
        self.refs[b] -= 1
        if self.refs[b] == 0:
            self.free.append(b)

    def release(self, b):
        if not 0 <= b < self.nb or self.refs[b] < 1:
            return "error"
        self._drop(b)
        return "ok"

    def ref(self, b):
        return str(self.refs[b]) if 0 <= b < self.nb else "error"

    def cap(self, s):
        return len(s["blocks"]) * self.bs

    def reserve(self, sid, n):
        s = self.seqs[sid]
        need = (max(n, 0) + self.bs - 1) // self.bs
        while len(s["blocks"]) < need:
            if not self.free:
                return "full"
            b = self.free.pop()
            self.refs[b] = 1
            s["blocks"].append(b)
        return "ok"

    def setlen(self, sid, n):
        s = self.seqs[sid]
        if n < 0 or n > self.cap(s):
            return "error"
        s["length"] = n
        return "ok"

    def _truncate(self, s, n):
        keep = (n + self.bs - 1) // self.bs
        while len(s["blocks"]) > keep:
            self._drop(s["blocks"].pop())
        s["length"] = n

    def truncate(self, sid, n):
        s = self.seqs[sid]
        if n < 0 or n > s["length"]:
            return "error"
        self._truncate(s, n)
        return "ok"

    def destroy(self, sid):
        s = self.seqs.pop(sid, None)
        if s is not None:
            self._truncate(s, 0)
        return "ok"

    def move(self, sid, tid):
        src = self.seqs[sid]
        moved = {"blocks": src["blocks"], "length": src["length"]}
        src["blocks"], src["length"] = [], 0
        old = self.seqs.get(tid)
        self.seqs[tid] = moved
        if old is not None:
            self._truncate(old, 0)
        return "ok"

    def assign(self, sid, tid):
        src, dst = self.seqs[sid], self.seqs[tid]
        if src is not dst:
            self._truncate(dst, 0)
            dst["blocks"], dst["length"] = src["blocks"], src["length"]
            src["blocks"], src["length"] = [], 0
        return "ok"

    def location(self, sid, pos, layer):
        s = self.seqs[sid]
        if pos < 0 or pos >= self.cap(s):
            return None
        return (s["blocks"][pos // self.bs], layer, pos % self.bs)

    def write(self, sid, pos, layer, v):
        loc = self.location(sid, pos, layer)
        if loc is None:
            return "error"
        self.mem[loc] = v
        return "ok"

    def read(self, sid, pos, layer):
        loc = self.location(sid, pos, layer)
        if loc is None:
            return "error"
        v = self.mem.get(loc)
        if v is None:
            return None   # never written: unspecified
        return f"{v:g} {v:g} {v + 0.5:g} {v + 0.5:g}"

    def state(self):
        out = str(len(self.free))
        for sid in sorted(self.seqs):
            s = self.seqs[sid]
            out += f" | {sid}: {s['length']} {self.cap(s)}" + "".join(f" {b}" for b in s["blocks"])
        return out


def make_script(rng: random.Random):
    layers, kv_dim = rng.choice([(1, 1), (2, 3), (3, 4)])
    bs, nb = rng.choice([1, 2, 3, 4, 7, 16]), rng.randint(1, 12)
    lines = [f"pool {layers} {kv_dim} {bs} {nb}"]
    model = Model(layers, kv_dim, bs, nb)
    expected = ["ok"]
    nid = 6
    for sid in range(3):
        lines.append(f"new {sid}")
        model.seqs[sid] = {"blocks": [], "length": 0}
        expected.append("ok")
    counter = 0
    held = []   # blocks allocated through "alloc" that we may retain / release
    for _ in range(rng.randint(30, 90)):
        op = rng.choice(["alloc", "alloc", "retain", "release", "refs", "reserve", "reserve", "reserve", "setlen", "setlen",
                         "truncate", "destroy", "newseq", "move", "assign", "write", "write", "read", "read", "state"])
        sids = sorted(model.seqs)
        if op in ("reserve", "setlen", "truncate", "destroy", "move", "assign", "write", "read") and not sids:
            op = "newseq"
        if op == "alloc":
            lines.append("alloc"); r = model.alloc(); expected.append(r)
            if r != "full":
                held.append(int(r))
        elif op == "retain":
            b = rng.choice(held + [rng.randint(-1, nb)] if held else [rng.randint(-1, nb)])
            lines.append(f"retain {b}"); r = model.retain(b); expected.append(r)
            if r == "ok" and b in held:
                held.append(b)
        elif op == "release":
            # only release references this test holds; otherwise use an id that must be an error
            bad = [x for x in range(-1, nb + 1) if not 0 <= x < nb or model.refs[x] == 0]
            b = rng.choice(held) if held and rng.random() < 0.7 else rng.choice(bad)
            lines.append(f"release {b}"); r = model.release(b); expected.append(r)
            if r == "ok" and b in held:
                held.remove(b)
        elif op == "refs":
            b = rng.randint(-1, nb)
            lines.append(f"refs {b}"); expected.append(model.ref(b))
        elif op == "newseq":
            if len(model.seqs) < 8:
                sid = nid; nid += 1
                lines.append(f"new {sid}"); model.seqs[sid] = {"blocks": [], "length": 0}; expected.append("ok")
            else:
                lines.append("state"); expected.append(model.state())
        elif op == "reserve":
            sid = rng.choice(sids); n = rng.randint(0, nb * bs + 3)
            lines.append(f"reserve {sid} {n}"); expected.append(model.reserve(sid, n))
        elif op == "setlen":
            sid = rng.choice(sids); n = rng.randint(-1, model.cap(model.seqs[sid]) + 2)
            lines.append(f"setlen {sid} {n}"); expected.append(model.setlen(sid, n))
        elif op == "truncate":
            sid = rng.choice(sids); n = rng.randint(-1, model.seqs[sid]["length"] + 2)
            lines.append(f"truncate {sid} {n}"); expected.append(model.truncate(sid, n))
        elif op == "destroy":
            sid = rng.choice(sids)
            lines.append(f"destroy {sid}"); expected.append(model.destroy(sid))
        elif op == "move" and len(sids) >= 1:
            sid = rng.choice(sids); tid = rng.choice(sids + [nid]) if len(model.seqs) < 8 else rng.choice(sids)
            if tid == nid:
                nid += 1
            if tid == sid:
                lines.append("state"); expected.append(model.state())
            else:
                lines.append(f"move {sid} {tid}"); expected.append(model.move(sid, tid))
        elif op == "assign" and len(sids) >= 1:
            sid, tid = rng.choice(sids), rng.choice(sids)
            lines.append(f"assign {sid} {tid}"); expected.append(model.assign(sid, tid))
        elif op == "write":
            sid = rng.choice(sids); cap = model.cap(model.seqs[sid])
            pos = rng.randint(-1, cap + 1); layer = rng.randrange(layers)
            counter += 1; v = float(counter)
            lines.append(f"write {sid} {pos} {layer} {v:g}"); expected.append(model.write(sid, pos, layer, v))
        elif op == "read":
            sid = rng.choice(sids); cap = model.cap(model.seqs[sid])
            pos = rng.randint(-1, cap + 1); layer = rng.randrange(layers)
            r = model.read(sid, pos, layer)
            if r is None:
                continue   # never written: skip
            lines.append(f"read {sid} {pos} {layer}"); expected.append(r)
        else:
            lines.append("state"); expected.append(model.state())
    lines.append("state"); expected.append(model.state())
    return lines, expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--scripts", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    failures = []
    ops = 0
    for n in range(args.scripts):
        lines, expected = make_script(rng)
        p = subprocess.run([args.exe], input="\n".join(lines) + "\n", capture_output=True, text=True)
        got = p.stdout.splitlines()
        ops += len(lines)
        if p.returncode != 0 or len(got) != len(lines):
            failures.append(f"script {n}: exit {p.returncode}, {len(got)} answers for {len(lines)} commands {p.stderr.strip()[:200]}")
            continue
        for i, (cmd, g, e) in enumerate(zip(lines, got, expected)):
            if g != e:
                failures.append(f"script {n}, command {i} {cmd!r}: got {g!r}, expected {e!r}")
                break
    bad = subprocess.run([args.exe], input="pool 0 1 1 1\npool 1 0 1 1\npool 1 1 0 1\npool 1 1 1 0\nalloc\n", capture_output=True, text=True)
    if bad.stdout.splitlines()[:4] != ["error"] * 4:
        failures.append(f"bad pool sizes were not all rejected: {bad.stdout!r}")
    if failures:
        print("FAIL")
        for f in failures[:15]:
            print("  -", f)
        return 1
    print(f"PASS  ({args.scripts} random scripts, {ops} commands: allocation order, reference counts, reserve / truncate / move, "
          f"error cases and memory addressing match the model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
