#!/usr/bin/env python3
"""verify_prefixcache.py - PrefixCache, Sequence::adopt and copy-on-write against an independent Python model.

gi_prefixcache_check is driven with random scripts (tiny token alphabet, so that prefixes collide a lot): sequences
are filled, their full blocks registered in the cache, new sequences adopt the longest cached prefix, sequences are
rewound into shared blocks and extended again (copy-on-write), the cache evicts. The model in this file predicts
every answer: matched block counts and ids, number of new cache entries, which entries are evictable and which one
goes first (least recently used leaf), pool reference counts, free block order, block tables, and the value read
back at every position (so a write into a shared block that leaked into another sequence would show up).

  python tests/verify_prefixcache.py --exe build/gi_prefixcache_check
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys


class Model:
    def __init__(self, bs, nb):
        self.bs, self.nb = bs, nb
        self.free = list(range(nb - 1, -1, -1))
        self.refs = [0] * nb
        self.mem = [[None] * bs for _ in range(nb)]      # value stored in each row (key value of the first float)
        self.seqs = {}                                    # id -> {"blocks": [], "length": n, "tokens": []}
        self.entries = {}                                 # block -> {"parent": b|-1, "tokens": tuple, "tick": n, "children": n}
        self.index = {}                                   # (parent, tokens) -> block
        self.clock = 0

    # ---- pool ----
    def alloc(self):
        if not self.free:
            return -1
        b = self.free.pop()
        self.refs[b] = 1
        return b

    def release(self, b):
        self.refs[b] -= 1
        assert self.refs[b] >= 0
        if self.refs[b] == 0:
            self.free.append(b)

    # ---- sequences ----
    def reserve(self, s, n):
        need = (n + self.bs - 1) // self.bs
        while len(s["blocks"]) < need:
            b = self.alloc()
            if b < 0:
                return False
            s["blocks"].append(b)
        return True

    def truncate_blocks(self, s, n):
        keep = (n + self.bs - 1) // self.bs
        while len(s["blocks"]) > keep:
            self.release(s["blocks"].pop())
        s["length"] = n

    def make_writable(self, s, idx):
        b = s["blocks"][idx]
        if self.refs[b] <= 1:
            return True
        fresh = self.alloc()
        if fresh < 0:
            return False
        self.mem[fresh] = list(self.mem[b])
        self.release(b)
        s["blocks"][idx] = fresh
        return True

    def fill(self, sid, toks):
        s = self.seqs[sid]
        ln = s["length"]
        if not self.reserve(s, ln + len(toks)):
            self.truncate_blocks(s, ln)
            return "full"
        if toks:
            for idx in range(ln // self.bs, (ln + len(toks) - 1) // self.bs + 1):
                if not self.make_writable(s, idx):
                    self.truncate_blocks(s, ln)
                    return "full"
        for i, t in enumerate(toks):
            pos = ln + i
            self.mem[s["blocks"][pos // self.bs]][pos % self.bs] = float(t)
        s["length"] = ln + len(toks)
        s["tokens"] += toks
        return "ok"

    # ---- cache ----
    def insert(self, sid):
        s = self.seqs[sid]
        full = min(s["length"] // self.bs, len(s["blocks"]))
        parent, added = -1, 0
        for i in range(full):
            key = (parent, tuple(s["tokens"][i * self.bs:(i + 1) * self.bs]))
            if key in self.index:
                self.clock += 1
                self.entries[self.index[key]]["tick"] = self.clock
                parent = self.index[key]
                continue
            b = s["blocks"][i]
            if b in self.entries:
                break
            self.clock += 1
            self.entries[b] = {"parent": parent, "tokens": key[1], "tick": self.clock, "children": 0}
            if parent >= 0:
                self.entries[parent]["children"] += 1
            self.refs[b] += 1
            self.index[key] = b
            parent = b
            added += 1
        return added

    def adopt(self, sid, toks):
        parent, blocks = -1, []
        i = 0
        while (i + 1) * self.bs <= len(toks):
            key = (parent, tuple(toks[i * self.bs:(i + 1) * self.bs]))
            if key not in self.index:
                break
            b = self.index[key]
            self.refs[b] += 1
            self.clock += 1
            self.entries[b]["tick"] = self.clock
            blocks.append(b)
            parent = b
            i += 1
        s = self.seqs[sid]
        if s["blocks"] or s["length"] != 0:
            for b in blocks:
                self.release(b)
            return "error"
        s["blocks"] = blocks
        s["length"] = len(blocks) * self.bs
        s["tokens"] = list(toks[:len(blocks) * self.bs])
        return " ".join([str(len(blocks))] + [str(b) for b in blocks])

    def remove(self, b):
        e = self.entries.pop(b)
        del self.index[(e["parent"], e["tokens"])]
        if e["parent"] >= 0:
            self.entries[e["parent"]]["children"] -= 1
        self.release(b)

    def evict(self, n):
        freed = 0
        while freed < n:
            best = None   # (frees, tick, block): prefer leaves only the cache holds, then the least recently used
            for b, e in self.entries.items():
                if e["children"] != 0:
                    continue
                frees = self.refs[b] == 1
                if not frees:
                    a, above = e["parent"], False
                    while a >= 0:
                        above = above or self.refs[a] == 1
                        a = self.entries[a]["parent"]
                    if not above:
                        continue
                key = (0 if frees else 1, e["tick"])
                if best is None or key < best[0]:
                    best = (key, b, frees)
            if best is None:
                break
            self.remove(best[1])
            if best[2]:
                freed += 1
        return freed

    def evictable(self):
        return sum(1 for b in self.entries if self.refs[b] == 1)

    def state(self):
        out = f"{len(self.free)} {len(self.entries)} |" + "".join(f" {r}" for r in self.refs)
        for sid in sorted(self.seqs):
            s = self.seqs[sid]
            out += f" | {sid}: {s['length']}" + "".join(f" {b}" for b in s["blocks"])
        return out


def make_script(rng):
    bs, nb = rng.choice([1, 2, 3, 4]), rng.randint(3, 14)
    alphabet = rng.choice([2, 3])
    m = Model(bs, nb)
    lines, expected = [f"pool {bs} {nb}"], ["ok"]
    live = set()
    for _ in range(rng.randint(40, 120)):
        op = rng.choice(["seq", "fill", "fill", "fill", "insert", "insert", "adopt", "adopt", "destroy", "evict",
                         "evictable", "truncate", "truncate", "cow", "read", "read", "read", "state"])
        sids = sorted(live)
        if op not in ("seq", "evict", "evictable", "state", "adopt") and not sids:
            op = "seq"
        if op == "seq" or (op == "adopt" and not sids):
            sid = rng.randint(0, 5)
            if sid in live:                       # replacing a sequence destroys the old one
                m.truncate_blocks(m.seqs[sid], 0)
            m.seqs[sid] = {"blocks": [], "length": 0, "tokens": []}
            live.add(sid)
            lines.append(f"seq {sid}"); expected.append("ok")
        elif op == "fill":
            sid = rng.choice(sids)
            toks = [rng.randrange(alphabet) for _ in range(rng.randint(0, 3 * bs + 1))]
            lines.append(f"fill {sid} " + " ".join(map(str, toks))); expected.append(m.fill(sid, toks))
        elif op == "insert":
            sid = rng.choice(sids)
            lines.append(f"insert {sid}"); expected.append(str(m.insert(sid)))
        elif op == "adopt":
            sid = rng.choice(sids)
            toks = [rng.randrange(alphabet) for _ in range(rng.randint(0, 5 * bs))]
            lines.append(f"adopt {sid} " + " ".join(map(str, toks))); expected.append(m.adopt(sid, toks))
        elif op == "destroy":
            sid = rng.choice(sids)
            m.truncate_blocks(m.seqs.pop(sid), 0); live.discard(sid)
            lines.append(f"destroy {sid}"); expected.append("ok")
        elif op == "evict":
            n = rng.randint(0, 4)
            lines.append(f"evict {n}"); expected.append(str(m.evict(n)))
        elif op == "evictable":
            lines.append("evictable"); expected.append(str(m.evictable()))
        elif op == "truncate":
            sid = rng.choice(sids)
            n = rng.randint(-1, m.seqs[sid]["length"] + 1)
            lines.append(f"truncate {sid} {n}")
            if 0 <= n <= m.seqs[sid]["length"]:
                m.truncate_blocks(m.seqs[sid], n)
                m.seqs[sid]["tokens"] = m.seqs[sid]["tokens"][:n]
                expected.append("ok")
            else:
                expected.append("error")
        elif op == "cow":
            sid = rng.choice(sids)
            s = m.seqs[sid]
            idx = rng.randint(-1, len(s["blocks"]))
            lines.append(f"cow {sid} {idx}")
            if 0 <= idx < len(s["blocks"]):
                ok = m.make_writable(s, idx)
                expected.append(f"ok {s['blocks'][idx]}" if ok else "full")
            else:
                expected.append("error")
        elif op == "read":
            sid = rng.choice(sids)
            s = m.seqs[sid]
            pos = rng.randint(-1, s["length"] + 1)
            lines.append(f"read {sid} {pos}")
            if 0 <= pos < s["length"]:
                v = m.mem[s["blocks"][pos // bs]][pos % bs]
                expected.append(f"{v:g}")
            else:
                expected.append("error")
        else:
            lines.append("state"); expected.append(m.state())
    lines.append("state"); expected.append(m.state())
    return lines, expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--scripts", type=int, default=600)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    failures, ops = [], 0
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
    if failures:
        print("FAIL")
        for f in failures[:12]:
            print("  -", f)
        return 1
    print(f"PASS  ({args.scripts} random scripts, {ops} commands: cache matches, insertions, LRU eviction, reference counts and "
          f"copy-on-write agree with the model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
