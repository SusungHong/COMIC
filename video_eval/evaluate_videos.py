"""Automated video evaluation from COMIC: Agentic Sketch Comedy Generation (arXiv:2603.11048).
Reports win rate and inter/intra diversity per critic set and combined."""

import argparse, csv, json, os, random, re, time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
try:
    from google import genai
    from google.genai import types as gtypes
    _GEMINI = True
except ImportError:
    _GEMINI = False; genai = gtypes = None  # type: ignore

DIR = Path(__file__).resolve().parent
DATA = DIR.parent / "data"

INSTRUCTIONS = """Core Objective
Evaluate multiple candidate scene videos and vote for the one that you find most appealing and enjoyable to watch.

Requirements (MUST DO)
* Apply your perspective to evaluate the videos
* Output in the specified format

Constraints (MUST AVOID)
* Indecisive evaluations

Voting Framework
* Evaluate each candidate video based on your perspective
* Select the candidate that you would most enjoy watching

Input Format
You will receive multiple candidate videos labeled as "Candidate #1", "Candidate #2", etc.

Output Format
[VOTING START]

[WINNER START]Candidate #[number][WINNER END]

[REVIEW START]
Based on your perspective, provide a paragraph explaining your decision
[REVIEW END]

[VOTING END]

The candidate number should be an integer from 1 to N, where N is the number of candidate videos provided."""


class GeminiCriticAgent:
    MAX_RETRIES = 10; RETRY_DELAY = 30

    def __init__(self, model: str, spec: Optional[Dict] = None):
        self.model, self.spec = model, spec or {}
        self.name = self.spec.get("name", "Agent")
        self._ki = 0
        base = self.spec.get("content") or self.spec.get("system_instructions")
        if base:
            for ph in ["{{TASK-SPECIFIC INSTRUCTIONS WILL BE INSERTED HERE}}",
                        "{TASK-SPECIFIC INSTRUCTIONS WILL BE INSERTED HERE}"]:
                if ph in base: base = base.replace(ph, INSTRUCTIONS); break
            self.instr = base
        else:
            self.instr = INSTRUCTIONS
        self.client = None
        if _GEMINI:
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                try: self.client = genai.Client(api_key=key)
                except Exception: pass

    def _yt_id(self, url):
        if not url: return None
        m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        return m.group(1) if m else None

    def _local(self, url):
        vid = self._yt_id(url)
        if not vid: return None
        p = DIR / "val_videos" / f"{vid}.mp4"
        return str(p) if p.exists() else None

    def _wait_active(self, f, c, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = c.files.get(name=f.name).state
            if s == gtypes.FileState.ACTIVE: return True
            if s == gtypes.FileState.FAILED: return False
            time.sleep(5)
        return False

    def _add_vid(self, parts, url, lp, c, ufs):
        path = lp or self._local(url or "")
        if path:
            f = c.files.upload(file=path); ufs.append(f)
            if self._wait_active(f, c):
                parts.append(c.files.get(name=f.name)); return
        parts.append(gtypes.Part(file_data=gtypes.FileData(file_uri=url)))

    def _parse(self, txt):
        v = re.search(r"\[VOTING START\](.+?)\[VOTING END\]", txt, re.DOTALL | re.I)
        if not v: return {"best_candidate": random.choice([1, 2]), "reasoning": ""}
        c = v.group(1)
        w = re.search(r"\[WINNER START\](.+?)\[WINNER END\]", c, re.I)
        m = re.search(r"Candidate\s*#?(\d+)", w.group(1) if w else c, re.I)
        best = int(m.group(1)) if m and 1 <= int(m.group(1)) <= 2 else random.choice([1, 2])
        r = re.search(r"\[REVIEW START\](.+?)\[REVIEW END\]", c, re.DOTALL | re.I)
        return {"best_candidate": best, "reasoning": r.group(1).strip() if r else ""}

    def evaluate(self, u1, u2, keys=None, lp1=None, lp2=None) -> Dict:
        if not self.client and not keys:
            return {"best_candidate": 1, "error": "Gemini not available", "agent_name": self.name}
        if not (u1 or lp1) or not (u2 or lp2):
            return {"best_candidate": 1, "error": "Missing video", "agent_name": self.name}
        ks = list(keys) if keys else [None]
        si = self._ki % len(ks) if keys else 0
        err = None
        for off in range(len(ks)):
            ki = (si + off) % len(ks); k = ks[ki]
            try: c = genai.Client(api_key=k) if k else self.client
            except Exception as e: err = e; continue
            ufs = []
            try:
                p = [self.instr, "Candidate Videos:\n", "Candidate #1:"]
                self._add_vid(p, u1, lp1, c, ufs)
                p.append("Candidate #2:")
                self._add_vid(p, u2, lp2, c, ufs)
                p.append("Please evaluate all candidate videos and provide your response following the format specified in your instructions. Select the best candidate (1 or 2).")
                for att in range(self.MAX_RETRIES):
                    try:
                        r = c.models.generate_content(model=self.model, contents=p)
                        out = self._parse(r.text)
                        out.update(agent_name=self.name, raw_response=r.text)
                        u = getattr(r, "usage_metadata", None)
                        if u:
                            out["input_tokens"] = getattr(u, "prompt_token_count", None) or getattr(u, "input_token_count", None)
                            out["output_tokens"] = getattr(u, "candidates_token_count", None) or getattr(u, "output_token_count", None)
                        if keys: self._ki = ki
                        return out
                    except Exception as e:
                        err = e
                        if att < self.MAX_RETRIES - 1: time.sleep(self.RETRY_DELAY * (att + 1))
            finally:
                for f in ufs:
                    try: c.files.delete(name=f.name)
                    except Exception: pass
        return {"best_candidate": 1, "error": str(err), "agent_name": self.name}


@dataclass
class Ref:
    id: str; url: str; channel: str

@dataclass
class Gen:
    method: str; path: Path
    @property
    def did(self): return self.path.stem

def load_videos(d: Path) -> Dict[str, List[Gen]]:
    out: Dict[str, List[Gen]] = {}
    for sub in sorted(x for x in d.iterdir() if x.is_dir()):
        vids = sorted(sub.glob("*.mp4"))
        if vids: out[sub.name] = [Gen(sub.name, v) for v in vids]
    return out

def load_refs(p: Path) -> Dict[str, List[Ref]]:
    out: Dict[str, List[Ref]] = {}
    if not p.exists(): return out
    ch = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            ch = line[1:].strip(); continue
        if not ch: continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out.setdefault(ch, []).append(Ref(parts[0], parts[1], ch))
    return out

def load_critics(p: Path) -> List[Dict]:
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["video_critics"] if isinstance(d.get("video_critics"), list) else (d if isinstance(d, list) else [])

def load_clm(p: Path) -> Dict[str, str]:
    try: return json.loads(p.read_text(encoding="utf-8")).get("channel_leads_map", {})
    except Exception: return {}

def build_cca(paths) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for p in paths:
        for ch, cid in load_clm(p).items(): out.setdefault(cid, set()).add(ch)
    return out

def load_keys(p: Path) -> List[str]:
    if not p.exists(): return []
    s, o = set(), []
    for l in p.read_text(encoding="utf-8").splitlines():
        k = l.split("#")[0].strip()
        if k and k not in s: s.add(k); o.append(k)
    return o


FIELDS = ["method","test_channel","win_rate","wins","trials","method_video_ids",
          "reference_video_ids","evaluation_calls","input_tokens","output_tokens",
          "gemini_outputs","critic_id","critic_name","critic_set","metadata"]

def _csv_append(row, path, first):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if first: w.writeheader()
        w.writerow(row)

def _csv_write(rows, path):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def run_cmp(agent, refs, gens, method, ch, double, delay, keys):
    w = t = ti = to = 0; calls = []; gouts = []
    for ref in refs:
        for gen in gens:
            ords = [(ref.url, "", None, str(gen.path), True)]
            if double: ords.append(("", ref.url, str(gen.path), None, False))
            elif random.random() >= 0.5: ords = [("", ref.url, str(gen.path), None, False)]
            for u1, u2, l1, l2, gw2 in ords:
                t0 = time.time()
                r = agent.evaluate(u1, u2, keys=keys, lp1=l1, lp2=l2)
                dt = time.time() - t0
                if r.get("error"):
                    print(f"      [ERR] {ref.id[:24]} vs {gen.did[:24]}: {r['error']}")
                    calls.append({"ref_id": ref.id, "gen_id": gen.did, "gen_wins": False, "error": r["error"]})
                    t += 1; time.sleep(delay); continue
                b = r.get("best_candidate", 1)
                gw = (b == 2) if gw2 else (b == 1)
                w += int(gw); t += 1
                ri = r.get("input_tokens") or 0; ro = r.get("output_tokens") or 0
                ti += ri; to += ro
                raw = r.get("raw_response") or ""
                calls.append({"ref_id": ref.id, "gen_id": gen.did, "gen_wins": gw, "raw_response": raw})
                gouts.append(raw)
                print(f"      [eval] {ref.id[:24]} vs {gen.did[:24]} -> #{b} gw={gw} {dt:.1f}s")
                time.sleep(delay)
    return {"method": method, "test_channel": ch, "win_rate": w / t if t else 0,
            "wins": w, "trials": t, "method_video_ids": json.dumps([g.did for g in gens]),
            "reference_video_ids": json.dumps([r.id for r in refs]),
            "evaluation_calls": json.dumps(calls, ensure_ascii=False),
            "input_tokens": ti, "output_tokens": to,
            "gemini_outputs": json.dumps(gouts, ensure_ascii=False)}


def _prob_matrix(rows):
    W = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    T = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    E, B, A = set(), set(), set()
    for row in rows:
        e = row["critic_id"]; E.add(e)
        ec = row["evaluation_calls"]
        if isinstance(ec, str): ec = json.loads(ec)
        for c in ec:
            if c.get("error"): continue
            a, b, gw = c["ref_id"], c["gen_id"], c["gen_wins"]
            A.add(a); B.add(b); T[e][b][a] += 1
            if gw: W[e][b][a] += 1
    E, B, A = sorted(E), sorted(B), sorted(A)
    P = {e: {b: {a: (W[e][b][a] / T[e][b][a] if T[e][b][a] else None) for a in A} for b in B} for e in E}
    return P, E, B, A

def metrics(rows):
    if not rows: return {"avg_win_rate": 0.0, "g_norm_inter": 0.0, "g_norm_intra": 0.0}
    P, E, B, A = _prob_matrix(rows)
    vs = [P[e][b][a] for e in E for b in B for a in A if P[e][b][a] is not None]
    wr = float(np.mean(vs)) if vs else 0.0
    prof = {b: np.array([P[e][b][a] for e in E for a in A if P[e][b][a] is not None]) for b in B}
    intra = float(np.mean([np.var(prof[b]) for b in B if len(prof[b]) > 0])) if B else 0.0
    iv = []
    for e in E:
        for a in A:
            pv = [P[e][b][a] for b in B if P[e][b][a] is not None]
            if len(pv) > 1: iv.append(np.var(pv))
    inter = float(np.mean(iv)) if iv else 0.0
    mx = wr * (1 - wr)
    gi, ga = (inter / mx, intra / mx) if mx > 1e-9 else (0.0, 0.0)
    return {"avg_win_rate": wr, "g_norm_inter": gi, "g_norm_intra": ga}

def _metrics_table(results, label, critic_sets=None):
    rows = results
    if critic_sets is not None:
        rows = [r for r in results if r.get("critic_set") in critic_sets]
    if not rows: return {}
    grp = defaultdict(lambda: defaultdict(list))
    for r in rows: grp[r["method"]][r["test_channel"]].append(r)
    meths = sorted(grp, key=lambda m: (0 if m == "ours" else 1, m))
    chs = sorted({c for m in grp for c in grp[m]})
    out = {}
    for m in meths:
        pc = {c: metrics(grp[m][c]) for c in chs if grp[m].get(c)}
        avg = {k: float(np.mean([pc[c][k] for c in pc])) for k in ["avg_win_rate", "g_norm_inter", "g_norm_intra"]} if pc else {}
        out[m] = {"per_channel": pc, "avg": avg}
    print(f"\n{'='*72}\n  {label} — avg across channels\n{'='*72}")
    print(f"{'Method':<12} {'WinRate':>10} {'G-N-Inter':>12} {'G-N-Intra':>12}")
    print("-" * 72)
    for m in meths:
        a = out[m]["avg"]
        if a: print(f"{m:<12} {a['avg_win_rate']:>10.4f} {a['g_norm_inter']:>12.4f} {a['g_norm_intra']:>12.4f}")
    print(f"\n{'='*72}\n  {label} — per-channel win rate\n{'='*72}")
    hdr = f"{'Method':<12}"; hdr += "".join(f" {c[:14]:>14}" for c in chs); hdr += f" {'MEAN':>8}"
    print(hdr); print("-" * len(hdr))
    for m in meths:
        pc = out[m]["per_channel"]; vs = []
        ln = f"{m:<12}"
        for c in chs:
            if c in pc: v = pc[c]["avg_win_rate"]; ln += f" {v:>14.4f}"; vs.append(v)
            else: ln += f" {'—':>14}"
        ln += f" {float(np.mean(vs)):>8.4f}" if vs else ""
        print(ln)
    print()
    return out

def print_all_metrics(results, combined=True):
    csets = sorted({r.get("critic_set", "") for r in results})
    per_set = {}
    for cs in csets:
        per_set[cs] = _metrics_table(results, f"CRITIC SET: {cs}", critic_sets={cs})
    out = {f"set_{cs}": per_set[cs] for cs in csets}
    if combined:
        out["combined"] = _metrics_table(results, "COMBINED (global_best + channel_best)")
    return out

def save(results, csv_path, args_d, combined=True):
    _csv_write(results, csv_path)
    m = print_all_metrics(results, combined=combined)
    jp = csv_path.with_suffix(".json"); jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w", encoding="utf-8") as f: json.dump({**args_d, "metrics": m}, f, indent=2, ensure_ascii=False)
    print(f"  CSV: {csv_path}\n  JSON: {jp}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos-dir", type=Path, default=DATA / "videos")
    p.add_argument("--test-refs", type=Path, default=DATA / "test" / "middle.txt")
    p.add_argument("--critics", type=Path, nargs="+", default=[
        DATA / "critics" / "global_best.json",
        DATA / "critics" / "channel_best.json"])
    p.add_argument("--gemini-model", default="gemini-3-flash-preview")
    p.add_argument("--request-delay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--double-evaluation", action="store_true")
    p.add_argument("--gemini-api-keys-file", type=Path, default=DIR / "gemini_api_keys.txt")
    p.add_argument("--methods", type=str, nargs="*", default=None)
    p.add_argument("--no-combined", action="store_true")
    p.add_argument("--max-refs", type=int, default=None)
    p.add_argument("--max-gens", type=int, default=None)
    a = p.parse_args(); random.seed(a.seed)

    if not a.output_csv:
        a.output_csv = DIR / "evaluations" / f"eval_{datetime.now():%Y%m%d_%H%M%S}.csv"

    if not a.videos_dir.exists(): raise SystemExit(f"Not found: {a.videos_dir}")
    meths = load_videos(a.videos_dir)
    if a.methods: meths = {m: v for m, v in meths.items() if m in a.methods}
    if a.max_gens: meths = {m: v[:a.max_gens] for m, v in meths.items()}
    if not meths: raise SystemExit("No method videos found.")
    print(f"Methods: {list(meths.keys())}")

    refs = load_refs(a.test_refs)
    if a.max_refs: refs = {ch: v[:a.max_refs] for ch, v in refs.items()}
    if not refs: raise SystemExit(f"No test refs found in {a.test_refs}")
    print(f"Channels: {list(refs.keys())}")

    specs = []; seen = set()
    for cp in a.critics:
        if not cp.exists(): continue
        for s in load_critics(cp):
            cid = s.get("id", "?")
            if cid not in seen: seen.add(cid); specs.append((cp.stem, s))
    if not specs: raise SystemExit("No critics.")
    cca = build_cca(a.critics)
    specs.sort(key=lambda x: (0 if x[1].get("id") not in cca else 1))

    keys = load_keys(a.gemini_api_keys_file)
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not keys and env_key:
        keys = [env_key]
    if not keys:
        raise SystemExit(f"No API keys found (checked {a.gemini_api_keys_file} and env vars GOOGLE_API_KEY / GEMINI_API_KEY)")
    print(f"Keys: {len(keys)}, model: {a.gemini_model}, double: {a.double_evaluation}")

    ch_ord = []
    for cp in a.critics:
        for c in load_clm(cp):
            if c not in ch_ord: ch_ord.append(c)
    ch_ord += sorted(c for c in refs if c not in ch_ord)

    sm = sorted(meths, key=lambda m: (0 if m == "ours" else 1, m))
    bn, bd = a.output_csv.stem, a.output_csv.parent
    ad = {"videos_dir": str(a.videos_dir), "test_refs": str(a.test_refs),
          "model": a.gemini_model, "double": a.double_evaluation, "seed": a.seed}

    cs = {}; results = []
    for ci, (sn, sp) in enumerate(specs, 1):
        cid = sp.get("id", f"C{ci}"); cp = bd / f"{bn}_critic_{ci:02d}_{cid}.csv"
        asgn = cca.get(cid)
        chs = [c for c in ch_ord if c in asgn] if asgn else list(ch_ord)
        if cp.exists(): cp.unlink()
        cs[cid] = dict(i=ci, sn=sn, sp=sp, cp=cp, fst=True, chs=chs, ag=None)

    for mn in sm:
        gv = meths[mn]
        print(f"\n{'#'*72}\nMETHOD: {mn} ({len(gv)} videos)\n{'#'*72}")
        for ci, (sn, sp) in enumerate(specs, 1):
            cid = sp.get("id", f"C{ci}"); c = cs[cid]
            random.seed(a.seed)
            if not c["ag"]: c["ag"] = GeminiCriticAgent(a.gemini_model, sp)
            print(f"  Critic {ci}/{len(specs)}: {sp.get('name', cid)} [{sn}]")
            for ch in c["chs"]:
                if ch not in refs: continue
                rv = refs[ch]
                print(f"    {mn} vs {ch}: {len(gv)}x{len(rv)}")
                row = run_cmp(c["ag"], rv, gv, mn, ch, a.double_evaluation, a.request_delay, keys)
                row.update(critic_id=cid, critic_name=sp.get("name", cid), critic_set=sn,
                           metadata=json.dumps({"model": a.gemini_model}))
                results.append(row); _csv_append(row, c["cp"], c["fst"]); c["fst"] = False
                print(f"      wr={row['win_rate']:.4f} ({row['wins']}/{row['trials']})")
        save(results, a.output_csv, ad, combined=not a.no_combined)

if __name__ == "__main__":
    main()
