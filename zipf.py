#!/usr/bin/env python3
"""Rank the words in my commit messages and plot them against Zipf's law.

Run by .github/workflows/zipf.yml every morning. Counts accumulate in
data/counts.json, so the corpus grows with every push instead of resetting.

  python scripts/zipf.py          # fetch from the GitHub API
  python scripts/zipf.py --demo   # render from synthetic data, no network
"""

import collections
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

def repo_root():
    """Walk up from this file to the checkout, wherever the script was put."""
    for d in pathlib.Path(__file__).resolve().parents:
        if (d / ".git").exists():
            return d
    return pathlib.Path.cwd()


ROOT = repo_root()
STATE = ROOT / "data" / "counts.json"
OUT = ROOT / "zipf.svg"

USER = os.environ.get("GH_USER", "tanayagrawal")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
MAX_PAGES = 10          # search caps at 1000 results
MAX_POINTS = 1200       # keep the svg small
INK_LIGHT, INK_DARK = "#1f2328", "#e6edf3"

WORD = re.compile(r"[a-z][a-z'-]*[a-z]")
TRAILER = re.compile(r"^(co-authored-by|signed-off-by|reviewed-by|reported-by|"
                     r"cc|refs?|closes|fixes #)\b", re.I)
NOISE = re.compile(r"https?://\S+|`[^`]*`|\b[0-9a-f]{7,40}\b|\S+\.\w{1,4}\b|\S*[/_@]\S*")
SKIP_MSG = re.compile(r"^(merge|revert|bump|initial commit)\b", re.I)


# ---------------------------------------------------------------- fetching

def fetch_messages():
    """Most recent commits authored by USER across public repos."""
    out = {}
    for page in range(1, MAX_PAGES + 1):
        q = urllib.parse.urlencode({
            "q": f"author:{USER}", "sort": "author-date",
            "order": "desc", "per_page": 100, "page": page,
        })
        req = urllib.request.Request(
            f"https://api.github.com/search/commits?{q}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"{USER}-zipf",
                **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"  page {page}: HTTP {e.code} — stopping", file=sys.stderr)
            break
        items = data.get("items", [])
        for it in items:
            out[it["sha"]] = it["commit"]["message"]
        print(f"  page {page}: {len(items)} commits")
        if len(items) < 100:
            break
        time.sleep(2)  # search api is 30 req/min
    return out


# ---------------------------------------------------------------- counting

def tokenize(message):
    lines = [l for l in message.splitlines() if not TRAILER.match(l.strip())]
    text = NOISE.sub(" ", " ".join(lines).lower())
    return WORD.findall(text)


def tally(messages, state):
    counts = collections.Counter(state.get("counts", {}))
    seen = set(state.get("shas", []))
    added = 0
    for sha, msg in messages.items():
        if sha in seen or SKIP_MSG.match(msg.strip()):
            continue
        seen.add(sha)
        counts.update(tokenize(msg))
        added += 1
    return counts, seen, added


# ---------------------------------------------------------------- fitting

def fit_slope(ranks, freqs):
    """Least squares on log10(freq) ~ a + b*log10(rank). Returns -b."""
    xs = [math.log10(r) for r in ranks]
    ys = [math.log10(f) for f in freqs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return -(num / den) if den else 0.0


# ---------------------------------------------------------------- rendering

def render(counts, n_commits):
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    freqs = [c for _, c in ranked]
    alpha = fit_slope(range(1, len(freqs) + 1), freqs)

    W, H = 600, 340
    L, R, T, B = 52, 22, 34, 52
    pw, ph = W - L - R, H - T - B
    # axes run to the data, not to the next decade, so nothing sits empty
    xmax = math.log10(len(ranked)) + .04
    ymax = math.log10(freqs[0]) + .07

    def px(rank):
        return L + math.log10(rank) / xmax * pw

    def py(freq):
        return T + ph - math.log10(freq) / ymax * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" aria-label="Log-log plot of word '
         f'frequency against rank in my commit messages, fitted exponent '
         f'{alpha:.2f}">']

    s.append(f"""<style>
text{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:{INK_LIGHT}}}
.grid{{stroke:{INK_LIGHT};stroke-width:1;opacity:.12}}
.dot{{fill:{INK_LIGHT};opacity:.5}}
.ideal{{stroke:{INK_LIGHT};stroke-width:1.25;stroke-dasharray:4 4;fill:none;opacity:.35}}
.fit{{stroke:{INK_LIGHT};stroke-width:1.75;fill:none;opacity:.9}}
.tick{{font-size:10px;opacity:.45}}
.lbl{{font-size:10.5px;opacity:.85}}
.cap{{font-size:11px;opacity:.55}}
.eyebrow{{font-size:10px;opacity:.4;letter-spacing:.08em}}
@media (prefers-color-scheme:dark){{
text{{fill:{INK_DARK}}}.grid,.ideal,.fit{{stroke:{INK_DARK}}}.dot{{fill:{INK_DARK}}}}}
.dot{{opacity:0;animation:rise .5s ease-out forwards}}
@keyframes rise{{from{{opacity:0;transform:translate(0,7px)}}to{{opacity:.5;transform:translate(0,0)}}}}
{''.join(f'.b{i}{{animation-delay:{i*0.075:.3f}s}}' for i in range(1, 12))}
.ideal,.fit{{stroke-dashoffset:0;animation:draw 1.1s .55s ease-out backwards}}
@keyframes draw{{from{{stroke-dashoffset:{pw*1.6:.0f}}}to{{stroke-dashoffset:0}}}}
.fit{{stroke-dasharray:{pw*1.6:.0f}}}
@media (prefers-reduced-motion:reduce){{
.dot{{opacity:.5;animation:none}}.ideal,.fit{{animation:none}}
.fit{{stroke-dasharray:none}}}}
</style>""")

    # grid + ticks
    for d in range(int(xmax) + 1):
        x = L + d / xmax * pw
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{T+ph+16}" text-anchor="middle">'
                 f'{10**d:,}</text>')
    for d in range(int(ymax) + 1):
        y = T + ph - d / ymax * ph
        s.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">'
                 f'{10**d:,}</text>')

    # the two lines: pure 1/rank, and the actual fit
    top = freqs[0]
    last = len(ranked)
    s.append(f'<path class="ideal" d="M{px(1):.1f} {py(top):.1f} '
             f'L{px(last):.1f} {py(max(top/last, 1)):.1f}"/>')
    fit_end = 10 ** (math.log10(top) - alpha * math.log10(last))
    s.append(f'<path class="fit" d="M{px(1):.1f} {py(top):.1f} '
             f'L{px(last):.1f} {py(max(fit_end, 1)):.1f}"/>')

    # points, bucketed into 12 bands so the stagger costs 11 css rules
    pts = ranked[:MAX_POINTS]
    for i, (_, c) in enumerate(pts):
        band = min(11, int(12 * i / len(pts)))
        s.append(f'<circle class="dot b{band}" cx="{px(i+1):.1f}" '
                 f'cy="{py(c):.1f}" r="2.1"/>')

    # label a few words to show the 1/rank fall-off
    for rank in (1, 2, 3, 10, 100, 1000):
        if rank > len(ranked):
            continue
        w, c = ranked[rank - 1]
        s.append(f'<text class="lbl" x="{px(rank)+7:.1f}" y="{py(c)-6:.1f}">'
                 f'{w} <tspan class="tick">{c}</tspan></text>')

    s.append(f'<text class="eyebrow" x="{L}" y="16">WORDS IN MY COMMIT MESSAGES, BY RANK</text>')
    s.append(f'<text class="cap" x="{L}" y="{H-16}">'
             f'{sum(counts.values()):,} words · {len(ranked):,} unique · '
             f'{n_commits:,} commits · fitted exponent {alpha:.2f} '
             f'(dashed: exactly 1/rank)</text>')
    s.append("</svg>")
    return "\n".join(s), alpha


# ---------------------------------------------------------------- demo data

def demo_messages():
    """Synthetic commits so the svg renders before the first workflow run.

    Tokens are sampled from a Zipf distribution over a realistic commit
    vocabulary, then chunked into messages. The shape is right; the sentences
    are nonsense, and the first workflow run overwrites all of it.
    """
    rng = random.Random(11)
    head = ("the to a in for of and on is it not when with from that now use if so we "
            "this but be are all up out more into back just also only make get set do "
            "no longer was should does had has can will its their there here").split()
    body = ("fix add update remove refactor clean handle bump rename move split cache "
            "skip guard log parser tokenizer lexer corpus token embedding vocab model "
            "eval dataset batch loader index worker queue retry timeout config schema "
            "migration endpoint handler router middleware session cookie auth header "
            "payload response request socket stream buffer pointer allocation leak race "
            "deadlock mutex thread async await promise callback hook state props render "
            "layout style theme dark mode mobile viewport focus keyboard label form "
            "input validation error message trace metric dashboard alert threshold "
            "latency throughput memory disk docker image build pipeline workflow action "
            "secret flag arg script test fixture mock stub snapshot coverage lint format "
            "typing annotation docstring readme changelog license version release tag "
            "branch conflict patch diff blame broken flaky stale missing duplicate wrong "
            "slow empty null undefined edge case regression hotfix workaround temporary "
            "proper correct simplify extract inline delete restore catch raise throw "
            "ignore disable enable toggle default fallback override wrap parse serialize "
            "encode decode normalize strip trim escape sanitize actually finally again "
            "properly instead everywhere somewhere obviously apparently").split()

    vocab = head + body
    for w in body:                       # morphological variants make the tail
        if len(w) > 4 and rng.random() < .7:
            vocab += [w + "s", w + "ing" if not w.endswith("e") else w[:-1] + "ing",
                      w + "d" if w.endswith("e") else w + "ed"]
    weights = [1 / (i + 1) ** 1.02 for i in range(len(vocab))]

    tokens = rng.choices(vocab, weights=weights, k=11000)
    out, i, n = {}, 0, 0
    while i < len(tokens):
        k = rng.randint(4, 11)
        out[f"demo{n:04d}"] = " ".join(tokens[i:i + k])
        i, n = i + k, n + 1
    return out


# ---------------------------------------------------------------- main

def main():
    demo = "--demo" in sys.argv
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    if demo:
        print("rendering from synthetic data")
        messages, state = demo_messages(), {}
    else:
        print(f"fetching commits by {USER}")
        messages = fetch_messages()
        if not messages and not state:
            print("no commits found and no saved corpus — nothing to draw", file=sys.stderr)
            return 1

    counts, seen, added = tally(messages, state)
    n_commits = state.get("commits", 0) + added
    print(f"{added} new commits, {sum(counts.values()):,} words, {len(counts):,} unique")

    svg, alpha = render(counts, n_commits)
    OUT.write_text(svg)
    if not demo:
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({
            "commits": n_commits,
            "shas": sorted(seen),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }, indent=0))
    print(f"wrote {OUT} · exponent {alpha:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
