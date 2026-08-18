#!/usr/bin/env python3
"""Rank the words in my commit messages and plot them against Zipf's law.

Run by .github/workflows/zipf.yml every morning. Counts accumulate in
data/counts.json, so the corpus grows with every push instead of resetting.

  python scripts/zipf.py          # fetch from the GitHub API
  python scripts/zipf.py --demo   # render from synthetic data, no network
"""

import collections
import fnmatch
import hashlib
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
# ZIPF_TOKEN is a PAT that can see private repos; GITHUB_TOKEN sees public only
PAT = os.environ.get("ZIPF_TOKEN", "")
TOKEN = PAT or os.environ.get("GITHUB_TOKEN", "")
# repos to leave out entirely, e.g. "work-*,client/secret-thing"
EXCLUDE = [p.strip() for p in os.environ.get("ZIPF_EXCLUDE", "").split(",") if p.strip()]
# a word is only ever shown in plaintext once it is this common; rarer words
# are stored as hashes, so private codenames never land in a public file
LABEL_MIN = int(os.environ.get("ZIPF_LABEL_MIN", "4"))
# also read non-default branches — costs one extra call per repo
ALL_BRANCHES = os.environ.get("ZIPF_ALL_BRANCHES", "1") != "0"
# repos to read even if enumeration misses them, e.g. "HarkerDev/parking,Org/thing"
EXTRA_REPOS = [r.strip() for r in os.environ.get("ZIPF_REPOS", "").split(",") if r.strip()]
MAX_BRANCHES = 20
MAX_PAGES = 10          # 1000 commits per repo, and search caps there too
MAX_POINTS = 1200       # keep the svg small
INK_LIGHT, INK_DARK = "#1f2328", "#e6edf3"

WORD = re.compile(r"[a-z][a-z'-]*[a-z]")
TRAILER = re.compile(r"^(co-authored-by|signed-off-by|reviewed-by|reported-by|"
                     r"cc|refs?|closes|fixes #)\b", re.I)
NOISE = re.compile(r"https?://\S+|`[^`]*`|\b[0-9a-f]{7,40}\b|\S*[/_@]\S*")
SKIP_MSG = re.compile(r"^(merge|revert|bump|initial commit)\b", re.I)


# ---------------------------------------------------------------- fetching

def excluded(full_name):
    name = full_name.split("/")[-1]
    return any(fnmatch.fnmatch(full_name, p) or fnmatch.fnmatch(name, p)
               for p in EXCLUDE)


def key(word):
    return hashlib.blake2s(word.encode(), digest_size=6).hexdigest()


def api(path, **params):
    """GET one page of the REST API. Returns (payload, ok)."""
    url = f"https://api.github.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"{USER}-zipf",
        **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r), True
    except urllib.error.HTTPError as e:
        if e.code not in (403, 404, 409):      # 409 = empty repo
            print(f"  {path}: HTTP {e.code}", file=sys.stderr)
        return None, False
    except Exception as e:                      # noqa: BLE001 - network is best effort
        print(f"  {path}: {e}", file=sys.stderr)
        return None, False


def paged(path, cap=10, **params):
    """Yield items across pages until exhausted or cap pages read."""
    for page in range(1, cap + 1):
        data, ok = api(path, per_page=100, page=page, **params)
        if not ok or not data:
            return
        items = data.get("items", data) if isinstance(data, dict) else data
        yield from items
        if len(items) < 100:
            return


def list_repos():
    """Every repo we can see: personal, collaborator, and org-member."""
    repos, seen = [], set()

    def take(items):
        for r in items:
            full = r.get("full_name", "")
            if full and full not in seen and not r.get("fork"):
                seen.add(full)
                repos.append(full)

    if PAT:
        # organization_member is the one that catches org repos you did not
        # create and were never added to individually
        take(paged("/user/repos", sort="pushed",
                   affiliation="owner,collaborator,organization_member"))
        # belt and braces: walk the orgs directly, since affiliation can miss
        # repos granted through a team rather than to you personally
        for org in paged("/user/orgs", cap=2):
            login = org.get("login")
            if login:
                take(paged(f"/orgs/{login}/repos", type="all", sort="pushed"))
    else:
        take(paged(f"/users/{USER}/repos", sort="pushed"))

    # anything the API still will not surface can be named outright
    take({"full_name": r} for r in EXTRA_REPOS)

    return [r for r in repos if not excluded(r)]


def fetch_via_repos():
    """Walk repos and read their commit lists. Fresh, unlike the search index."""
    out = {}
    repos = list_repos()
    print(f"  {len(repos)} repos visible")
    unreachable = []
    for full in repos:
        before = len(out)
        # the default branch, plus any other branches (commits only live on
        # the default branch once merged, and plenty never get merged)
        refs = [None]
        if ALL_BRANCHES:
            refs += [b["name"] for b in paged(f"/repos/{full}/branches", cap=2)][:MAX_BRANCHES]
        for ref in refs:
            extra = {"sha": ref} if ref else {}
            for c in paged(f"/repos/{full}/commits", cap=MAX_PAGES,
                           author=USER, **extra):
                msg = (c.get("commit") or {}).get("message")
                if msg and c.get("sha"):
                    out[c["sha"]] = msg
        if len(out) > before:
            print(f"    {full}: {len(out) - before}")
        elif full in EXTRA_REPOS:
            unreachable.append(full)
    if unreachable:
        print(f"  no commits readable in: {', '.join(unreachable)} "
              f"(token lacks access, or none authored by {USER})", file=sys.stderr)
    return out


def fetch_via_search():
    """The old path. Kept as a net for commits in repos we do not own."""
    out = {}
    for it in paged("/search/commits", cap=MAX_PAGES,
                    q=f"author:{USER}", sort="author-date", order="desc"):
        repo = (it.get("repository") or {}).get("full_name", "")
        if not excluded(repo):
            out[it["sha"]] = it["commit"]["message"]
    return out


def fetch_messages():
    """Union of both sources, so neither one's blind spots cost us commits."""
    by_repo = fetch_via_repos()
    print(f"  {len(by_repo):,} commits from repo listings")
    by_search = fetch_via_search()
    extra = len(set(by_search) - set(by_repo))
    print(f"  {len(by_search):,} from search index ({extra:,} it alone had)")
    return {**by_search, **by_repo}


# ---------------------------------------------------------------- counting

def tokenize(message):
    lines = [l for l in message.splitlines() if not TRAILER.match(l.strip())]
    text = NOISE.sub(" ", " ".join(lines).lower())
    return WORD.findall(text)


def tally(messages, state):
    if state.get("version") == 2:
        counts = collections.Counter(state.get("counts", {}))
        labels = dict(state.get("labels", {}))
    else:  # migrate a v1 plaintext file
        old = state.get("counts", {})
        counts = collections.Counter({key(w): c for w, c in old.items()})
        labels = {key(w): w for w, c in old.items() if c >= LABEL_MIN}

    seen = set(state.get("shas", []))
    added = skipped = 0
    for sha, msg in messages.items():
        if sha in seen:
            continue
        if SKIP_MSG.match(msg.strip()):
            skipped += 1
            continue
        seen.add(sha)
        words = tokenize(msg)
        counts.update(key(w) for w in words)
        for w in set(words):                    # promote once it is common
            if counts[key(w)] >= LABEL_MIN:
                labels[key(w)] = w
        added += 1
    return counts, labels, seen, added, skipped


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

def render(counts, labels, n_commits):
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    freqs = [c for _, c in ranked]
    alpha = fit_slope(range(1, len(freqs) + 1), freqs)

    W, H = 600, 300
    L, R, T, B = 52, 22, 28, 22
    pw, ph = W - L - R, H - T - B
    # axes run to the data, not to the next decade, so nothing sits empty
    xmax = math.log10(len(ranked)) + .04
    ymax = math.log10(freqs[0]) + .07

    def px(rank):
        return L + math.log10(rank) / xmax * pw

    def py(freq):
        return T + ph - math.log10(freq) / ymax * ph

    def along(x1, y1, x2, y2, text, cls):
        """Set a label on its own line, rotated to match and nudged above it."""
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        rad = math.radians(ang)
        ax = x1 + (x2 - x1) * .98 + math.sin(rad) * 7
        ay = y1 + (y2 - y1) * .98 - math.cos(rad) * 7
        return (f'<text class="{cls}" x="{ax:.1f}" y="{ay:.1f}" text-anchor="end" '
                f'transform="rotate({ang:.2f} {ax:.1f} {ay:.1f})">{text}</text>')

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" aria-label="Log-log plot of word '
         f'frequency against rank in my commit messages, fitted exponent '
         f'{alpha:.2f} against a 1/rank reference line">']

    s.append(f"""<style>
text{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:{INK_LIGHT}}}
.grid{{stroke:{INK_LIGHT};stroke-width:1;opacity:.12}}
.dot{{fill:{INK_LIGHT}}}
.ideal{{stroke:{INK_LIGHT};stroke-width:1.25;stroke-dasharray:4 4;fill:none;opacity:.3}}
.fit{{stroke:{INK_LIGHT};stroke-width:1.75;fill:none;opacity:.9}}
.tick{{font-size:10px;opacity:.45}}
.lbl{{font-size:10.5px;opacity:.85}}
.note{{font-size:10px;opacity:.4}}
.note-fit{{font-size:10.5px;opacity:.75}}
@media (prefers-color-scheme:dark){{
text{{fill:{INK_DARK}}}.grid,.ideal,.fit{{stroke:{INK_DARK}}}.dot{{fill:{INK_DARK}}}}}
.dot{{opacity:0;animation:rise .5s ease-out forwards}}
.once{{animation-name:riseOnce}}
@keyframes rise{{from{{opacity:0;transform:translate(0,7px)}}to{{opacity:.55;transform:translate(0,0)}}}}
@keyframes riseOnce{{from{{opacity:0;transform:translate(0,7px)}}to{{opacity:.2;transform:translate(0,0)}}}}
{''.join(f'.b{i}{{animation-delay:{i*0.075:.3f}s}}' for i in range(1, 12))}
.ideal,.fit{{stroke-dashoffset:0;animation:draw 1.1s .55s ease-out backwards}}
@keyframes draw{{from{{stroke-dashoffset:{pw*1.6:.0f}}}to{{stroke-dashoffset:0}}}}
.fit{{stroke-dasharray:{pw*1.6:.0f}}}
@media (prefers-reduced-motion:reduce){{
.dot{{opacity:.55;animation:none}}.once{{opacity:.2}}
.ideal,.fit{{animation:none}}.fit{{stroke-dasharray:none}}}}
</style>""")

    # decade grid: vertical lines keep the log rhythm legible without labels,
    # only the y axis is numbered
    for d in range(int(xmax) + 1):
        x = L + d / xmax * pw
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
    for d in range(int(ymax) + 1):
        y = T + ph - d / ymax * ph
        s.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">'
                 f'{10**d:,}</text>')

    # the two lines: pure 1/rank, and the actual fit, each labelled in place
    top, last = freqs[0], len(ranked)

    def endpoint(exponent):
        """Stop a line where it hits a count of 1 rather than piling both
        lines into the same bottom corner on a short corpus."""
        if exponent <= 0:
            return px(last), py(max(top * last ** -exponent, 1))
        r = min(float(last), max(top ** (1 / exponent), 1.0))
        return px(r), py(max(top * r ** -exponent, 1))

    x1, y1 = px(1), py(top)
    ix, iy = endpoint(1.0)
    fx, fy = endpoint(alpha)
    s.append(f'<path class="ideal" d="M{x1:.1f} {y1:.1f} L{ix:.1f} {iy:.1f}"/>')
    s.append(f'<path class="fit" d="M{x1:.1f} {y1:.1f} L{fx:.1f} {fy:.1f}"/>')
    s.append(along(x1, y1, ix, iy, "1/rank", "note"))
    s.append(along(x1, y1, fx, fy, f"&#945; = {alpha:.2f}", "note-fit"))

    # points, bucketed into 12 bands so the stagger costs 11 css rules.
    # words used exactly once are half the vocabulary and the least interesting
    # half, so they sit back and let the head of the distribution read first
    pts = ranked[:MAX_POINTS]
    for i, (_, c) in enumerate(pts):
        band = min(11, int(12 * i / len(pts)))
        once = " once" if c == 1 else ""
        s.append(f'<circle class="dot{once} b{band}" cx="{px(i+1):.1f}" '
                 f'cy="{py(c):.1f}" r="{1.6 if c == 1 else 2.1}"/>')

    for rank in (1, 2, 3, 10, 100, 1000):
        if rank > len(ranked):
            continue
        k, c = ranked[rank - 1]
        w = labels.get(k)
        if not w:
            continue
        s.append(f'<text class="lbl" x="{px(rank)+7:.1f}" y="{py(c)-6:.1f}">'
                 f'{w} <tspan class="tick">{c}</tspan></text>')

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

    counts, labels, seen, added, skipped = tally(messages, state)
    n_commits = state.get("commits", 0) + added
    print(f"{len(messages):,} fetched · {added:,} new to the corpus · "
          f"{skipped:,} merge/revert/bump skipped · "
          f"{len(messages) - added - skipped:,} already counted")
    print(f"corpus now: {n_commits:,} commits · {sum(counts.values()):,} words · "
          f"{len(counts):,} unique · {sum(1 for c in counts.values() if c == 1):,} used once")

    svg, alpha = render(counts, labels, n_commits)
    OUT.write_text(svg)
    if not demo:
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({
            "version": 2,
            "commits": n_commits,
            "shas": sorted(seen),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "labels": labels,
        }, indent=0))
    print(f"wrote {OUT} · exponent {alpha:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
