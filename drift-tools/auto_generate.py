#!/usr/bin/env python3
"""
Ask Claude for candidate Drift word pairs, filter them against USF cue
coverage and existing puzzle history, then verify survivors with
chain_pathfinder.

Default mode just prints results. Pass --write to also enrich qualifying
pairs with meaning_neighbors and append them to puzzles.json / puzzles_data.js.
"""
import os, sys, csv, json, re, time, argparse
import concurrent.futures
from datetime import date, timedelta

import requests
import anthropic

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
DRIFT_DIR = os.path.dirname(TOOLS_DIR)

sys.path.insert(0, TOOLS_DIR)
import chain_pathfinder as cp  # noqa: E402 — initializes WORD_SET etc.

MODEL          = "claude-haiku-4-5-20251001"
NUM_CANDIDATES = 20
MIN_LEN, MAX_LEN = 3, 6
MAX_PATHS      = 4   # stop searching once this many valid paths are found
MIN_VALID      = 3   # a pair qualifies with at least this many
TIMEOUT        = 180

USF_PATH       = os.path.join(TOOLS_DIR, "usf_associations.csv")
PUZZLES_PATH   = os.path.join(DRIFT_DIR, "puzzles.json")
PUZZLES_JS_PATH = os.path.join(DRIFT_DIR, "puzzles_data.js")
WORDS_FILE     = os.path.join(DRIFT_DIR, "words_alpha.txt")
USF_THRESHOLD  = 0.01

PROMPT = f"""Generate {NUM_CANDIDATES} candidate word pairs for a word-association \
puzzle game called Drift. In Drift, a player bridges a START word to an END word \
through a chain of intermediate words, where each step is either a one-letter \
change (add/remove/substitute a letter) or a meaning jump (synonym/close \
association). Good pairs feel far apart in meaning at first glance but turn out \
to be bridgeable.

Rules for each pair:
- Both words must be common, everyday English words, {MIN_LEN}-{MAX_LEN} letters long.
- Lowercase, single words only, no proper nouns, no plurals-only-because-lazy, no abbreviations.
- The two words in a pair must have good semantic distance — not synonyms, not \
obviously related (avoid pairs like "cat"/"dog" or "hot"/"cold" that are too on-the-nose).
- Prefer concrete, visualizable nouns/adjectives over abstract words.
- All {NUM_CANDIDATES} pairs must be distinct from each other.

Respond with ONLY a JSON object, no prose, no markdown fences, in this exact shape:
{{"pairs": [["word1", "word2"], ...]}}
"""


def generate_candidates():
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(text)
    pairs = []
    seen = set()
    for item in data.get("pairs", []):
        if len(item) != 2:
            continue
        a, b = item[0].strip().lower(), item[1].strip().lower()
        if not (a.isalpha() and b.isalpha()):
            continue
        if not (MIN_LEN <= len(a) <= MAX_LEN and MIN_LEN <= len(b) <= MAX_LEN):
            continue
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def load_usf_cues():
    cues = set()
    with open(USF_PATH) as f:
        for row in csv.reader(f):
            if row:
                cues.add(row[0].strip().lower())
    return cues


def load_used_words():
    with open(PUZZLES_PATH) as f:
        puzzles = json.load(f)
    used = set()
    for p in puzzles.values():
        used.add(p["start"].lower())
        used.add(p["end"].lower())
    return used


# ── Enrichment (meaning_neighbors), same logic as run_new_pairs.py ─────────

WORD_SET_ENRICH = frozenset(
    w.strip().lower() for w in open(WORDS_FILE)
    if MIN_LEN <= len(w.strip()) <= MAX_LEN
)


def load_usf_index():
    index = {}
    with open(USF_PATH) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            cue, target = row[0].lower(), row[1].lower()
            try:
                fs = float(row[2])
            except ValueError:
                continue
            for a, b in ((cue, target), (target, cue)):
                index.setdefault(a, {})[b] = max(index.get(a, {}).get(b, 0), fs)
    return index


def usf_neighbors(word, usf_index):
    return {
        n for n, s in usf_index.get(word, {}).items()
        if s >= USF_THRESHOLD and MIN_LEN <= len(n) <= MAX_LEN
        and n in WORD_SET_ENRICH and n not in cp.BLOCKLIST
    }


def datamuse_syn(word, session):
    try:
        r = session.get("https://api.datamuse.com/words",
                         params={"rel_syn": word, "max": 100}, timeout=8)
        if not r.ok:
            return set()
        return {
            item["word"].lower() for item in r.json()
            if (MIN_LEN <= len(item["word"]) <= MAX_LEN
                and "_" not in item["word"] and "-" not in item["word"]
                and item["word"].lower() in WORD_SET_ENRICH
                and item["word"].lower() not in cp.BLOCKLIST
                and item["word"].lower() != word)
        }
    except Exception:
        return set()


def get_neighbors(word, session, usf_index):
    return sorted(usf_neighbors(word, usf_index) | datamuse_syn(word, session))


def m_sources(valid_paths):
    words = set()
    for entry in valid_paths:
        path, moves = entry["path"], entry["moves"]
        for i, move in enumerate(moves):
            if move == "M":
                w = path[i].lower()
                if w not in cp.BLOCKLIST and MIN_LEN <= len(w) <= MAX_LEN and w.isalpha():
                    words.add(w)
    return words


def write_puzzles(qualifying):
    usf_index = load_usf_index()

    with open(PUZZLES_PATH) as f:
        puzzles = json.load(f)

    last_date_str = max(puzzles.keys())
    last_id       = max(p["id"] for p in puzzles.values())
    next_date     = date.fromisoformat(last_date_str) + timedelta(days=1)
    next_id       = last_id + 1

    print(f"\nLast existing: {last_date_str}  id={last_id}", flush=True)
    print(f"New entries starting: {next_date}  id={next_id}", flush=True)

    session = requests.Session()
    new_entries = {}
    assigned = []

    for q in qualifying:
        start, end = q["start"], q["end"]
        date_key   = str(next_date)
        print(f"\nEnriching {start.upper()} -> {end.upper()} ({date_key}) ...", flush=True)

        phase1_words = {start, end}
        phase1_nbrs  = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(get_neighbors, w, session, usf_index): w for w in sorted(phase1_words)}
            for fut in concurrent.futures.as_completed(futs):
                w = futs[fut]; phase1_nbrs[w] = fut.result()
                print(f"  {w}: {len(phase1_nbrs[w])} neighbors", flush=True)

        phase2_words = set(phase1_nbrs.get(start, [])) | m_sources(q["valid_paths"])
        phase2_words -= phase1_words
        phase2_nbrs  = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(get_neighbors, w, session, usf_index): w for w in sorted(phase2_words)}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                w = futs[fut]; phase2_nbrs[w] = fut.result()
                done += 1
                if done % 10 == 0 or done == len(phase2_words):
                    print(f"  phase2: {done}/{len(phase2_words)}", flush=True)

        all_nbrs = {**phase1_nbrs, **phase2_nbrs}
        words_needed = {start, end} | set(phase1_nbrs.get(start, [])) | m_sources(q["valid_paths"])
        meaning_neighbors = {w: all_nbrs[w] for w in sorted(words_needed) if w in all_nbrs}

        new_entries[date_key] = {
            "id":                next_id,
            "start":             start,
            "end":               end,
            "meaning_neighbors": meaning_neighbors,
        }
        assigned.append((date_key, next_id, start, end))
        next_date += timedelta(days=1)
        next_id   += 1
        time.sleep(0.2)

    puzzles.update(new_entries)

    with open(PUZZLES_PATH, "w") as f:
        json.dump(puzzles, f, indent=2)

    with open(PUZZLES_JS_PATH, "w") as f:
        f.write("window.DRIFT_PUZZLES=")
        json.dump(puzzles, f, indent=2)
        f.write(";")

    print(f"\n{'='*55}", flush=True)
    print("WROTE PUZZLES:", flush=True)
    for date_key, pid, start, end in assigned:
        print(f"  {date_key}  #{pid}  {start.upper()} -> {end.upper()}", flush=True)
    print(f"\nWrote {PUZZLES_PATH}", flush=True)
    print(f"Wrote {PUZZLES_JS_PATH}", flush=True)
    return assigned


def run_pair(start, end):
    print(f"\n{'='*55}", flush=True)
    print(f"  {start.upper():10} -> {end.upper()}", flush=True)
    print(f"{'='*55}", flush=True)
    raw = cp.find_paths(start, end, max_paths=MAX_PATHS, timeout=TIMEOUT)
    valid = []
    for path, moves in raw:
        lc = moves.count("L"); mc = moves.count("M")
        chain = " -> ".join(
            f"{w.upper()}[{m}]" if m else w.upper()
            for w, m in zip(path, [""] + moves))
        print(f"  ✓ {chain}  ({lc}L {mc}M)", flush=True)
        valid.append({"path": path, "moves": moves})
    flag = "QUALIFIES" if len(valid) >= MIN_VALID else "skip"
    print(f"  -> {len(valid)} valid path(s)  ({flag})", flush=True)
    return valid


def run_pipeline(write=False):
    print(f"Requesting {NUM_CANDIDATES} candidate pairs from {MODEL} ...", flush=True)
    candidates = generate_candidates()
    print(f"Got {len(candidates)} candidate pair(s):", flush=True)
    for a, b in candidates:
        print(f"  {a} / {b}", flush=True)

    usf_cues  = load_usf_cues()
    used_words = load_used_words()

    survivors = []
    for a, b in candidates:
        if a not in usf_cues or b not in usf_cues:
            missing = [w for w in (a, b) if w not in usf_cues]
            print(f"\nskip {a}/{b}: not a USF cue ({', '.join(missing)})", flush=True)
            continue
        if a in used_words or b in used_words:
            dup = [w for w in (a, b) if w in used_words]
            print(f"\nskip {a}/{b}: already used in puzzles.json ({', '.join(dup)})", flush=True)
            continue
        survivors.append((a, b))

    print(f"\n{len(survivors)} pair(s) survive filtering, running pathfinder ...", flush=True)

    qualifying = []
    for start, end in survivors:
        valid = run_pair(start, end)
        if len(valid) >= MIN_VALID:
            qualifying.append({"start": start, "end": end, "valid_paths": valid})
        time.sleep(0.3)

    print(f"\n{'='*55}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*55}", flush=True)
    if qualifying:
        for q in qualifying:
            print(f"  ✓ {q['start'].upper()} -> {q['end'].upper()}  "
                  f"({len(q['valid_paths'])} paths)", flush=True)
    else:
        print("  (none qualified)", flush=True)

    if write:
        if qualifying:
            write_puzzles(qualifying)
        else:
            print("\nNo qualifying pairs to write.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                         help="also enrich qualifying pairs and append them to puzzles.json / puzzles_data.js")
    args = parser.parse_args()

    run_pipeline(write=args.write)


if __name__ == "__main__":
    main()
