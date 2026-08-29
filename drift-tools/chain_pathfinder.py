"""
Chain Game - Path Finder
Meaning moves: Datamuse rel_trg + rel_syn + ml, run in parallel.
Letter moves: restricted to the Google 10k most-common-words list.
Stops per pair at 20 valid paths; a pair qualifies with 10+ paths.
"""
import requests, argparse, time, json, os, random, threading
from concurrent.futures import ThreadPoolExecutor
import nltk as _nltk
_nltk.download('names',   quiet=True)
_nltk.download('wordnet', quiet=True)
from nltk.corpus import names as _nltk_names, wordnet as _wn

MIN_WORD_LEN  = 3
MAX_WORD_LEN  = 6
STEPS         = 5
MIN_EACH_TYPE = 2
MAX_PATHS     = 20
MIN_VALID     = 10
TIMEOUT       = 300
BRANCH_CAP    = 3

CANDIDATES = [
    ("green",  "hot"),
    ("stone",  "fire"),
    ("sharp",  "blue"),
    ("grave",  "light"),
    ("blank",  "bold"),
    ("pale",   "fierce"),
    ("still",  "burn"),
    ("keen",   "hollow"),
    ("raw",    "bright"),
    ("bitter", "gold"),
    ("deep",   "loud"),
    ("grey",   "sharp"),
]

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(BASE_DIR)
WORD_LIST_PATH = os.path.join(REPO_ROOT, "words_alpha.txt")
FREQ_LIST_PATH = os.path.join(REPO_ROOT, "google-10000.txt")


def _load_set(path, length_filter=False):
    try:
        with open(path) as f:
            words = (w.strip().lower() for w in f if w.strip())
            if length_filter:
                return set(w for w in words if MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN)
            return set(words)
    except FileNotFoundError:
        print(f"Warning: {path} not found.", flush=True)
        return set()


WORD_SET = _load_set(WORD_LIST_PATH, length_filter=True)
FREQ_SET = _load_set(FREQ_LIST_PATH)

# ── Proper-noun filter ────────────────────────────────────────────────────────
# Build at startup so _datamuse can do a fast set-membership check.
_NLTK_NAMES = frozenset(n.lower() for n in _nltk_names.words()
                         if MIN_WORD_LEN <= len(n) <= MAX_WORD_LEN)

def _is_proper(w):
    syns = _wn.synsets(w)
    # Rule 1: recognised first/last name with no common English meaning
    if w in _NLTK_NAMES and not syns:
        return True
    # Rule 2: not common enough for the 10k list AND WordNet only knows it as a
    # specific entity (instance), or doesn't know it at all → proper/niche term
    if w not in FREQ_SET:
        if not syns:
            return True
        if all(s.instance_hypernyms() for s in syns):
            return True
    return False

PROPER_NOUNS = frozenset(w for w in WORD_SET if _is_proper(w))
print(f"Proper-noun filter: {len(PROPER_NOUNS)} words blocked "
      f"(e.g. {sorted(PROPER_NOUNS)[:6]})", flush=True)

BLOCKLIST = frozenset({
    "vic", "vid", "phot", "peter", "lan", "pts", "gzip",
})

# ── Meaning-move lookup ────────────────────────────────────────────────────────
syn_cache = {}   # word -> set of meaning-move neighbors
_pool     = ThreadPoolExecutor(max_workers=6)


def _datamuse(rel, word):
    try:
        r = requests.get(
            "https://api.datamuse.com/words",
            params={rel: word, "max": 100},
            timeout=8
        )
        if not r.ok:
            return set()
        out = set()
        for item in r.json():
            w = item["word"].lower()
            if (w != word and "_" not in w and "-" not in w
                    and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN
                    and w in WORD_SET and w not in PROPER_NOUNS
                    and w not in BLOCKLIST):
                out.add(w)
        return out
    except Exception:
        return set()


def get_synonyms(word):
    if word in syn_cache:
        return syn_cache[word]
    f_trg = _pool.submit(_datamuse, "rel_trg", word)
    f_syn = _pool.submit(_datamuse, "rel_syn", word)
    f_ml  = _pool.submit(_datamuse, "ml",      word)
    result = f_trg.result() | f_syn.result() | f_ml.result()
    syn_cache[word] = result
    return result


def warm_cache(start, end):
    for w in (start, end):
        if w not in syn_cache:
            print(f"  [warming: {w}]", flush=True)
            get_synonyms(w)


# ── Letter-move lookup ─────────────────────────────────────────────────────────

def letter_neighbors(word):
    """Edit-distance-1 neighbors restricted to the 10k common-word list."""
    neighbors = set()
    alpha = "abcdefghijklmnopqrstuvwxyz"
    for i in range(len(word)):
        for c in alpha:
            if c != word[i]:
                w = word[:i] + c + word[i+1:]
                if w in FREQ_SET and w not in BLOCKLIST:
                    neighbors.add(w)
    for i in range(len(word)):
        w = word[:i] + word[i+1:]
        if MIN_WORD_LEN <= len(w) and w in FREQ_SET and w not in BLOCKLIST:
            neighbors.add(w)
    if len(word) < MAX_WORD_LEN:
        for i in range(len(word) + 1):
            for c in alpha:
                w = word[:i] + c + word[i:]
                if w in FREQ_SET and w not in BLOCKLIST:
                    neighbors.add(w)
    return neighbors


# ── DFS ───────────────────────────────────────────────────────────────────────
_stop_event = threading.Event()


def _sub_dfs(first_word, first_move, start, end, max_sub):
    lc0 = 1 if first_move == "L" else 0
    mc0 = 1 if first_move == "M" else 0
    found = []
    stack = [(first_word, [start, first_word], [first_move], lc0, mc0)]
    while stack and len(found) < max_sub and not _stop_event.is_set():
        current, path, moves, lc, mc = stack.pop()
        depth = len(path) - 1
        if depth == STEPS:
            if current == end and lc >= MIN_EACH_TYPE and mc >= MIN_EACH_TYPE:
                found.append((path[:], moves[:]))
            continue
        remaining = STEPS - depth
        if lc + remaining < MIN_EACH_TYPE: continue
        if mc + remaining < MIN_EACH_TYPE: continue
        visited = set(path)
        syns = list(get_synonyms(current))
        random.shuffle(syns)
        for nbr in syns:
            if nbr in visited: continue
            if depth + 1 == STEPS and nbr != end: continue
            stack.append((nbr, path + [nbr], moves + ["M"], lc, mc + 1))
        lns = list(letter_neighbors(current))
        random.shuffle(lns)
        for nbr in lns:
            if nbr in visited: continue
            if depth + 1 == STEPS and nbr != end: continue
            stack.append((nbr, path + [nbr], moves + ["L"], lc + 1, mc))
    return found


def find_paths(start, end, max_paths=MAX_PATHS, timeout=TIMEOUT):
    WORD_SET.add(start); WORD_SET.add(end)
    FREQ_SET.add(start); FREQ_SET.add(end)

    warm_cache(start, end)

    first_steps = []
    for nbr in get_synonyms(start):
        first_steps.append(("M", nbr))
    for nbr in letter_neighbors(start):
        first_steps.append(("L", nbr))
    random.shuffle(first_steps)

    found = []
    seen_paths = set()

    _stop_event.clear()
    timer = threading.Timer(timeout, _stop_event.set)
    timer.start()
    try:
        for move_type, first_word in first_steps:
            if len(found) >= max_paths or _stop_event.is_set():
                break
            sub = _sub_dfs(first_word, move_type, start, end, BRANCH_CAP)
            for path, moves in sub:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    found.append((path, moves))
                if len(found) >= max_paths:
                    break
    finally:
        timer.cancel()
        if _stop_event.is_set():
            print(f"  [timeout after {timeout}s]", flush=True)
    return found


def run_pair(start, end, max_paths=MAX_PATHS, timeout=TIMEOUT):
    print(f"\n{'='*55}", flush=True)
    print(f"  {start.upper():10} → {end.upper()}", flush=True)
    print(f"{'='*55}", flush=True)
    raw = find_paths(start, end, max_paths=max_paths, timeout=timeout)
    print(f"  DFS raw: {len(raw)} path(s)", flush=True)
    valid = []
    for path, moves in raw:
        lc = moves.count("L"); mc = moves.count("M")
        chain = " → ".join(
            f"{w.upper()}[{m}]" if m else w.upper()
            for w, m in zip(path, [""] + moves))
        print(f"  ✓ {chain}  ({lc}L {mc}M)", flush=True)
        valid.append({"path": path, "moves": moves})
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-paths", type=int, default=MAX_PATHS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    if args.batch or (not args.start and not args.end):
        results = {}
        for start, end in CANDIDATES:
            valid = run_pair(start, end, max_paths=args.max_paths, timeout=args.timeout)
            results[f"{start}_{end}"] = {
                "start": start, "end": end,
                "valid_paths": valid, "count": len(valid)
            }
            time.sleep(0.3)

        out = os.path.join(BASE_DIR, "dfs_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*55}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'='*55}", flush=True)
        for k, v in results.items():
            flag = "✓" if v["count"] >= MIN_VALID else "~" if v["count"] >= 3 else "✗"
            print(f"  {flag} {v['start'].upper():8} → {v['end'].upper():8}  {v['count']} valid",
                  flush=True)
        qualifying = [v for v in results.values() if v["count"] >= MIN_VALID]
        print(f"\nQualifying (>= {MIN_VALID}):", flush=True)
        if qualifying:
            for v in qualifying:
                print(f"  ✓ {v['start'].upper()} → {v['end'].upper()}", flush=True)
        else:
            print("  (none)", flush=True)
    else:
        start = args.start.lower()
        end   = args.end.lower()
        print(f"\nSearching: {start.upper()} → {end.upper()}")
        valid = run_pair(start, end, max_paths=args.max_paths, timeout=args.timeout)
        print(f"\nFound {len(valid)} verified path(s) (need {MIN_VALID} to qualify)")
        for i, p in enumerate(valid, 1):
            chain = " → ".join(f"{w.upper()}[{m}]" if m else w.upper()
                               for w, m in zip(p["path"], [""] + p["moves"]))
            print(f"  {i}. {chain}")


if __name__ == "__main__":
    main()
