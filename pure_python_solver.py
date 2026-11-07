# Slang-puzzel CoVer-Almanak 2003-2004 — oplosser zonder externe libraries
#
# Aanpak: de puzzel wordt vertaald naar logische clausules (SAT) en opgelost
# met een eigen, kleine CDCL-SAT-solver in puur Python.
#   * variabele per cel: zit er een blokje (aan) of niet
#   * elke rij uit de drie aanzichten: precies k van de 10 cellen aan
#   * de slang raakt zichzelf niet: een aan-cel heeft precies 2 aan-buren,
#     begin- en eindpunt precies 1
#   * losse ringen die dan nog kunnen ontstaan worden achteraf verboden
#     met een extra clausule, waarna opnieuw wordt opgelost.
import heapq
import itertools
import time

N, LEN = 10, 351
START, END = (0, 0, 0), (6, 3, 9)

# Bovenste rij = coördinaat 9, linkse kolom = 0.
# Bovenaanzicht: rijen y=9..0, kolommen x=0..9, telt langs z
TOP = [
    [4,5,3,4,5,3,1,3,5,6],
    [3,5,3,5,5,4,5,3,1,6],
    [2,3,1,3,1,4,3,2,2,4],
    [7,1,4,5,6,4,4,5,4,4],
    [1,4,4,4,2,4,6,1,3,4],
    [3,7,2,3,6,5,0,5,4,2],
    [4,3,0,1,1,2,3,3,3,4],
    [3,3,3,5,5,5,4,5,2,3],
    [2,1,3,2,3,5,2,1,4,5],
    [7,3,6,5,4,4,4,0,4,6],
]
# Vooraanzicht: rijen z=9..0, kolommen y=0..9, telt langs x
FRONT = [
    [5,2,4,3,5,5,6,3,6,3],
    [4,4,5,5,3,2,2,1,3,5],
    [3,1,4,1,4,5,5,5,4,4],
    [4,3,3,3,2,1,4,3,3,3],
    [5,5,1,1,4,3,5,2,3,4],
    [4,2,2,0,5,1,4,2,4,6],
    [3,3,6,2,4,2,6,4,3,2],
    [7,2,4,2,2,5,4,2,6,3],
    [4,1,5,3,4,4,1,1,2,5],
    [4,5,4,4,4,5,7,2,6,4],
]
# Zijaanzicht: rijen z=9..0, kolommen x=0..9, telt langs y
SIDE = [
    [0,5,5,5,6,3,3,3,6,6],
    [2,4,2,4,4,4,5,3,2,4],
    [6,2,4,3,6,4,3,2,2,4],
    [5,1,0,4,5,4,1,5,1,3],
    [3,5,4,4,3,3,0,2,4,5],
    [2,3,4,1,2,6,6,2,3,1],
    [5,3,3,5,3,4,1,4,2,5],
    [4,2,3,3,2,5,3,4,5,6],
    [3,4,2,5,2,2,5,1,3,3],
    [6,6,2,3,5,5,5,2,4,7],
]
R = range(N)
top   = lambda x, y: TOP[9 - y][x]
front = lambda y, z: FRONT[9 - z][y]
side  = lambda x, z: SIDE[9 - z][x]


def nbrs(c):
    for d in range(3):
        for s in (-1, 1):
            n = list(c); n[d] += s
            if 0 <= n[d] < N:
                yield tuple(n)


# ---------------------------------------------------------------- SAT-solver
class Solver:
    """Kleine CDCL-solver: twee bewaakte literalen, 1UIP-leren, VSIDS,
    fase-onthouden, Luby-herstarts en opruimen van geleerde clausules.
    Literaal: 2*var voor 'waar', 2*var+1 voor 'onwaar' (var vanaf 0)."""

    def __init__(self, nvars):
        self.n = nvars
        self.clauses = []            # lijsten van literalen
        self.learnt = set()          # indices van geleerde clausules
        self.lbd = {}
        self.watches = [[] for _ in range(2 * nvars)]
        self.val = [-1] * nvars      # -1 onbekend, 0 onwaar, 1 waar
        self.level = [0] * nvars
        self.reason = [None] * nvars
        self.trail, self.lim = [], []
        self.qhead = 0
        self.act = [0.0] * nvars
        self.inc = 1.0
        self.phase = [0] * nvars     # meeste cellen zijn leeg
        self.heap = [(0.0, v) for v in range(nvars)]
        self.ok = True
        self.conflicts = 0

    # waarde van een literaal: 1 waar, 0 onwaar, -1 onbekend
    def lv(self, l):
        v = self.val[l >> 1]
        return v if v < 0 else v ^ (l & 1)

    def add_clause(self, lits, learnt=False):
        lits = list(dict.fromkeys(lits))
        if not learnt:
            # op niveau 0: vervulde clausules overslaan, onware literalen weg
            if any(self.lv(l) == 1 and self.level[l >> 1] == 0 for l in lits):
                return
            lits = [l for l in lits if not (self.lv(l) == 0 and self.level[l >> 1] == 0)]
        if not lits:
            self.ok = False
            return
        if len(lits) == 1:
            if self.lv(lits[0]) == 0:
                self.ok = False
            elif self.lv(lits[0]) == -1:
                self.assign(lits[0], None)
            return
        ci = len(self.clauses)
        self.clauses.append(lits)
        self.watches[lits[0]].append(ci)
        self.watches[lits[1]].append(ci)
        if learnt:
            self.learnt.add(ci)
        return ci

    def assign(self, l, reason):
        v = l >> 1
        self.val[v] = 1 - (l & 1)
        self.level[v] = len(self.lim)
        self.reason[v] = reason
        self.trail.append(l)

    def propagate(self):
        clauses, watches, val = self.clauses, self.watches, self.val
        while self.qhead < len(self.trail):
            p = self.trail[self.qhead]; self.qhead += 1
            fl = p ^ 1                       # deze literaal is nu onwaar
            ws = watches[fl]
            i = j = 0
            while i < len(ws):
                ci = ws[i]; i += 1
                c = clauses[ci]
                if c is None:                # opgeruimde clausule
                    continue
                if c[0] == fl:
                    c[0], c[1] = c[1], c[0]
                o = c[0]
                vo = val[o >> 1]
                if vo >= 0 and vo ^ (o & 1) == 1:
                    ws[j] = ci; j += 1
                    continue
                for k in range(2, len(c)):
                    lk = c[k]; vk = val[lk >> 1]
                    if vk < 0 or vk ^ (lk & 1) == 1:
                        c[1], c[k] = lk, fl
                        watches[lk].append(ci)
                        break
                else:
                    ws[j] = ci; j += 1
                    if vo < 0:
                        self.assign(o, ci)
                    else:                    # conflict
                        while i < len(ws):
                            ws[j] = ws[i]; j += 1; i += 1
                        del ws[j:]
                        self.qhead = len(self.trail)
                        return ci
            del ws[j:]
        return None

    def bump(self, v):
        self.act[v] += self.inc
        if self.act[v] > 1e100:
            self.act = [a * 1e-100 for a in self.act]
            self.inc *= 1e-100
            self.heap = [(-self.act[u], u) for u in range(self.n) if self.val[u] < 0]
            heapq.heapify(self.heap)
        if self.val[v] < 0:
            heapq.heappush(self.heap, (-self.act[v], v))

    def analyze(self, ci):
        seen = [False] * self.n
        learnt = [None]
        cur = len(self.lim)
        count = 0
        p = None
        idx = len(self.trail) - 1
        while True:
            for q in self.clauses[ci]:
                if p is not None and q == p:
                    continue
                v = q >> 1
                if not seen[v] and self.level[v] > 0:
                    seen[v] = True
                    self.bump(v)
                    if self.level[v] == cur:
                        count += 1
                    else:
                        learnt.append(q)
            while not seen[self.trail[idx] >> 1]:
                idx -= 1
            p = self.trail[idx]; idx -= 1
            seen[p >> 1] = False
            count -= 1
            if count == 0:
                break
            ci = self.reason[p >> 1]
        learnt[0] = p ^ 1
        # eenvoudige minimalisatie: literalen waarvan de reden al gedekt is
        keep = [learnt[0]]
        for q in learnt[1:]:
            r = self.reason[q >> 1]
            if r is None or any(not seen[x >> 1] and self.level[x >> 1] > 0
                                for x in self.clauses[r] if x != q ^ 1):
                keep.append(q)
        learnt = keep
        if len(learnt) == 1:
            bt = 0
        else:
            mi = max(range(1, len(learnt)), key=lambda k: self.level[learnt[k] >> 1])
            learnt[1], learnt[mi] = learnt[mi], learnt[1]
            bt = self.level[learnt[1] >> 1]
        lbd = len({self.level[q >> 1] for q in learnt})
        self.inc /= 0.95
        return learnt, bt, lbd

    def backtrack(self, lvl):
        if len(self.lim) <= lvl:
            return
        for k in range(len(self.trail) - 1, self.lim[lvl] - 1, -1):
            v = self.trail[k] >> 1
            self.phase[v] = self.val[v]
            self.val[v] = -1
            self.reason[v] = None
            heapq.heappush(self.heap, (-self.act[v], v))
        del self.trail[self.lim[lvl]:]
        del self.lim[lvl:]
        self.qhead = len(self.trail)

    def decide(self):
        while self.heap:
            a, v = heapq.heappop(self.heap)
            if self.val[v] < 0 and -a == self.act[v]:
                return v
        for v in range(self.n):              # vangnet
            if self.val[v] < 0:
                return v
        return None

    def reduce_db(self):
        locked = {self.reason[l >> 1] for l in self.trail}
        cand = sorted((ci for ci in self.learnt if ci not in locked and self.lbd[ci] > 2),
                      key=lambda ci: -self.lbd[ci])
        for ci in cand[:len(cand) // 2]:
            self.clauses[ci] = None
            self.learnt.discard(ci)
            del self.lbd[ci]

    @staticmethod
    def luby(i):
        k = 1
        while (1 << k) - 1 < i + 1:
            k += 1
        while (1 << k) - 1 != i + 1:
            i -= (1 << (k - 1)) - 1
            k = 1
            while (1 << k) - 1 < i + 1:
                k += 1
        return 1 << (k - 1)

    def solve(self):
        if not self.ok:
            return False
        self.backtrack(0)
        if self.propagate() is not None:
            return False
        restart = 0
        next_reduce = 4000
        while True:
            limit = 100 * self.luby(restart); restart += 1
            local = 0
            while True:
                ci = self.propagate()
                if ci is not None:
                    self.conflicts += 1; local += 1
                    if not self.lim:
                        return False
                    learnt, bt, lbd = self.analyze(ci)
                    self.backtrack(bt)
                    if len(learnt) == 1:
                        self.assign(learnt[0], None)
                    else:
                        nci = self.add_clause(learnt, learnt=True)
                        self.lbd[nci] = lbd
                        self.assign(learnt[0], nci)
                    if self.conflicts % 5000 == 0:
                        print(f"  {self.conflicts:,} conflicten", flush=True)
                    continue
                if local >= limit:
                    self.backtrack(0)
                    break
                if len(self.learnt) > next_reduce:
                    self.reduce_db(); next_reduce += 1000
                v = self.decide()
                if v is None:
                    return True
                self.lim.append(len(self.trail))
                self.assign(2 * v + (1 - self.phase[v]), None)


# ---------------------------------------------------------------- het model
def solve():
    cells = [(x, y, z) for x in R for y in R for z in R]
    var = {c: i for i, c in enumerate(cells)}
    on = lambda c: 2 * var[c]          # literaal "cel c is aan"
    off = lambda c: 2 * var[c] + 1
    s = Solver(len(cells))

    def exactly(group, k):
        n = len(group)
        for sub in itertools.combinations(group, k + 1):      # hoogstens k
            s.add_clause([off(c) for c in sub])
        for sub in itertools.combinations(group, n - k + 1):  # minstens k
            s.add_clause([on(c) for c in sub])

    for x in R:
        for y in R: exactly([(x, y, z) for z in R], top(x, y))
    for y in R:
        for z in R: exactly([(x, y, z) for x in R], front(y, z))
    for x in R:
        for z in R: exactly([(x, y, z) for y in R], side(x, z))

    s.add_clause([on(START)]); s.add_clause([on(END)])
    for c in cells:
        nb = list(nbrs(c))
        d = 1 if c in (START, END) else 2
        for sub in itertools.combinations(nb, d + 1):          # hoogstens d buren
            s.add_clause([off(c)] + [off(n) for n in sub])
        for sub in itertools.combinations(nb, len(nb) - d + 1):  # minstens d
            s.add_clause([off(c)] + [on(n) for n in sub])

    t = time.time()
    rondes = 0
    while True:
        rondes += 1
        if not s.solve():
            print("Geen oplossing.")
            return None
        aan = {c for c in cells if s.val[var[c]] == 1}
        path, prev = [START], None
        while path[-1] != END:
            nxt = next(n for n in nbrs(path[-1]) if n in aan and n != prev)
            prev = path[-1]; path.append(nxt)
        if len(path) == LEN:
            print(f"Opgelost in {time.time() - t:.0f} s, {s.conflicts:,} conflicten, "
                  f"{rondes} ronde(s).")
            return path
        # losse ringen verbieden en verder zoeken
        rest = aan - set(path)
        while rest:
            comp, stack = set(), [rest.pop()]
            while stack:
                c = stack.pop(); comp.add(c)
                for n in nbrs(c):
                    if n in rest: rest.discard(n); stack.append(n)
            s.backtrack(0)
            s.add_clause([off(c) for c in comp])


def print_sums(path):
    """Print de drie aanzichten berekend uit een oplossing; afwijkingen krijgen *."""
    P = set(map(tuple, path))
    views = (
        ("bovenaanzicht (rij y, kolom x, telt langs z)", top,
         lambda r, k: sum((k, r, z) in P for z in R)),
        ("vooraanzicht (rij z, kolom y, telt langs x)", front,
         lambda r, k: sum((x, k, r) in P for x in R)),
        ("zijaanzicht (rij z, kolom x, telt langs y)", side,
         lambda r, k: sum((k, y, r) in P for y in R)),
    )
    fouten = 0
    for naam, gegeven, tel in views:
        print(naam)
        for r in reversed(R):
            cellen = []
            for k in R:
                v = tel(r, k)
                ok = v == gegeven(k, r)
                fouten += not ok
                cellen.append(f"{v}{' ' if ok else '*'}")
            print(f"{r} | " + " ".join(cellen))
        print("    " + " ".join(f"{k} " for k in R) + "\n")
    print("Alles klopt." if fouten == 0 else f"{fouten} afwijking(en).")


if __name__ == "__main__":
    p = solve()
    if p:
        print(" -> ".join(map(str, p)))
        print()
        print_sums(p)
