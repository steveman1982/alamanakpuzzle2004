# Slang-puzzel CoVer-Almanak 2003-2004 — oplosser met OR-Tools CP-SAT
# pip install ortools
from ortools.sat.python import cp_model

N, LEN = 10, 351
START, END = (0, 0, 0), (6, 3, 9)

# Overgetypt uit de uitvergrote aanzichten; bovenste rij = coördinaat 9, linkse kolom = 0.
# Bovenaanzicht: rijen y=9..0, kolommen x=0..9, waarde = #blokjes langs z
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
# Vooraanzicht: rijen z=9..0, kolommen y=0..9, waarde = #blokjes langs x
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
# Zijaanzicht: rijen z=9..0, kolommen x=0..9, waarde = #blokjes langs y
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
top   = lambda x, y: TOP[9 - y][x]
front = lambda y, z: FRONT[9 - z][y]
side  = lambda x, z: SIDE[9 - z][x]
R = range(N)


def check():
    ok = True
    for name, g in (("boven", TOP), ("voor", FRONT), ("zij", SIDE)):
        s = sum(map(sum, g))
        if s != LEN:
            ok = False
            print(f"{name}aanzicht: som {s} != {LEN}")
    # Kruiscontroles: elk vlak telt per x/y/z-laag hetzelfde aantal blokjes
    for x in R:
        a, b = sum(top(x, y) for y in R), sum(side(x, z) for z in R)
        if a != b: ok = False; print(f"x={x}: boven {a} vs zij {b}")
    for y in R:
        a, b = sum(top(x, y) for x in R), sum(front(y, z) for z in R)
        if a != b: ok = False; print(f"y={y}: boven {a} vs voor {b}")
    for z in R:
        a, b = sum(front(y, z) for y in R), sum(side(x, z) for x in R)
        if a != b: ok = False; print(f"z={z}: voor {a} vs zij {b}")
    return ok


def nbrs(c):
    for d in range(3):
        for s in (-1, 1):
            n = list(c); n[d] += s
            if 0 <= n[d] < N:
                yield tuple(n)


def solve():
    m = cp_model.CpModel()
    cells = [(x, y, z) for x in R for y in R for z in R]
    b = {c: m.NewBoolVar(str(c)) for c in cells}
    for x in R:
        for y in R: m.Add(sum(b[x, y, z] for z in R) == top(x, y))
    for y in R:
        for z in R: m.Add(sum(b[x, y, z] for x in R) == front(y, z))
    for x in R:
        for z in R: m.Add(sum(b[x, y, z] for y in R) == side(x, z))
    # Slang raakt zichzelf niet => elk blokje heeft precies 2 bezette buren
    # (kop en staart precies 1). Dan is het een pad plus evt. losse ringen.
    for c in cells:
        deg = sum(b[n] for n in nbrs(c))
        if c in (START, END):
            m.Add(b[c] == 1); m.Add(deg == 1)
        else:
            m.Add(deg == 2).OnlyEnforceIf(b[c])

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 8
    it = 0
    while True:
        it += 1
        if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Geen oplossing (controleer de overgetypte getallen).")
            return None
        on = {c for c in cells if solver.Value(b[c])}
        # Pad volgen vanaf START
        path, prev, cur = [START], None, START
        while cur != END:
            nxt = next(n for n in nbrs(cur) if n in on and n != prev)
            prev, cur = cur, nxt
            path.append(cur)
        if len(path) == LEN:
            print(f"Opgelost na {it} iteratie(s).")
            return path
        # Losse ringen verbieden en opnieuw oplossen
        rest = on - set(path)
        while rest:
            comp, stack = set(), [rest.pop()]
            while stack:
                c = stack.pop(); comp.add(c)
                for n in nbrs(c):
                    if n in rest: rest.discard(n); stack.append(n)
            m.Add(sum(b[c] for c in comp) <= len(comp) - 1)


def print_sums(path):
    """Print de drie aanzichten berekend uit een oplossing, in dezelfde
    opmaak als in het boekje (bovenste rij = 9, linker kolom = 0).
    Afwijkingen van de puzzelgetallen worden met * gemarkeerd."""
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
    if not check():
        print("Waarschuwing: aanzichten zijn niet consistent; corrigeer eerst de getallen.\n")
    p = solve()
    if p:
        print(" -> ".join(map(str, p)))
        print()
        print_sums(p)
