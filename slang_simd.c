/* Slang-puzzel CoVer-Almanak 2003-2004 — oplosser in C met SIMD
 *
 * Compileren:  gcc -O3 -march=native -o slang slang_simd.c
 *              (clang werkt ook; geen externe libraries nodig)
 *
 * Aanpak
 *  - De puzzel wordt een SAT-probleem: 1 variabele per cel (blokje of niet).
 *      * elke rij uit de drie aanzichten: precies k van de 10 cellen aan
 *      * slang raakt zichzelf niet: een aan-cel heeft precies 2 aan-buren,
 *        begin- en eindpunt precies 1
 *  - Een CDCL-SAT-solver (bewaakte literalen, 1UIP-leren, VSIDS-heap,
 *    Luby-herstarts, opruimen van geleerde clausules) lost dat op.
 *  - De kubus wordt daarnaast als bitbord bijgehouden: per (x,y) een 16-bits
 *    kolom met bit z. Met SIMD-vectoren (GCC/Clang vector extensions; worden
 *    SSE/AVX2/AVX-512 op x86 of NEON op ARM) gebeurt:
 *      * flood fill om losse ringen te vinden (6 buurrichtingen tegelijk
 *        voor 16 kolommen per instructie),
 *      * het tellen van de drie aanzichten voor controle en print_sums.
 *    Gevonden ringen worden verboden met een extra clausule; dan opnieuw.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N    10
#define LEN  351
#define NV   1000
static const int START[3] = {0, 0, 0}, END[3] = {6, 3, 9};

/* Bovenste rij = coördinaat 9, linker kolom = 0 (zoals in het boekje). */
/* Bovenaanzicht: rijen y=9..0, kolommen x, telt langs z */
static const int TOP[N][N] = {
    {4,5,3,4,5,3,1,3,5,6},{3,5,3,5,5,4,5,3,1,6},{2,3,1,3,1,4,3,2,2,4},
    {7,1,4,5,6,4,4,5,4,4},{1,4,4,4,2,4,6,1,3,4},{3,7,2,3,6,5,0,5,4,2},
    {4,3,0,1,1,2,3,3,3,4},{3,3,3,5,5,5,4,5,2,3},{2,1,3,2,3,5,2,1,4,5},
    {7,3,6,5,4,4,4,0,4,6}};
/* Vooraanzicht: rijen z=9..0, kolommen y, telt langs x */
static const int FRONT[N][N] = {
    {5,2,4,3,5,5,6,3,6,3},{4,4,5,5,3,2,2,1,3,5},{3,1,4,1,4,5,5,5,4,4},
    {4,3,3,3,2,1,4,3,3,3},{5,5,1,1,4,3,5,2,3,4},{4,2,2,0,5,1,4,2,4,6},
    {3,3,6,2,4,2,6,4,3,2},{7,2,4,2,2,5,4,2,6,3},{4,1,5,3,4,4,1,1,2,5},
    {4,5,4,4,4,5,7,2,6,4}};
/* Zijaanzicht: rijen z=9..0, kolommen x, telt langs y */
static const int SIDE[N][N] = {
    {0,5,5,5,6,3,3,3,6,6},{2,4,2,4,4,4,5,3,2,4},{6,2,4,3,6,4,3,2,2,4},
    {5,1,0,4,5,4,1,5,1,3},{3,5,4,4,3,3,0,2,4,5},{2,3,4,1,2,6,6,2,3,1},
    {5,3,3,5,3,4,1,4,2,5},{4,2,3,3,2,5,3,4,5,6},{3,4,2,5,2,2,5,1,3,3},
    {6,6,2,3,5,5,5,2,4,7}};
static int top(int x, int y)   { return TOP[9 - y][x]; }
static int front(int y, int z) { return FRONT[9 - z][y]; }
static int side(int x, int z)  { return SIDE[9 - z][x]; }

static int cid(int x, int y, int z) { return (x * N + y) * N + z; }

/* ======================================================= SIMD-bitbord */
/* Kolom (x,y) staat op index PAD + x*STR + y; bits 0..9 zijn z.
 * STR=12 laat 2 lege kolommen tussen x-vlakken, zodat y±1 nooit omklapt
 * naar een ander x-vlak. PAD vangt de verschoven loads aan beide kanten. */
#define STR  12
#define PAD  16
#define BBN  160
#define LANES 16
typedef uint16_t v16 __attribute__((vector_size(32)));   /* 16 x 16 bit */
typedef struct { _Alignas(32) uint16_t c[BBN]; } Board;

static int bix(int x, int y) { return PAD + x * STR + y; }

#define ld(p) ({ v16 ld_v_; memcpy(&ld_v_, (p), sizeof ld_v_); ld_v_; })

/* Eén groeistap: alles wat aan 'cur' grenst (6 richtingen), binnen 'mask'. */
static int dilate(const Board *cur, Board *nxt, const Board *mask)
{
    v16 diff = {0};
    for (int i = PAD; i < PAD + N * STR; i += LANES) {
        v16 c = ld(cur->c + i);
        v16 g = c | (c << 1) | (c >> 1)             /* z ± 1 */
              | ld(cur->c + i - 1) | ld(cur->c + i + 1)       /* y ± 1 */
              | ld(cur->c + i - STR) | ld(cur->c + i + STR);  /* x ± 1 */
        g &= ld(mask->c + i);
        memcpy(nxt->c + i, &g, sizeof g);
        diff |= g ^ c;
    }
    for (int k = 0; k < LANES; k++) if (diff[k]) return 1;
    return 0;
}

/* Samenhangend deel van 'mask' dat 'seed' bevat. */
static void flood(Board *comp, const Board *seed, const Board *mask)
{
    Board a = *seed, b;
    memset(&b, 0, sizeof b);
    for (;;) {
        if (!dilate(&a, &b, mask)) { *comp = b; return; }
        if (!dilate(&b, &a, mask)) { *comp = a; return; }
    }
}

static int popcount_board(const Board *b)
{
    int n = 0;
    for (int i = 0; i < BBN; i++) n += __builtin_popcount(b->c[i]);
    return n;
}

static void andnot(Board *r, const Board *a, const Board *b)  /* r = a & ~b */
{
    for (int i = 0; i < BBN; i += LANES) {
        v16 v = ld(a->c + i) & ~ld(b->c + i);
        memcpy(r->c + i, &v, sizeof v);
    }
}

/* Aanzichten tellen uit een bitbord.
 * boven: popcount per kolom; voor en zij: per z-bit sommeren over x resp. y,
 * steeds 16 kolommen tegelijk. */
static void views(const Board *b, int t[N][N], int f[N][N], int s[N][N])
{
    for (int x = 0; x < N; x++)
        for (int y = 0; y < N; y++) t[x][y] = __builtin_popcount(b->c[bix(x, y)]);
    for (int z = 0; z < N; z++) {
        /* voor (y,z): som over x van bit z, voor alle y tegelijk */
        v16 acc = {0};
        for (int x = 0; x < N; x++)
            acc += (ld(b->c + bix(x, 0)) >> z) & 1;
        for (int y = 0; y < N; y++) f[y][z] = acc[y];
        /* zij (x,z): som over y van bit z, per x-vlak horizontaal optellen */
        for (int x = 0; x < N; x++) {
            v16 v = (ld(b->c + bix(x, 0)) >> z) & 1;
            int sum = 0;
            for (int y = 0; y < N; y++) sum += v[y];
            s[x][z] = sum;
        }
    }
}

/* ======================================================= SAT-solver */
typedef struct { int *d; int n, cap; } Vec;
static void vpush(Vec *v, int x)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->d = realloc(v->d, (size_t)v->cap * sizeof(int));
        if (!v->d) { perror("realloc"); exit(1); }
    }
    v->d[v->n++] = x;
}

static Vec arena, cstart, clen, clbd, learnts;   /* clausules */
static Vec cflags;                                /* bit0 geleerd, bit1 weg */
static Vec watches[2 * NV];
static int8_t val[NV];
static int level[NV], reason[NV], trail[NV], trail_n, qhead;
static int lim[NV + 1], nlim;
static double act[NV], inc = 1.0;
static int8_t phase[NV];
static int heap[NV], hpos[NV], hn;
static int ok = 1;
static long conflicts;

static int lv(int l) { int v = val[l >> 1]; return v < 0 ? -1 : v ^ (l & 1); }

static void heap_up(int i)
{
    int v = heap[i];
    while (i > 0) {
        int p = (i - 1) / 2;
        if (act[heap[p]] >= act[v]) break;
        heap[i] = heap[p]; hpos[heap[i]] = i; i = p;
    }
    heap[i] = v; hpos[v] = i;
}
static void heap_down(int i)
{
    int v = heap[i];
    for (;;) {
        int c = 2 * i + 1;
        if (c >= hn) break;
        if (c + 1 < hn && act[heap[c + 1]] > act[heap[c]]) c++;
        if (act[heap[c]] <= act[v]) break;
        heap[i] = heap[c]; hpos[heap[i]] = i; i = c;
    }
    heap[i] = v; hpos[v] = i;
}
static void heap_insert(int v)
{
    if (hpos[v] >= 0) return;
    heap[hn] = v; hpos[v] = hn; hn++; heap_up(hn - 1);
}
static int heap_pop(void)
{
    int v = heap[0];
    hpos[v] = -1; hn--;
    if (hn > 0) { heap[0] = heap[hn]; hpos[heap[0]] = 0; heap_down(0); }
    return v;
}

static void assign(int l, int r)
{
    int v = l >> 1;
    val[v] = 1 - (l & 1);
    level[v] = nlim;
    reason[v] = r;
    trail[trail_n++] = l;
}

static int new_clause(const int *lits, int n, int learnt)
{
    int ci = cstart.n;
    vpush(&cstart, arena.n);
    vpush(&clen, n);
    vpush(&clbd, 0);
    vpush(&cflags, learnt ? 1 : 0);
    for (int k = 0; k < n; k++) vpush(&arena, lits[k]);
    vpush(&watches[lits[0]], ci);
    vpush(&watches[lits[1]], ci);
    if (learnt) vpush(&learnts, ci);
    return ci;
}

/* Clausule toevoegen op niveau 0 (vereenvoudigd tegen vaste waarden). */
static void add_clause(const int *in, int n)
{
    int lits[16], m = 0;
    for (int k = 0; k < n; k++) {
        int l = in[k];
        if (lv(l) == 1 && level[l >> 1] == 0) return;          /* al vervuld */
        if (lv(l) == 0 && level[l >> 1] == 0) continue;        /* valt weg */
        int dup = 0;
        for (int j = 0; j < m; j++) if (lits[j] == l) dup = 1;
        if (!dup) lits[m++] = l;
    }
    if (m == 0) { ok = 0; return; }
    if (m == 1) {
        if (lv(lits[0]) == 0) ok = 0;
        else if (lv(lits[0]) < 0) assign(lits[0], -1);
        return;
    }
    new_clause(lits, m, 0);
}
/* Lange clausule (losse ring) mag langer dan 16 zijn. */
static void add_long_clause(const int *lits, int n)
{
    if (n == 1) { add_clause(lits, 1); return; }
    new_clause(lits, n, 0);
}

static int propagate(void)
{
    while (qhead < trail_n) {
        int fl = trail[qhead++] ^ 1;            /* deze literaal is nu onwaar */
        Vec *ws = &watches[fl];
        int i = 0, j = 0;
        while (i < ws->n) {
            int ci = ws->d[i++];
            if (cflags.d[ci] & 2) continue;     /* opgeruimd: laten vallen */
            int *c = arena.d + cstart.d[ci], len = clen.d[ci];
            if (c[0] == fl) { c[0] = c[1]; c[1] = fl; }
            int o = c[0];
            if (lv(o) == 1) { ws->d[j++] = ci; continue; }
            int k;
            for (k = 2; k < len; k++)
                if (lv(c[k]) != 0) {
                    c[1] = c[k]; c[k] = fl;
                    vpush(&watches[c[1]], ci);
                    break;
                }
            if (k < len) continue;
            ws->d[j++] = ci;
            if (lv(o) < 0) assign(o, ci);
            else {                              /* conflict */
                while (i < ws->n) ws->d[j++] = ws->d[i++];
                ws->n = j;
                qhead = trail_n;
                return ci;
            }
        }
        ws->n = j;
    }
    return -1;
}

static void bump(int v)
{
    if ((act[v] += inc) > 1e100) {
        for (int u = 0; u < NV; u++) act[u] *= 1e-100;
        inc *= 1e-100;
    }
    if (hpos[v] >= 0) heap_up(hpos[v]);
}

static int8_t seen[NV];
static int analyze(int ci, int *learnt, int *out_n, int *out_lbd)
{
    int n = 1, count = 0, p = -1, idx = trail_n - 1;
    for (;;) {
        int *c = arena.d + cstart.d[ci], len = clen.d[ci];
        for (int k = 0; k < len; k++) {
            int q = c[k], v = q >> 1;
            if (q == p) continue;
            if (!seen[v] && level[v] > 0) {
                seen[v] = 1;
                bump(v);
                if (level[v] == nlim) count++;
                else learnt[n++] = q;
            }
        }
        while (!seen[trail[idx] >> 1]) idx--;
        p = trail[idx--];
        seen[p >> 1] = 0;
        if (--count == 0) break;
        ci = reason[p >> 1];
    }
    learnt[0] = p ^ 1;
    /* minimaliseren: literaal weg als zijn reden al door de rest gedekt is */
    static int orig[NV + 1];
    memcpy(orig, learnt, (size_t)n * sizeof(int));
    int m = 1;
    for (int k = 1; k < n; k++) {
        int q = learnt[k], r = reason[q >> 1], keep = (r < 0);
        if (!keep) {
            int *c = arena.d + cstart.d[r], len = clen.d[r];
            for (int t = 0; t < len; t++) {
                int x = c[t];
                if (x != (q ^ 1) && !seen[x >> 1] && level[x >> 1] > 0) { keep = 1; break; }
            }
        }
        if (keep) learnt[m++] = q;
    }
    for (int k = 1; k < n; k++) seen[orig[k] >> 1] = 0;
    n = m;
    int bt = 0;
    if (n > 1) {
        int mi = 1;
        for (int k = 2; k < n; k++) if (level[learnt[k] >> 1] > level[learnt[mi] >> 1]) mi = k;
        int t = learnt[1]; learnt[1] = learnt[mi]; learnt[mi] = t;
        bt = level[learnt[1] >> 1];
    }
    /* LBD: aantal verschillende niveaus */
    int lbd = 0;
    static int stamp[NV + 1], sid = 0;
    sid++;
    for (int k = 0; k < n; k++) {
        int L = level[learnt[k] >> 1];
        if (stamp[L] != sid) { stamp[L] = sid; lbd++; }
    }
    inc /= 0.95;
    *out_n = n; *out_lbd = lbd;
    return bt;
}

static void backtrack(int lvl)
{
    if (nlim <= lvl) return;
    for (int k = trail_n - 1; k >= lim[lvl]; k--) {
        int v = trail[k] >> 1;
        phase[v] = val[v];
        val[v] = -1;
        reason[v] = -1;
        heap_insert(v);
    }
    trail_n = lim[lvl];
    qhead = trail_n;
    nlim = lvl;
}

static int cmp_lbd_desc(const void *a, const void *b)
{
    return clbd.d[*(const int *)b] - clbd.d[*(const int *)a];
}
static void reduce_db(void)
{
    Vec cand = {0};
    for (int k = 0; k < learnts.n; k++) {
        int ci = learnts.d[k];
        if (cflags.d[ci] & 2) continue;
        int lit0 = arena.d[cstart.d[ci]];
        int locked = (lv(lit0) == 1 && reason[lit0 >> 1] == ci);
        if (!locked && clbd.d[ci] > 2) vpush(&cand, ci);
    }
    qsort(cand.d, (size_t)cand.n, sizeof(int), cmp_lbd_desc);
    for (int k = 0; k < cand.n / 2; k++) cflags.d[cand.d[k]] |= 2;
    /* lijst van levende geleerde clausules opnieuw opbouwen */
    int j = 0;
    for (int k = 0; k < learnts.n; k++)
        if (!(cflags.d[learnts.d[k]] & 2)) learnts.d[j++] = learnts.d[k];
    learnts.n = j;
    free(cand.d);
}

static int luby(int i)
{
    int k = 1;
    while ((1 << k) - 1 < i + 1) k++;
    while ((1 << k) - 1 != i + 1) {
        i -= (1 << (k - 1)) - 1;
        k = 1;
        while ((1 << k) - 1 < i + 1) k++;
    }
    return 1 << (k - 1);
}

static int sat_solve(void)
{
    static int learnt[NV + 1];
    if (!ok) return 0;
    backtrack(0);
    if (propagate() >= 0) return 0;
    int restart = 0, next_reduce = 4000;
    for (;;) {
        long limit = 100L * luby(restart++), local = 0;
        for (;;) {
            int ci = propagate();
            if (ci >= 0) {
                conflicts++; local++;
                if (nlim == 0) return 0;
                int n, lbd, bt = analyze(ci, learnt, &n, &lbd);
                backtrack(bt);
                if (n == 1) assign(learnt[0], -1);
                else {
                    int nc = new_clause(learnt, n, 1);
                    clbd.d[nc] = lbd;
                    assign(learnt[0], nc);
                }
                continue;
            }
            if (local >= limit) { backtrack(0); break; }
            if (learnts.n > next_reduce) { reduce_db(); next_reduce += 1000; }
            int v = -1;
            while (hn > 0) { int u = heap_pop(); if (val[u] < 0) { v = u; break; } }
            if (v < 0) return 1;                 /* alles toegekend: model */
            lim[nlim++] = trail_n;
            assign(2 * v + (phase[v] ? 0 : 1), -1);
        }
    }
}

/* ======================================================= model */
static int ON(int c)  { return 2 * c; }
static int OFF(int c) { return 2 * c + 1; }

/* Alle k-deelverzamelingen van grp; per deelverzameling één clausule met
 * literalen lit(c) (pos=1: aan, pos=0: uit), eventueel met 'extra' ervoor. */
static void subsets(const int *grp, int n, int k, int pos, int extra)
{
    if (k <= 0 || k > n) { if (k == 0) { int l = extra; if (extra >= 0) add_clause(&l, 1); } return; }
    int idx[16], lits[17];
    for (int i = 0; i < k; i++) idx[i] = i;
    for (;;) {
        int m = 0;
        if (extra >= 0) lits[m++] = extra;
        for (int i = 0; i < k; i++) lits[m++] = pos ? ON(grp[idx[i]]) : OFF(grp[idx[i]]);
        add_clause(lits, m);
        int i = k - 1;
        while (i >= 0 && idx[i] == n - k + i) i--;
        if (i < 0) break;
        idx[i]++;
        for (int j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
    }
}
static void exactly(const int *grp, int k)
{
    subsets(grp, N, k + 1, 0, -1);       /* hoogstens k aan */
    subsets(grp, N, N - k + 1, 1, -1);   /* minstens k aan */
}

static int nbrs(int c, int *out)
{
    int x = c / 100, y = c / 10 % 10, z = c % 10, n = 0;
    if (x > 0) out[n++] = c - 100;
    if (x < 9) out[n++] = c + 100;
    if (y > 0) out[n++] = c - 10;
    if (y < 9) out[n++] = c + 10;
    if (z > 0) out[n++] = c - 1;
    if (z < 9) out[n++] = c + 1;
    return n;
}

static void build_model(void)
{
    int g[N];
    for (int x = 0; x < N; x++) for (int y = 0; y < N; y++) {
        for (int z = 0; z < N; z++) g[z] = cid(x, y, z);
        exactly(g, top(x, y));
    }
    for (int y = 0; y < N; y++) for (int z = 0; z < N; z++) {
        for (int x = 0; x < N; x++) g[x] = cid(x, y, z);
        exactly(g, front(y, z));
    }
    for (int x = 0; x < N; x++) for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) g[y] = cid(x, y, z);
        exactly(g, side(x, z));
    }
    int S = cid(START[0], START[1], START[2]), E = cid(END[0], END[1], END[2]);
    int l = ON(S); add_clause(&l, 1);
    l = ON(E);     add_clause(&l, 1);
    for (int c = 0; c < NV; c++) {
        int nb[6], n = nbrs(c, nb), d = (c == S || c == E) ? 1 : 2;
        subsets(nb, n, d + 1, 0, OFF(c));        /* hoogstens d buren aan */
        subsets(nb, n, n - d + 1, 1, OFF(c));    /* minstens d buren aan */
    }
}

static void model_board(Board *b)
{
    memset(b, 0, sizeof *b);
    for (int c = 0; c < NV; c++)
        if (val[c] == 1) b->c[bix(c / 100, c / 10 % 10)] |= (uint16_t)(1u << (c % 10));
}

static void print_sums(const Board *b)
{
    int t[N][N], f[N][N], s[N][N], fout = 0;
    views(b, t, f, s);
    const char *naam[3] = {"bovenaanzicht (rij y, kolom x, telt langs z)",
                           "vooraanzicht (rij z, kolom y, telt langs x)",
                           "zijaanzicht (rij z, kolom x, telt langs y)"};
    for (int w = 0; w < 3; w++) {
        printf("%s\n", naam[w]);
        for (int r = 9; r >= 0; r--) {
            printf("%d | ", r);
            for (int k = 0; k < N; k++) {
                int v = w == 0 ? t[k][r] : w == 1 ? f[k][r] : s[k][r];
                int g = w == 0 ? top(k, r) : w == 1 ? front(k, r) : side(k, r);
                fout += v != g;
                printf("%d%c ", v, v == g ? ' ' : '*');
            }
            printf("\n");
        }
        printf("    ");
        for (int k = 0; k < N; k++) printf("%d  ", k);
        printf("\n\n");
    }
    if (fout) printf("%d afwijking(en).\n", fout); else printf("Alles klopt.\n");
}

int main(void)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int v = 0; v < NV; v++) { val[v] = -1; reason[v] = -1; hpos[v] = -1; }
    for (int v = 0; v < NV; v++) heap_insert(v);
    build_model();

    int S = cid(START[0], START[1], START[2]), E = cid(END[0], END[1], END[2]);
    Board on, seed, comp, rest;
    int rondes = 0;
    for (;;) {
        rondes++;
        if (!sat_solve()) { printf("Geen oplossing.\n"); return 1; }
        model_board(&on);
        memset(&seed, 0, sizeof seed);
        seed.c[bix(START[0], START[1])] = 1u << START[2];
        flood(&comp, &seed, &on);                 /* de slang vanaf START */
        if (popcount_board(&comp) == LEN) break;
        /* overige aan-cellen zijn losse ringen: elk verbieden */
        andnot(&rest, &on, &comp);
        backtrack(0);
        while (popcount_board(&rest)) {
            int i = 0;
            while (!rest.c[i]) i++;
            memset(&seed, 0, sizeof seed);
            seed.c[i] = rest.c[i] & (uint16_t)-rest.c[i];   /* laagste bit */
            flood(&comp, &seed, &rest);
            Vec lits = {0};
            for (int x = 0; x < N; x++) for (int y = 0; y < N; y++)
                for (int z = 0; z < N; z++)
                    if (comp.c[bix(x, y)] >> z & 1) vpush(&lits, OFF(cid(x, y, z)));
            add_long_clause(lits.d, lits.n);
            free(lits.d);
            andnot(&rest, &rest, &comp);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("Opgelost in %.1f ms, %ld conflicten, %d ronde(s).\n", ms, conflicts, rondes);

    /* pad volgen van START naar END */
    int prev = -1, cur = S, n = 1;
    printf("(%d, %d, %d)", START[0], START[1], START[2]);
    while (cur != E) {
        int nb[6], k = nbrs(cur, nb), nx = -1;
        for (int i = 0; i < k; i++) if (val[nb[i]] == 1 && nb[i] != prev) { nx = nb[i]; break; }
        prev = cur; cur = nx; n++;
        printf(" -> (%d, %d, %d)", cur / 100, cur / 10 % 10, cur % 10);
    }
    printf("\n\nLengte: %d blokjes\n\n", n);
    print_sums(&on);
    return 0;
}
