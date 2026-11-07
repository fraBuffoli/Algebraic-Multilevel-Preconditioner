"""Emit TikZ for a small grid graph split into 3 subdomains, with the overlap of subdomain 2."""
NX, NY = 15, 8
def sub(i, j):
    b1 = 5 + (1 if j in (3, 4, 5) else 0) - (1 if j == 7 else 0)
    b2 = 10 - (1 if j in (0, 1, 2) else 0) + (1 if j == 6 else 0)
    return 1 if i < b1 else (2 if i < b2 else 3)
h = 0.42
out = []
# edges
for j in range(NY):
    for i in range(NX):
        if i + 1 < NX: out.append(r"\draw[black!18,line width=0.4pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (i*h, j*h, (i+1)*h, j*h))
        if j + 1 < NY: out.append(r"\draw[black!18,line width=0.4pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (i*h, j*h, i*h, (j+1)*h))
gam = set()
for j in range(NY):
    for i in range(NX):
        if sub(i, j) == 2:
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                a, b = i+di, j+dj
                if 0 <= a < NX and 0 <= b < NY and sub(a, b) != 2: gam.add((a, b))
for j in range(NY):
    for i in range(NX):
        s = sub(i, j)
        if s == 2: st = "own"
        elif (i, j) in gam: st = "gam%d" % s
        else: st = "oth%d" % s
        out.append(r"\node[%s] at (%.2f,%.2f) {};" % (st, i*h, j*h))
print("\n".join(out))
