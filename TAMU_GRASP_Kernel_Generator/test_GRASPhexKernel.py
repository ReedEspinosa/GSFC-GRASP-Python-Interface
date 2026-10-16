"""Checks of the kernel formulas in GRASPhexKernel.py that do not need a TAMUdust2020 run.

  python test_GRASPhexKernel.py [isca.dat kernels_299_00.txt]

1. volume_kernels: b/dlnr = Qe*A/V (= C/V) for every size descriptor, and the pre-2026-10
   formula Qe*3/(4*r_v) is low by exactly the sphericity pi*r_v^2/A.
2. Phase_Function_299: element 34 is written as kernels_299_34.txt with -P43*P11*b_sca, and the
   other elements as P_ij*P11*b_sca (synthetic input, temporary folder).
3. Optional: with a TAMUdust2020 isca.dat (one block of 273 sizes per refractive index) and a
   kernels_299_00.txt built from the same sizes, print how well volume_kernels reproduces the
   stored extinction for the best-matching refractive index of each block.
"""
import os
import sys
import tempfile
import types

import numpy as np

sys.modules.setdefault('fortranformat', types.SimpleNamespace(FortranRecordWriter=None))  # only used for grid1.dat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import GRASPhexKernel as G

rng = np.random.default_rng(1)
nmr, nmi, nr = 2, 3, 273
psi = 0.70064
D = np.geomspace(1e-3, 300, nr)
V = np.broadcast_to(0.4888**3*np.pi/6*D**3, (nmr, nmi, nr)).copy()        # r_v = 0.4888*D/2
rv = np.cbrt(3*V/(4*np.pi))
A = np.pi*rv**2/psi
Qe = 2 + rng.normal(0, 0.05, (nmr, nmi, nr))
Qe[:, 0, -1] = 2.0                                                        # geometric-optics limit
Qa = Qe*rng.uniform(0.1, 0.9, (nmr, nmi, nr))

ok = True
for sd in (None, 'VER', 'Allsaitoang', 'TAMUratioA2V_VER', 'ER', 'TAMUratioA2V_ER'):
    b_ext, b_abs, R, dlnr, _ = G.volume_kernels(Qe, Qa, A, V, sd, 0.7)
    e1 = np.abs(b_ext/dlnr/(Qe*A/V) - 1).max()
    e2 = np.abs(b_abs/dlnr/(Qa*A/V) - 1).max()
    old = Qe*3*dlnr/(4*rv)
    e3 = np.abs(old/b_ext/psi - 1).max()
    good = max(e1, e2, e3) < 1e-12
    ok &= good
    print('%-17s b/dlnr = Q*A/V: %.1e %.1e ; old/new = psi: %.1e  %s' % (sd, e1, e2, e3, 'ok' if good else 'FAIL'))
p, a = G.ensemble_sphericity(A, V)
good = abs(p/psi - 1) < 1e-12 and abs(a*psi - 1) < 1e-12
ok &= good
print('ensemble_sphericity: psi %.5f alpha %.5f  %s' % (p, a, 'ok' if good else 'FAIL'))

try:
    G.volume_kernels(Qe*psi, Qa, A, V)   # a kernel in the old convention must trip the large-x check
    print('large-x check did not trigger  FAIL'); ok = False
except AssertionError:
    print('large-x check triggers on a true-area-normalized kernel  ok')

# element files from synthetic phase matrices
nmr, nmi, nang = 2, 2, 3
b_sca = rng.uniform(1e-4, 1e-2, (nmr, nmi, nr))
P11 = rng.uniform(0.1, 10, (nmr, nmi*nr, nang))
Pn = {el: rng.uniform(-1, 1, (nmr, nmi*nr, nang)) for el in ('P12', 'P22', 'P33', 'P43', 'P44')}
Dict1 = dict(theta=np.array([0., 90., 180.]), rv=np.geomspace(1e-3, 150, nr), rri=np.array([1.4, 1.5]),
             iri=np.array([1e-3, 1e-2]), P11=P11, Qsca=b_sca, **Pn)
with tempfile.TemporaryDirectory() as tmp:
    for el in (11, 12, 22, 33, 43, 44):
        G.Phase_Function_299(Dict1, el, tmp)
    files = sorted(f for f in os.listdir(tmp) if f.startswith('kernels_299_'))
    good = files == ['kernels_299_%s.txt' % e for e in ('11', '12', '22', '33', '34', '44')]
    ok &= good
    print('files written: %s  %s' % (', '.join(files), 'ok' if good else 'FAIL'))
    for el, ref in (('11', P11), ('12', P11*Pn['P12']), ('34', -P11*Pn['P43'])):
        lines = open(os.path.join(tmp, 'kernels_299_%s.txt' % el)).read().split('\n')
        vals = [ln.split() for ln in lines if ln.count('E') == nang and 'element' not in ln and 'range' not in ln]
        got = np.array(vals, float).reshape(nmr, nmi, nr, nang)
        want = ref.reshape(nmr, nmi, nr, nang)*b_sca[..., None]
        err = np.abs(got - want).max()/np.abs(want).max()
        good = err < 1e-6
        ok &= good
        print('element %s = %s * b_sca: max rel err %.1e  %s' % (el, {'11': 'P11', '12': 'P12/P11*P11', '34': '-P43/P11*P11'}[el], err, 'ok' if good else 'FAIL'))

if len(sys.argv) > 2:
    import pandas as pd
    d = pd.read_csv(sys.argv[1], sep=r'\s+', header=None, comment='W').values.astype(float)
    nb = len(d)//273
    k = open(sys.argv[2]).read().split('\n')
    blocks, cur = [], None
    for ln in k:
        if 'EXTINCTION' in ln:
            cur = []; blocks.append(cur)
        elif 'ABSORPTION' in ln or 'element, ratio' in ln:
            cur = None
        elif cur is not None:
            cur += ln.split()
    stored = np.array(blocks, float)                      # (n_ri, 273)
    for b in range(nb):
        blk = d[b*273:(b+1)*273]
        Vb, Ab, Qb = blk[:, 2], blk[:, 3], blk[:, 4]
        new = (Qb*Ab/Vb)
        rvb = np.cbrt(3*Vb/(4*np.pi)); dl = np.log(rvb[1:]/rvb[:-1]).mean()
        ratio = stored/dl/new[None, :]
        best = np.argmin(np.median(np.abs(np.log(ratio)), axis=1))
        print('isca block %d: best-matching kernel refractive index #%d; stored/(Qe*A/V): median %.5f, 5-95%% %.5f-%.5f'
              % (b, best, np.median(ratio[best]), *np.percentile(ratio[best], [5, 95])))

print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
