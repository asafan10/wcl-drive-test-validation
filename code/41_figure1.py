"""Fig. 1 of the letter: (a) change in R2 from adding all 97 or the 61 plain 4G features to the coordinates-only model,
with 95 % trace-level t intervals; (b) 500 m-block minus LOTO gap after dropping 36 random plain features or the
36 time-window features.  usage: python 41_figure1.py   -> results/fig1.png, results/fig1.pdf"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
res = Path(__file__).resolve().parent.parent / 'results'; out = res
COL_W = 3.5
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#6b6b66'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'grid.linewidth': 0.4, 'grid.color': '#d9d9d6',
    'legend.frameon': False, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
def save(fig, out, name):
    fig.savefig(out / f'{name}.png'); fig.savefig(out / f'{name}.pdf'); plt.close(fig); print('wrote', name)
def figA4_mechanism(res, out):
    """(a) 4G contribution by design, full vs plain 4G; (b) interaction under random-36 drops vs time-window drop."""
    import glob as _g
    m = json.load(open(res / 'r4_main.json')); bl = json.load(open(res / 'r4_blocks.json'))
    rows = [('500 m\nblocks', m['d|spatial|g4c']['d'], m['d|spatial|g4c']['ct95'], m['d|spatial|g4c_notime']['d'], m['d|spatial|g4c_notime']['ct95'])]
    for bs, lab in [(1000, '1 km\nblocks'), (2000, '2 km\nblocks')]:
        x = bl[f'block{bs}']; rows.append((lab, x['d_g4c'][0], x['d_g4c'][1:], x['d_g4c_notime'][0], x['d_g4c_notime'][1:]))
    from scipy.stats import t as _t
    P5 = pd.concat([pd.read_csv(f) for f in _g.glob(str(res / 'r5_parts_*.csv'))]).drop_duplicates(['seed', 'trace'])
    def _ti(c):
        g = len(c); se = np.sqrt(g / (g - 1) * ((c - c.mean()) ** 2).sum()); q = _t.ppf(.975, g - 1); return c.sum(), [c.sum() - q * se, c.sum() + q * se]
    fa, fai = _ti(P5.groupby('trace').full.mean()); pa, pai = _ti(P5.groupby('trace').plain.mean())
    rows.append(('Trace ×5\n(100 part.)', fa, fai, pa, pai))
    rows.append(('LOTO', m['d|loto|g4c']['d'], m['d|loto|g4c']['ct95'], m['d|loto|g4c_notime']['d'], m['d|loto|g4c_notime']['ct95']))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw={'width_ratios': [1.75, 1]})
    x = np.arange(len(rows))
    for off, i, c, mk, lab in [(-.13, 1, ORANGE, 'o', 'All 97 4G features'), (.13, 3, BLUE, 's', 'Plain 4G (no time window)')]:
        v = np.array([r[i] for r in rows]); lo = np.array([r[i + 1][0] for r in rows]); hi = np.array([r[i + 1][1] for r in rows])
        ax.errorbar(x + off, v, yerr=[v - lo, hi - v], fmt=mk, color=c, ms=3.2, capsize=2, lw=.9, label=lab)
    ax.axhline(0, color='k', lw=.6); ax.set_xticks(x, [r[0] for r in rows], fontsize=5.6); ax.set_ylabel('Δ R² vs coordinates only', fontsize=6.5)
    ax.set_ylim(-.175, .14); ax.grid(alpha=.5, axis='y'); ax.legend(loc='upper right', fontsize=5.4, handletextpad=.3, borderaxespad=.2); ax.text(.03, .92, '(a)', transform=ax.transAxes, fontsize=7)
    rd = pd.concat([pd.read_csv(f) for f in _g.glob(str(res / 'r4_rand36_*.csv'))])
    c = {d: m[d]['coords'] for d in ('spatial', 'loto')}
    ri = (rd.spatial - c['spatial']) - (rd.loto - c['loto'])
    jit = np.random.default_rng(0).uniform(-.12, .12, len(ri))
    bx.scatter(np.zeros(len(ri)) + jit, ri, s=6, color=GREY, label='36 random dropped')
    bx.scatter([1], [m['inter|loto|g4c_notime']['d']], s=22, color=BLUE, marker='s', label='Time window dropped')
    bx.scatter([-.5 + 0], [m['inter|loto|g4c']['d']], s=0)
    bx.axhline(m['inter|loto|g4c']['d'], color=ORANGE, lw=.9, ls='--', label='All 97 features')
    bx.axhline(0, color='k', lw=.6); bx.set_xticks([0, 1], ['Random\n36', 'Time\nwindow'], fontsize=5.6); bx.set_xlim(-.5, 1.5)
    bx.set_ylabel('500 m − LOTO gap', fontsize=6.5); bx.grid(alpha=.5, axis='y'); bx.text(.05, .92, '(b)', transform=bx.transAxes, fontsize=7)
    bx.set_ylim(-.015, .135); bx.legend(loc='center right', fontsize=4.6, handletextpad=.3, borderaxespad=.2, bbox_to_anchor=(1.0, .23))
    fig.tight_layout(w_pad=.5); save(fig, out, 'fig1')


figA4_mechanism(res, out)
