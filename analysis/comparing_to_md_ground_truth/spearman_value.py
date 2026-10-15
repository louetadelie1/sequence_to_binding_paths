"""
The three metrics of edge_ablation_macrostates/plot_macro_edge_ablation.py, shown as scatter plots,
for the combined edge rule only (1/0 overlap x harmonic, max lift = macro_edges_zlink_harmonic.pkl).

    1. Transition Spearman   - MD transition count vs MLP edge weight, MD edges only (pairs MD connects)
    2. MD flux captured      - share of all MD transitions that sit in the MLP's k strongest pocket pairs
                               (k = number of MD edges)
    3. Hub strength Spearman - per pocket: total MD transitions vs total MLP edge weight

Grey = all 24 pockets, blue = the 12 significantly matched pockets.
Values match edge_ablation_macrostates/tables/macro_edge_rule_ablation.csv (lift = max, combined rule).
"""

import pickle
import re
import operator
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom, spearmanr

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text in the PDF
    "font.family": "sans-serif",
    "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#3d3d3a",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
    "axes.labelsize": 8,
    "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "legend.fontsize": 7,
})

BLUE, GRAY = '#2a78d6', '#b4b3a9'
INK = '#1f1f1d'

RUN_DIR    = '/ptmp/adlouet/camb/sequence_to_binding_paths/predicting_paths/run_msm_macro_micro'
MD_PKL_DIR = ('/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/'
              'community_identification/pickled_files/abeta_gabis')
fig_dir    = f'{RUN_DIR}/figures'

N_POCKETS  = 24
SIG        = 0.05
N_RESIDUES = 42
SEQ        = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
TOP_N      = 31          # MD macrostates used to build the MD macrostate graph (the knee)


def despine(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)


# ════════════════════════════════════════════════════════════════════════════
# 1. Hungarian matching of the top-24 MD and MLP pockets (same as comaring_mlp_to_md_macrostates.py)
# ════════════════════════════════════════════════════════════════════════════
md_df     = pickle.load(open('/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/'
                             'community_identification/df_core_015.pkl', 'rb'))
z_link_df = pickle.load(open(f'{RUN_DIR}/z_link_df.pkl', 'rb'))


def to_pockets(df, residue_col, population_col):
    """DataFrame -> list of (population, {residue numbers}), most populated first."""
    pockets = []
    for _, row in df.iterrows():
        value = row[residue_col]
        pieces = value.split(",") if isinstance(value, str) else \
                 list(value) if hasattr(value, "__iter__") else [value]
        residues = {int(m.group()) for p in pieces
                    for m in [re.search(r"\d+", str(p))] if m and 1 <= int(m.group()) <= N_RESIDUES}
        if residues:
            pockets.append((float(row[population_col]), residues))
    pockets.sort(key=operator.itemgetter(0), reverse=True)
    return pockets


md_pockets = to_pockets(md_df, "core", "P_eq")[:N_POCKETS]
zl_pockets = to_pockets(z_link_df, "residues", "sum_peq")[:N_POCKETS]
n = min(len(md_pockets), len(zl_pockets))
md_pockets, zl_pockets = md_pockets[:n], zl_pockets[:n]

ca_loaded = pickle.load(open('/ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps/'
                             'abeta/ZINC000000030986/ca_dist_matrix_boltz.pkl', 'rb'))
ca_dist = np.asarray(ca_loaded['matrix'] if isinstance(ca_loaded, dict) else ca_loaded, dtype=float)
d0 = np.median(ca_dist[~np.eye(ca_dist.shape[0], dtype=bool)])

KYTE_DOOLITTLE = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
                  'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
                  'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
                  'Y': -1.3, 'V': 4.2}
CHARGE      = {'R': 1.0, 'K': 1.0, 'H': 0.5, 'D': -1.0, 'E': -1.0}
HBOND_DONOR = set('RKWNQHSTY')
HBOND_ACC   = set('DENQHSTY')


def physchem_vec(residues):
    aa = [SEQ[r - 1] for r in residues]
    return [np.mean([KYTE_DOOLITTLE[a] for a in aa]),
            sum(a in HBOND_DONOR for a in aa),
            sum(a in HBOND_ACC for a in aa),
            sum(CHARGE.get(a, 0.0) for a in aa)]


def norm01(S):
    return (S - S.min()) / (S.max() - S.min() + 1e-12)


S_spatial = np.zeros((n, n))
S_hyper   = np.zeros((n, n))
S_overlap = np.zeros((n, n))
for i, (_, md_set) in enumerate(md_pockets):
    md_ix = [r - 1 for r in md_set]
    for j, (_, zl_set) in enumerate(zl_pockets):
        zl_ix = [r - 1 for r in zl_set]
        S_spatial[i, j] = np.exp(-ca_dist[np.ix_(md_ix, zl_ix)].mean() / d0)
        k = len(md_set & zl_set)
        S_hyper[i, j] = 0.0 if k == 0 else min(
            15.0, -np.log10(max(hypergeom.sf(k - 1, N_RESIDUES, len(md_set), len(zl_set)), 1e-15)))
        S_overlap[i, j] = k / min(len(md_set), len(zl_set))

md_vecs = np.array([physchem_vec(res) for _, res in md_pockets])
zl_vecs = np.array([physchem_vec(res) for _, res in zl_pockets])
mean, std = np.concatenate([md_vecs, zl_vecs]).mean(0), np.concatenate([md_vecs, zl_vecs]).std(0) + 1e-12
S_phys = 1.0 / (1.0 + np.linalg.norm((md_vecs - mean)[:, None, :] / std - (zl_vecs - mean)[None, :, :] / std, axis=-1))

S_blend = (norm01(S_spatial) + norm01(S_hyper) + norm01(S_phys) + norm01(S_overlap)) / 4
md_idx, zl_idx = linear_sum_assignment(-S_blend)

# the significantly matched pockets (p <= SIG) = the "12 matched pockets"
sub = []
for i, j in zip(md_idx, zl_idx):
    md_set, zl_set = md_pockets[i][1], zl_pockets[j][1]
    k = len(md_set & zl_set)
    p_value = hypergeom.sf(k - 1, N_RESIDUES, len(md_set), len(zl_set)) if k else 1.0
    if p_value <= SIG:
        sub.append(int(i))
sub = sorted(sub)
m = len(sub)


# ════════════════════════════════════════════════════════════════════════════
# 2. MD macrostate edges = summed MD microstate transition counts between two macrostates
# ════════════════════════════════════════════════════════════════════════════
inv_map_pkl     = pickle.load(open(f'{MD_PKL_DIR}/inv_map.pckl',         'rb'))
P_eq_keys       = pickle.load(open(f'{MD_PKL_DIR}/p_eq_keys.pckl',       'rb'))
filtered_merged = pickle.load(open(f'{MD_PKL_DIR}/filtered_merged.pckl', 'rb'))

G = nx.Graph()
for (node1, node2), weight in filtered_merged:
    if weight >= 1:
        G.add_edge(tuple(node1), tuple(node2), weight=weight)
G.remove_edges_from(list(nx.selfloop_edges(G)))
kept = set(G.nodes())

inv_map   = {int(k): [tuple(int(r) for r in t) for t in nodes] for k, nodes in inv_map_pkl.items()}
clean_peq = {t: float(np.ravel(P_eq_keys[t])[0]) for t in P_eq_keys}

macro_peq  = {k: sum(clean_peq[t] for t in members) for k, members in inv_map.items()}
k_by_rank  = sorted(macro_peq, key=macro_peq.get, reverse=True)          # most populated first == df_core order
row_of_k   = {k: row for row, k in enumerate(k_by_rank)}
nodes_by_k = {k: [t for t in v if t in kept] for k, v in inv_map.items()}
top_ids    = [k for k in k_by_rank[:TOP_N] if nodes_by_k.get(k)]
parts      = {node: k for k in top_ids for node in nodes_by_k[k]}

edge_weight = defaultdict(float)
for u, v, data in G.edges(data=True):
    k1, k2 = parts.get(u), parts.get(v)
    if k1 is None or k2 is None or k1 == k2:
        continue
    edge_weight[(min(k1, k2), max(k1, k2))] += data["weight"]

A_md = np.zeros((n, n))                     # A_md[i, j] = MD transitions between MD pockets of rank i and j
for (k1, k2), w in edge_weight.items():
    i, j = row_of_k[k1], row_of_k[k2]
    if i < n and j < n:
        A_md[i, j] = A_md[j, i] = w


# ════════════════════════════════════════════════════════════════════════════
# 3. MLP macrostate edges, combined rule, relabelled to MD pocket ranks via the Hungarian matching
# ════════════════════════════════════════════════════════════════════════════
mlp_macro_edges = pickle.load(open(f'{RUN_DIR}/macro_edges_zlink_harmonic.pkl', 'rb'))
A_mlp_raw = np.zeros((n, n))
for _, e in mlp_macro_edges.iterrows():
    i, j = int(e['macro_i']), int(e['macro_j'])
    if i < n and j < n:
        A_mlp_raw[i, j] = A_mlp_raw[j, i] = e['weight']

md_to_zl = {int(i): int(j) for i, j in zip(md_idx, zl_idx)}
perm  = np.array([md_to_zl[i] for i in range(n)])
A_mlp = A_mlp_raw[np.ix_(perm, perm)]


# ════════════════════════════════════════════════════════════════════════════
# 4. the three metrics, for all 24 pockets and the matched 12
# ════════════════════════════════════════════════════════════════════════════
def metrics(idx):
    iu   = np.triu_indices(len(idx), k=1)
    md   = A_md[np.ix_(idx, idx)][iu]                 # every pocket pair in this set
    mlp  = A_mlp[np.ix_(idx, idx)][iu]
    edge = md > 0                                     # pairs MD actually connects
    k    = int(edge.sum())

    # 1. transition Spearman, MD edges only
    rho_trans = spearmanr(md[edge], mlp[edge]).correlation

    # 2. MD flux captured by the MLP's k strongest pairs (tied MLP weights counted at their average)
    cut   = np.sort(mlp)[::-1][k - 1]
    above = mlp > cut
    tied  = mlp == cut
    flux  = (md[above].sum() + (k - above.sum()) * md[tied].mean()) / md.sum()
    in_top = above.astype(float) + tied * (k - above.sum()) / tied.sum()   # 1 = in top k, fraction for ties

    # 3. hub strength Spearman: per pocket, total MD transitions vs total MLP edge weight
    hub_md, hub_mlp = A_md[np.ix_(idx, idx)].sum(1), A_mlp[np.ix_(idx, idx)].sum(1)
    rho_hub = spearmanr(hub_md, hub_mlp).correlation

    return dict(md=md, mlp=mlp, edge=edge, k=k, in_top=in_top, rho_trans=rho_trans, flux=flux,
                flux_random=k / len(md), hub_md=hub_md, hub_mlp=hub_mlp, rho_hub=rho_hub)


sets = [(f'All {n} pockets',     GRAY, metrics(list(range(n)))),
        (f'{m} matched pockets', BLUE, metrics(sub))]

for label, _, r in sets:
    print(f"{label}: transition rho = {r['rho_trans']:.3f} ({r['edge'].sum()} MD edges) | "
          f"MD flux captured = {r['flux']:.3f} (random {r['flux_random']:.3f}) | hub rho = {r['rho_hub']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# 5. plot: three scatter panels (sized for a two-column figure, 180 mm wide)
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

rng = np.random.default_rng(0)
DOT = dict(edgecolor='white', linewidth=0.45)


def dot_style(colour):
    return dict(s=15 if colour == GRAY else 19, color=colour, zorder=3 if colour == BLUE else 2, **DOT)


def stats_legend(ax, entries, **where):
    """per-panel key inside the axes, in headroom above the data: coloured marker (pocket set) + ink text."""
    handles = [Line2D([], [], ls='', marker='o', ms=4.2, mfc=c, mec='white' if c != 'none' else 'none', mew=0.4, label=t)
               for c, t in entries]                              # colour 'none' = text-only line
    leg = ax.legend(handles=handles, frameon=False,
                    handletextpad=0.15, borderpad=0.2, borderaxespad=0.1, labelspacing=0.3, labelcolor=INK,
                    **where)
    leg.set_zorder(5)


NAMES = {GRAY: f'All {n}', BLUE: f'Matched {m}'}


# three separate figures, same size and same margins so the axes are identical when placed side by side
figs, axes = zip(*[plt.subplots(figsize=(2.5, 2.5)) for _ in range(3)])
NAMES_OUT = ['spearman_value_transition', 'spearman_value_flux', 'spearman_value_hub']

# ── a. transition Spearman: one point per MD edge (log x; counts are integers, so a small
#       multiplicative jitter separates stacked points -- it does not change any rank) ──
ax = axes[0]
for label, colour, r in sets:
    md_e, mlp_e = r['md'][r['edge']], r['mlp'][r['edge']]
    ax.scatter(md_e * 10 ** rng.uniform(-0.035, 0.035, md_e.size), mlp_e, **dot_style(colour))
ax.set_xscale('log')
ax.set_xlim(0.8, 110)
ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(-0.03, 0.88)                              # headroom above the data for the key
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_xlabel("MD transitions between pockets")
ax.set_ylabel("MLP edge weight")
stats_legend(ax, [(c, f"{NAMES[c]}:  $\\rho$ = {r['rho_trans']:.2f}  (n = {r['edge'].sum()})") for _, c, r in sets],
             loc='upper left')

# ── b. MD flux captured: one point per MD edge, x = where the MLP ranks that pocket pair ──
#       flux = share of all MD transitions in the MLP's top-k pairs (k = number of MD edges)
ax = axes[1]
for label, colour, r in sets:
    rank = np.argsort(np.argsort(-r['mlp'], kind='stable'), kind='stable') / (len(r['mlp']) - 1)
    ax.scatter(rank[r['edge']], r['md'][r['edge']], **dot_style(colour))
ax.set_xlim(-0.03, 1.03)
ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1'])
ax.set_ylim(-2, 108)                                  # headroom above the data for the key
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_xlabel("MLP rank of pocket pair\n(0 = strongest predicted edge)")
ax.set_ylabel("MD transitions between pockets")
stats_legend(ax, [(c, f"{NAMES[c]}:  flux = {r['flux']:.2f}") for _, c, r in sets]
             + [('none', f"random ranking:  {sets[0][2]['flux_random']:.2f}")],
             loc='upper left')

# ── c. hub strength Spearman: one point per pocket (each set scaled to its own maximum,
#       because a matched pocket's strength only counts edges among the 12) ──
ax = axes[2]
for label, colour, r in sets:
    ax.scatter(r['hub_md'] / r['hub_md'].max(), r['hub_mlp'] / r['hub_mlp'].max(), **dot_style(colour))
ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.36)    # headroom above the data for the key
for setter in (ax.set_xticks, ax.set_yticks):
    setter([0, 0.5, 1])
ax.set_xticklabels(['0', '0.5', '1']); ax.set_yticklabels(['0', '0.5', '1'])
ax.set_xlabel("MD hub strength (rel. to max)")
ax.set_ylabel("MLP hub strength (rel. to max)")
stats_legend(ax, [(c, f"{NAMES[c]}:  $\\rho$ = {r['rho_hub']:.2f}  (n = {len(r['hub_md'])})") for _, c, r in sets],
             loc='upper left')

for ax in axes:
    for s_ in ['top', 'right']:
        ax.spines[s_].set_visible(False)
    ax.tick_params(direction='out')

for fig, name in zip(figs, NAMES_OUT):
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.25, top=0.97)   # fixed (no bbox 'tight'), so all three match
    fig.savefig(f'{fig_dir}/{name}.pdf')
    fig.savefig(f'{fig_dir}/{name}.png', dpi=600)
plt.show()
