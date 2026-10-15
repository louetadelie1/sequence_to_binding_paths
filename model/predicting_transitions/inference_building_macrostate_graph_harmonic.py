

from transformers import AutoModel, AutoTokenizer
import torch
import glob
import pickle
import linecache
import numpy as np
from collections import defaultdict
# from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations
import re
from itertools import combinations
from scipy.stats import spearmanr
import random
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
import networkx as nx
import math
import itertools
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['figure.dpi'] = 600
rcParams['savefig.dpi'] = 600
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TOP_ACTIVES = 10
K_NN       = 40 #50
BATCH_SIZE = 4096
# N_SHUFFLES = 100
WEIGHTS      = '/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments/exp_5_boltz_distances_inf_pred/weights/best_model_change_ranking.pt'
# BASE       = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps'
tokeniser  = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model  = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)


# base = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_probes'
base='//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps/abeta/ZINC000000030986'
fig_dir='/ptmp/adlouet/camb/sequence_to_binding_paths/predicting_paths/run_msm_macro_micro/figures'

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout    = torch.nn.Dropout(0.5)
        self.layer1     = torch.nn.Linear(480 * 3 + 10, 512)
        self.ln1        = torch.nn.LayerNorm([512])
        self.layer2     = torch.nn.Linear(512, 128)
        self.ln2        = torch.nn.LayerNorm([128])
        self.layer3     = torch.nn.Linear(128, 32)
        self.ln3        = torch.nn.LayerNorm([32])
        self.regression_head = torch.nn.Linear(32, 1)

    def forward(self, x1, x2, x3, pos):
        x = torch.cat([x1, x2, x3, pos], dim=-1)
        x = self.dropout(self.leaky_relu(self.ln1(self.layer1(x))))
        x = self.dropout(self.leaky_relu(self.ln2(self.layer2(x))))
        x = self.dropout(self.leaky_relu(self.ln3(self.layer3(x))))
        return self.regression_head(x)


def clean(input):
    return {tuple(int(i) for i in key): float(value[0]) for key, value in input.items()}


model = Model().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
print("Loaded weights from exp_18")

os.makedirs('figures', exist_ok=True)


cmap = plt.cm.tab10


# ── ESM embeddings (once per protein) ────────────────────────────────────
seq = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
ca_dist_file = f'{base}/ca_dist_matrix_boltz.pkl'
prot_name = 'ab'

L   = len(seq)
inputs = tokeniser(seq, return_tensors="pt", truncation=True, max_length=1024)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    embeddings = esm_model(**inputs).last_hidden_state.squeeze(0).detach().cpu()
n_emb = embeddings.shape[0]

# ── distance matrix ───────────────────────────────────────────────────────
# ca_dist_file = ref_protein.removesuffix('pickled_files/p_eq_keys.pckl') + 'ca_dist_matrix.pkl'


_loaded = pickle.load(open(ca_dist_file, 'rb'))
ca_dist_matrix = _loaded['matrix'] if isinstance(_loaded, dict) and 'matrix' in _loaded else _loaded
_labels = _loaded.get('labels', None) if isinstance(_loaded, dict) else None

# ── kNN candidate triplets ────────────────────────────────────────────────
n_res = ca_dist_matrix.shape[0]
candidate_triplets = set()
for i in range(n_res):
    neighbors = np.argsort(ca_dist_matrix[i])[1:K_NN + 1]
    for j, l in combinations(neighbors, 2):
        candidate_triplets.add(tuple(sorted((i, int(j), int(l)))))
candidate_triplets = list(candidate_triplets)
print(f"{prot_name}: {n_res} residues, {len(candidate_triplets):,} kNN candidates (k={K_NN})")

# ── build input tensors ───────────────────────────────────────────────────
all_x1, all_x2, all_x3, all_pos, valid_triplets = [], [], [], [], []
for trio in candidate_triplets:
    ri, rj, rk = trio
    if any(r >= L for r in (ri, rj, rk)):
        continue
    dist_ij     = float(ca_dist_matrix[ri, rj])
    dist_jk     = float(ca_dist_matrix[rj, rk])
    dist_ik     = float(ca_dist_matrix[ri, rk])
    min_dist    = min(dist_ij, dist_jk, dist_ik)
    max_dist    = max(dist_ij, dist_jk, dist_ik)
    mean_dist   = (dist_ij + dist_jk + dist_ik) / 3
    compactness = min_dist / max_dist if max_dist > 0 else 0.0
    # embeddings[0] is the <cls> token, so residue i's embedding is at index i+1
    all_x1.append(embeddings[ri + 1])
    all_x2.append(embeddings[rj + 1])
    all_x3.append(embeddings[rk + 1])
    all_pos.append(torch.tensor([
        ri/L, rj/L, rk/L,
        dist_ij, dist_jk, dist_ik,
        min_dist, max_dist, mean_dist, compactness
    ], dtype=torch.float32))
    valid_triplets.append(trio)

# ── batched inference ─────────────────────────────────────────────────────
preds = []
with torch.no_grad():
    for start in range(0, len(all_x1), BATCH_SIZE):
        x1_b  = torch.stack(all_x1[start:start + BATCH_SIZE]).to(device)
        x2_b  = torch.stack(all_x2[start:start + BATCH_SIZE]).to(device)
        x3_b  = torch.stack(all_x3[start:start + BATCH_SIZE]).to(device)
        pos_b = torch.stack(all_pos[start:start + BATCH_SIZE]).to(device)
        preds.extend(model(x1_b, x2_b, x3_b, pos_b).squeeze(-1).cpu().tolist())

ranked_triplets = sorted(zip(valid_triplets, preds), key=lambda t: t[1], reverse=True)

peq_real = {t: p for t, p in ranked_triplets if p > 0}
triplets = list(peq_real)

ratios = {
    # (a, b): min(1.0, peq_real[b] / peq_real[a]) # replcae the metropolis scheme with teh ratio between the two
    (a, b): math.sqrt(peq_real[a]*peq_real[b]) #controls the edge weight, the geometric mean of the two ratios
    for a in triplets
    for b in triplets
    # if a != b and set(a) & set(b)
    if a != b and len(set(a) & set(b)) == 2 #only accept edges if two of the three residues are shared
}


# Z-LINKEAGE CLUSTERING #
# Use spatial/chemical/physical cluistering to combine like triplets into nodes

for rank, (trio, score) in enumerate(ranked_triplets[:50], start=1):
    print(f"{rank:>4}  {trio}  {score:.4f}")

sorted_scores = np.array([s for _, s in ranked_triplets])

zero_cross = int(np.searchsorted(-sorted_scores, 0))  # sorted_scores is descending

ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}
resid_map = {i: (i + 1, ONE_TO_THREE[aa]) for i, aa in enumerate(seq)}

def fmt_triplet(t):
    return tuple(f"{resid_map[i][1]}{resid_map[i][0]}" for i in t)


KYTE_DOOLITTLE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9,
    'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9,
    'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5,
    'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
}
HBOND_DONORS    = {'ARG', 'LYS', 'TRP', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
HBOND_ACCEPTORS = {'ASP', 'GLU', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
CHARGE          = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5, 'ASP': -1.0, 'GLU': -1.0}

n_top = zero_cross
top_triplets = [t for t, _ in ranked_triplets[:n_top]]
top_scores   = np.array([s for _, s in ranked_triplets[:n_top]])
print(f"\nClustering top {n_top} triplets (score > 0, {n_top / len(ranked_triplets):.2%} of {len(ranked_triplets):,} scored)")

def triplet_chem_features(t):
    names = [resid_map[i][1] for i in t]
    hydrophobicity = np.mean([KYTE_DOOLITTLE.get(n, 0.0) for n in names])
    n_donors       = sum(n in HBOND_DONORS for n in names)
    n_acceptors    = sum(n in HBOND_ACCEPTORS for n in names)
    net_charge     = sum(CHARGE.get(n, 0.0) for n in names)
    return np.array([hydrophobicity, n_donors, n_acceptors, net_charge])

chem_raw  = np.stack([triplet_chem_features(t) for t in top_triplets])
chem_z    = (chem_raw - chem_raw.mean(0)) / (chem_raw.std(0) + 1e-12)
chem_dist = squareform(pdist(chem_z, metric='euclidean')) # two triplets are close if they have similar physcial features

spatial_dist = np.zeros((n_top, n_top))
for a in range(n_top):
    for b in range(a + 1, n_top):
        d = np.mean([ca_dist_matrix[i, j] for i in top_triplets[a] for j in top_triplets[b]])
        spatial_dist[a, b] = spatial_dist[b, a] = d

def normalise(mat):
    mx = mat.max()
    return mat / mx if mx > 0 else mat

score_z    = (top_scores - top_scores.mean()) / (top_scores.std() + 1e-12)
score_dist = squareform(pdist(score_z.reshape(-1, 1), metric='euclidean'))

# now adding the transitions
peq_real = {t: p for t, p in ranked_triplets if p > 0}
triplets = list(peq_real)

trans_rate = np.zeros((n_top, n_top))
for a in range(n_top): 
    for b in range(a + 1, n_top):
        aij = 1.0 if set(top_triplets[a]) & set(top_triplets[b]) else 0.0
        pa, pb = top_scores[a], top_scores[b]
        trans_rate[a, b] = aij * min(1.0, pb / pa)   # rate a -> b
        trans_rate[b, a] = aij * min(1.0, pa / pb)   # rate b -> a

W_SCORE = 0.40 #score distance 
W_SPATIAL = 0.60 #spatial distance 
W_CHEM= 0.20 #chemical distance 
W_TRANS=0.0 # transition distances


combined_dist = (W_SPATIAL * normalise(spatial_dist)
                  + W_CHEM  * normalise(chem_dist)
                  + W_SCORE * normalise(score_dist)
                  + W_TRANS * normalise(trans_rate))

np.fill_diagonal(combined_dist, 0.0)

# ── locally-constrained agglomerative merge (identical to inference_pocket.py) ──
Z = linkage(squareform(combined_dist, checks=False), method='average')
MAX_SPAN = 8          # residues per cluster; lower = smaller clusters

n_top_     = len(top_triplets)
dsu_parent = list(range(n_top_))

def find(x):
    while dsu_parent[x] != x:
        dsu_parent[x] = dsu_parent[dsu_parent[x]]
        x = dsu_parent[x]
    return x

leaves_of   = {i: {i} for i in range(n_top_)}
residues_of = {i: set(top_triplets[i]) for i in range(n_top_)}
capped      = set()

for step, (a, b, _dist, _cnt) in enumerate(Z):
    a, b = int(a), int(b)
    new_id = n_top_ + step
    if a in capped or b in capped:
        capped.add(new_id)
        leaves_of[new_id]   = leaves_of[a] | leaves_of[b]
        residues_of[new_id] = residues_of[a] | residues_of[b]
        continue
    merged_residues     = residues_of[a] | residues_of[b]
    leaves_of[new_id]   = leaves_of[a] | leaves_of[b]
    residues_of[new_id] = merged_residues
    if len(merged_residues) > MAX_SPAN:
        capped.add(new_id)
        continue
    ra, rb = find(next(iter(leaves_of[a]))), find(next(iter(leaves_of[b])))
    if ra != rb:
        dsu_parent[ra] = rb

cluster_ids_raw = np.array([find(i) for i in range(n_top_)])

raw_sum = pd.Series(top_scores).groupby(cluster_ids_raw).sum().sort_values(ascending=False)
relabel = {raw_id: rank for rank, raw_id in enumerate(raw_sum.index, start=1)}
cluster_ids = np.array([relabel[c] for c in cluster_ids_raw])
n_macro = int(cluster_ids.max())

macrostates = [[top_triplets[i] for i in np.where(cluster_ids == c)[0]]
               for c in range(1, n_macro + 1)]

macro_peq = [float(np.sum([peq_real[t] for t in nodes])) for nodes in macrostates]

rows = []
for c, (nodes, mpeq) in enumerate(zip(macrostates, macro_peq), start=1):
    residues = sorted({r for t in nodes for r in fmt_triplet(t)},
                      key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
    rows.append({'macrostate': c, 'n_triplets': len(nodes),
                 'sum_peq': mpeq, 'residues': ','.join(residues)})
z_link_df = pd.DataFrame(rows).drop_duplicates(subset="residues", keep="first")  

keep = z_link_df.index.to_numpy()        
macrostates=[macrostates[i] for i in keep]
macro_peq  =[macro_peq[i] for i in keep]

z_link_df = z_link_df.reset_index(drop=True)
z_link_df['macrostate']= np.arange(1, len(z_link_df) + 1)   
z_link_df['sum_peq_percent']=z_link_df['sum_peq'] / z_link_df['sum_peq'].sum() * 100

z_link_df.to_pickle("/ptmp/adlouet/camb/sequence_to_binding_paths/predicting_paths/run_msm_macro_micro/z_link_df.pkl")


#### Now onto linking the macrostates together --> building the edges (defining the transitions between any two triplets, and using these to bridge together whole nodes in the macrostate graphs)

weight_sum_peq = {
    (a, b): math.sqrt(peq_real[a] * peq_real[b])           # geometric mean of the two populations, symmetric
    for a in triplets
    for b in triplets
    if a != b
}

weight_metropolis = {
    (a, b): min(1.0, peq_real[b] / peq_real[a])            # Metropolis acceptance ratio a->b, directed (p(a->b) != p(b->a))
    for a in triplets
    for b in triplets
    if a != b
}

weight_boltzmann_inv = {
    (a, b): max(peq_real[a], peq_real[b]) / min(peq_real[a], peq_real[b])  # inverse ratio: >=1, symmetric, large when populations differ a lot
    for a in triplets
    for b in triplets
    if a != b
}

weight_harmonic_mean = {
    (a, b): 2 * peq_real[a] * peq_real[b] / (peq_real[a] + peq_real[b])  # harmonic mean: "weakest-link" bottleneck weighting, won the edge-config sweep against the real MD network
    for a in triplets
    for b in triplets
    if a != b
}

weight_shared_residues = {
    (a, b): len(set(a) & set(b)) / 3                       # fraction of the triplet shared: 1/3, 2/3 (3/3/self excluded)
    for a in triplets
    for b in triplets
    if a != b and set(a) & set(b)                          # drop pairs with 0 shared residues -> no edge
}

combined_weight = {
    (a, b): weight_shared_residues[(a, b)] * weight_sum_peq[(a, b)]
    for (a, b) in weight_shared_residues
}


# Builting links between z-linkeage macrostates
# triplets are linked if they share >=1 residue, weighted by shared-residue

t2m_zlink = {t: m for m, nodes in enumerate(macrostates) for t in nodes}  
Gz = nx.Graph()
for (a, b), w in weight_shared_residues.items():
    Gz.add_edge(a, b, weight=weight_harmonic_mean[(a, b)])

macro_w_z, macro_n_z = defaultdict(float), defaultdict(int)
for a, b, d in Gz.edges(data=True):
    i, j = t2m_zlink.get(a), t2m_zlink.get(b)
    if i is None or j is None or i == j:
        continue
    key = (min(i, j), max(i, j))
    macro_w_z[key] = max(macro_w_z[key], d['weight'])     
    macro_n_z[key] += 1

zlink_residues = [sorted({r for t in nodes for r in fmt_triplet(t)},
                          key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
                  for nodes in macrostates]

macro_edges_zlink = pd.DataFrame(
    [{'macro_i': i, 'macro_j': j, 'weight': w, 'n_bridges': macro_n_z[(i, j)],
      'residues_i': zlink_residues[i], 'residues_j': zlink_residues[j]}
     for (i, j), w in macro_w_z.items()]
).sort_values('weight', ascending=False).reset_index(drop=True)

GMz = nx.Graph()
GMz.add_nodes_from(range(len(macrostates)))
for _, r in macro_edges_zlink.iterrows():
    GMz.add_edge(int(r.macro_i), int(r.macro_j), weight=float(r.weight))

macro_edges_zlink.to_pickle(f'/ptmp/adlouet/camb/sequence_to_binding_paths/predicting_paths/run_msm_macro_micro/macro_edges_zlink_boltz.pkl')


# ── knee in the z-link macrostate population curve ──────────────────────────
sorted_scores = z_link_df['sum_peq_percent'].sort_values(ascending=False).to_numpy()
ranks = np.arange(len(sorted_scores))

x_norm = ranks / (len(ranks) - 1)
y_norm = (sorted_scores - sorted_scores.min()) / (np.ptp(sorted_scores) + 1e-12)
x0, y0, x1, y1 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
dx, dy = x1 - x0, y1 - y0
chord_len = np.hypot(dx, dy)
dist_to_chord = np.abs(dy * (x_norm - x0) - dx * (y_norm - y0)) / chord_len
knee_idx = int(np.argmax(dist_to_chord))
print(f"\nKnee (max chord distance) at rank {knee_idx} / {len(sorted_scores):,} "
      f"(top {knee_idx / len(sorted_scores):.2%}), score={sorted_scores[knee_idx]:.4f}")

zero_cross = int(np.searchsorted(-sorted_scores, 0))  # sorted_scores is descending
# (degenerate here: sum_peq_percent is a % of a sum of positive p_eq values, so
# it's never negative -> zero_cross always equals len(sorted_scores))

ZOOM_N = min(100, len(sorted_scores))  
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, n in zip(axes, [len(sorted_scores), ZOOM_N]):
    ax.plot(ranks[:n], sorted_scores[:n], color='steelblue', lw=1.6)
    if knee_idx < n:
        ax.axvline(knee_idx, color='crimson', ls='--', lw=1.2, label=f'knee (rank {knee_idx})')
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(frameon=False, fontsize=14)
axes[0].set_title(f'Full curve (n={len(sorted_scores):,})', fontsize=15)
axes[1].set_title(f'Zoomed to top {ZOOM_N:,} ranks', fontsize=15)
fig.text(0.5, 0.02, 'Rank (descending macrostate population)', ha='center', va='center', fontsize=17)
fig.text(0.01, 0.5, 'sum_peq (% of total)', ha='center', va='center', rotation='vertical', fontsize=17)
sns.despine(fig=fig)
fig.tight_layout(rect=[0.03, 0.05, 1, 1])
fig.savefig('figures/rank_score_curve_zlink.png', dpi=200)
plt.show()



#################################### Z-LINK MACROSTATE NETWORK PLOT ################################################

label  = 'Aβ42'
cmap  = LinearSegmentedColormap.from_list("abeta_blue", ["#d1e5f0", "#2166ac"])  # light -> dark blue


TOP_N  = knee_idx
n_show = len(macrostates) if TOP_N is None else min(TOP_N, len(macrostates))
keep   = list(range(n_show))  

macro_frac = np.asarray(macro_peq, dtype=float)
macro_frac = macro_frac / macro_frac.sum() if macro_frac.sum() > 0 else macro_frac #normalizing i think

# keep only the top nodes and the edges that run between two of them
GMz_plot = GMz.subgraph(keep).copy()
GMz_plot.remove_edges_from(list(nx.selfloop_edges(GMz_plot)))

pos = nx.spring_layout(nx.cycle_graph(n_show), scale=100, seed=2)

color_cycle = itertools.cycle([cmap(i) for i in np.linspace(0.3, 1, n_show)])

def sigmoid_magnification(val, A, k, x0):
    return A / (1 + math.exp(-k * (val - x0)))

plt.figure(figsize=(3.5, 3.5))
for nodes, color in zip([[m] for m in keep], color_cycle):
    sizes_shared   = [macro_frac[n] for n in nodes]
    sigmoid_values = [(sigmoid_magnification(val, 1000, 200, 0.025)) / 1.2 for val in sizes_shared]
    nx.draw_networkx_nodes(
        GMz_plot,
        pos=pos,
        nodelist=nodes,
        node_color=[color],
        node_size=sigmoid_values,
        alpha=0.85,
        edgecolors="none",
    )

edges  = list(GMz_plot.edges())
edge_w = np.array([GMz_plot[u][v]['weight'] for u, v in edges])

if len(edge_w):
    log_w     = np.log1p(edge_w)
    lo, hi    = log_w.min(), log_w.max()
    norm      = (log_w - lo) / (hi - lo) if hi > lo else np.ones_like(log_w)
    width     = 0.4 + 5.0 * norm
    alpha     = 0.08 + 0.5 * norm
    edge_rgba = [(0.4, 0.4, 0.4, a) for a in alpha]
else:
    width, edge_rgba = [], []

nx.draw_networkx_edges(
    GMz_plot,
    pos=pos,
    edgelist=edges,        
    edge_color=edge_rgba,  
    width=list(width),
    connectionstyle="arc3,rad=0.1",
)

ax = plt.gca()
ax.set_axis_off()
for sp in ax.spines.values():
    sp.set_visible(False)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))
plt.subplots_adjust(top=0.85)
# ax.set_title(label, fontsize=8, pad=4)
plt.tight_layout()
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, "ab_macrostates_zlink_harmonic.pdf"), format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()


# TRIPLET-CLUSTER t-SNE EMBEDDING # this is plotting for visual purposes- allows to see how distinct clusters are


N_HIGHLIGHT = min(20, n_macro)   

cluster_ids_mds = cluster_ids  
top_scores_mds  = top_scores

score_floor = float(top_scores_mds.min())
score_span  = float(top_scores_mds.max() - score_floor) + 1e-12

PRIMARY, SECONDARY, TERTIARY = '#2a78d6', '#eb6834', '#1baf7a'
OTHER_COLOR = '#c3c2b7'   # BASELINE_GRAY

from sklearn.manifold import MDS as _MDS
_brand_wheel = LinearSegmentedColormap.from_list(
    "brand_wheel", [PRIMARY, SECONDARY, TERTIARY, PRIMARY])   # cyclic: 4th stop repeats the 1st

member_idx = {c: np.where(cluster_ids_mds[:n_top] == c)[0] for c in range(1, N_HIGHLIGHT + 1)}
_cluster_dist = np.zeros((N_HIGHLIGHT, N_HIGHLIGHT))
for _a in range(1, N_HIGHLIGHT + 1):
    for _b in range(_a + 1, N_HIGHLIGHT + 1):
        if len(member_idx[_a]) and len(member_idx[_b]):
            _d = combined_dist[np.ix_(member_idx[_a], member_idx[_b])].mean()
            _cluster_dist[_a - 1, _b - 1] = _cluster_dist[_b - 1, _a - 1] = _d

_mds_coords = _MDS(n_components=2, dissimilarity='precomputed', random_state=2,
                    normalized_stress='auto').fit_transform(_cluster_dist)


def _norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


_hue_pos = _norm01(_mds_coords[:, 0])         

CLUSTER_COLORS = [_brand_wheel(h) for h in _hue_pos]   

_perplexity = min(30, max(5, n_top // 4 - 1))
tsne_coords = TSNE(n_components=2, metric='precomputed', init='random',
                    perplexity=_perplexity, random_state=2).fit_transform(combined_dist)

embedding_plots = [
    (N_HIGHLIGHT, 'figures/top_triplet_clusters_tsne.png',       10, 60, 20, 60),
    (4,           'figures/top_triplet_clusters_tsne_top4.png',   6, 30, 30, 90),   
]

for n_highlight, out_path, other_size_base, other_size_range, main_size_base, main_size_range in embedding_plots:
    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    below_cutoff_mask = cluster_ids_mds > n_highlight
    if below_cutoff_mask.any():
        other_sizes = other_size_base + other_size_range * (top_scores_mds[below_cutoff_mask] - score_floor) / score_span
        ax.scatter(tsne_coords[below_cutoff_mask, 0], tsne_coords[below_cutoff_mask, 1], s=other_sizes, color=OTHER_COLOR,
                   edgecolor='none', alpha=0.25, zorder=1,
                   label='other clusters')

    for cid in range(1, n_highlight + 1):
        cluster_mask = cluster_ids_mds == cid
        if not cluster_mask.any():
            continue
        cluster_sizes = main_size_base + main_size_range * (top_scores_mds[cluster_mask] - score_floor) / score_span
        ax.scatter(tsne_coords[cluster_mask, 0], tsne_coords[cluster_mask, 1], s=cluster_sizes, color=CLUSTER_COLORS[cid - 1],
                   edgecolor='white', linewidth=0.6, alpha=0.9, zorder=2)

    ax.set_xlabel('t-SNE dim 1',size=10)
    ax.set_ylabel('t-SNE dim 2',size=10)
    # ax.set_title(f'triplet clusters (t-SNE, top {n_highlight})')
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()


from sklearn.cluster import KMeans

print("\n--- split-cluster diagnostic ---")
for cid in range(1, N_HIGHLIGHT + 1):
    idx = np.where(cluster_ids_mds[:n_top] == cid)[0]
    if len(idx) < 6:
        continue
    pts = tsne_coords[idx]
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(pts)
    c0, c1 = km.cluster_centers_
    centroid_dist = np.linalg.norm(c0 - c1)
    spread0 = pts[km.labels_ == 0].std(axis=0).mean() if (km.labels_ == 0).any() else 0
    spread1 = pts[km.labels_ == 1].std(axis=0).mean() if (km.labels_ == 1).any() else 0
    within = max(spread0, spread1, 1e-6)
    if centroid_dist > 3 * within and min((km.labels_ == 0).sum(), (km.labels_ == 1).sum()) >= 3:
        print(f"\ncluster {cid}: SPLIT (centroid dist {centroid_dist:.1f} vs within-group spread {within:.1f})")
        for sub in (0, 1):
            members = [top_triplets[i] for i in idx[km.labels_ == sub]]
            residues = sorted({r for t in members for r in fmt_triplet(t)},
                               key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
            print(f"  sub-group {sub} (n={len(members)}): residues involved = {residues}")
