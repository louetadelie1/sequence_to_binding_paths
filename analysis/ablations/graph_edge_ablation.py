"""
Edge-rule ablation for the triplet binding-path graph (reviewer control experiment).

The MLP graph in inference_building_macrostates_graph_abltation.py links two triplets
if they share >= 1 residue and weights the link by the harmonic mean of their
predicted populations, 2 p_i p_j / (p_i + p_j). Here we take that rule apart:

    (1) population-only : harmonic mean for EVERY pair, overlap ignored
    (2) overlap-only    : fraction of the triplet shared: 1/3 (1 residue), 2/3 (2 residues), else 0
    (3) combined        : shared fraction * harmonic mean (0 for pairs that share no residue)

(an earlier version used a 1/0 overlap instead of 1/3, 2/3 -- its results are kept in
 tables/edge_rule_ablation_binary_overlap.csv)

All three are scored on the SAME candidate pairs against the SAME MD truth set:
triplet pairs i != j with a nonzero transition count in the MSM (either direction,
self-transitions excluded). Candidate pairs = all pairs among the top-N predicted
(p > 0) triplets that also appear in the MD.

Metrics: ROC-AUC, PR-AUC (average precision), precision at k = number of MD edges,
MD flux captured at that k, and Spearman(weight, MD count) among the real MD edges.
Overlap-only has only three weight values (lots of ties), so its ROC is two points --
precision / flux at k use the expected value under random tie-breaking.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 7
rcParams['axes.labelsize'] = 7
rcParams['axes.titlesize'] = 7
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 6

# ── colors -- same palette as comaring_mlp_to_md_macrostates.py ──
BLUE, RED, GREEN, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#c3c2b7'
GRAPH_COLORS = {'population-only': RED, 'overlap-only': GREEN, 'combined': BLUE}


def despine(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

K_NN       = 40
BATCH_SIZE = 4096
WEIGHTS    = '/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments/exp_5_boltz_distances_inf_pred/weights/best_model_change_ranking.pt'
base       = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps/abeta/ZINC000000030986'
out_dir    = '/ptmp/adlouet/camb/sequence_to_binding_paths/predicting_paths/run_msm_macro_micro'
fig_dir    = f'{out_dir}/figures'
table_dir  = f'{out_dir}/tables'
MD_PKL_DIR = ('/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/'
              'community_identification/pickled_files/abeta_gabis')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

TOP_NS        = [50, 100, 150, 200, 300, 400, 500, 750, 1000, None]   # None = every p > 0 triplet that is also in the MD
MAIN_TOP_N    = None                    # which candidate set goes in the ROC / PR figure
MIN_MD_COUNT  = 1                       # MD edge = transition count >= this (either direction)
MIN_SHARED    = 1                       # overlap rule: share >= this many residues (same as Gz)


# ════════════════════════════════════════════════════════════════════════════
# 1. MLP triplet populations (same model / inputs as the ablation script)
# ════════════════════════════════════════════════════════════════════════════
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


tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
model = Model().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

seq = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
L   = len(seq)
inputs = tokeniser(seq, return_tensors="pt", truncation=True, max_length=1024)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    embeddings = esm_model(**inputs).last_hidden_state.squeeze(0).detach().cpu()

_loaded = pickle.load(open(f'{base}/ca_dist_matrix_boltz.pkl', 'rb'))
ca_dist_matrix = _loaded['matrix'] if isinstance(_loaded, dict) and 'matrix' in _loaded else _loaded

n_res = ca_dist_matrix.shape[0]
candidate_triplets = set()
for i in range(n_res):
    neighbors = np.argsort(ca_dist_matrix[i])[1:K_NN + 1]
    for j, l in combinations(neighbors, 2):
        candidate_triplets.add(tuple(sorted((i, int(j), int(l)))))
candidate_triplets = list(candidate_triplets)

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

preds = []
with torch.no_grad():
    for start in range(0, len(all_x1), BATCH_SIZE):
        x1_b  = torch.stack(all_x1[start:start + BATCH_SIZE]).to(device)
        x2_b  = torch.stack(all_x2[start:start + BATCH_SIZE]).to(device)
        x3_b  = torch.stack(all_x3[start:start + BATCH_SIZE]).to(device)
        pos_b = torch.stack(all_pos[start:start + BATCH_SIZE]).to(device)
        preds.extend(model(x1_b, x2_b, x3_b, pos_b).squeeze(-1).cpu().tolist())

ranked_triplets = sorted(zip(valid_triplets, preds), key=lambda t: t[1], reverse=True)
peq_real = {t: p for t, p in ranked_triplets if p > 0}          # same node set as the MLP graph
print(f"MLP: {len(ranked_triplets):,} scored triplets, {len(peq_real):,} with p > 0")


# ════════════════════════════════════════════════════════════════════════════
# 2. MD truth: triplet pairs i != j with a nonzero MSM transition count
# ════════════════════════════════════════════════════════════════════════════
filtered_merged = pickle.load(open(f'{MD_PKL_DIR}/filtered_merged.pckl', 'rb'))
P_eq_keys       = pickle.load(open(f'{MD_PKL_DIR}/p_eq_keys.pckl', 'rb'))

md_triplets = {tuple(int(r) for r in t) for t in P_eq_keys}
md_counts = {}                                                    # undirected pair -> summed count (both directions)
for (node1, node2), weight in filtered_merged:
    a, b = tuple(int(r) for r in node1), tuple(int(r) for r in node2)
    md_triplets |= {a, b}
    if a == b:
        continue                                                  # self-transitions are not edges
    key = (min(a, b), max(a, b))
    md_counts[key] = md_counts.get(key, 0.0) + float(weight)
md_edges = {k for k, c in md_counts.items() if c >= MIN_MD_COUNT}
print(f"MD: {len(md_triplets):,} triplets, {len(md_edges):,} undirected i!=j edges")



# ════════════════════════════════════════════════════════════════════════════
# 3. the three edge rules, on the same triplet pairs
# ════════════════════════════════════════════════════════════════════════════
# keep the MLP triplets (p > 0) that also exist in the MD, most populated first
triplets_both = [t for t, p in ranked_triplets if p > 0 and t in md_triplets]

results = []

for top_n in TOP_NS:
    if top_n is None:
        nodes = triplets_both
        tag = 'all'
    else:
        nodes = triplets_both[:top_n]
        tag = str(top_n)

    # one row per triplet pair: is it an MD edge? + the weight each graph gives it
    is_md_edge, md_count, n_shared, weight_pop, weight_overlap, weight_combined = [], [], [], [], [], []
    for a, b in combinations(nodes, 2):
        harmonic_mean = 2 * peq_real[a] * peq_real[b] / (peq_real[a] + peq_real[b])
        shared        = len(set(a) & set(b))                          # 0, 1 or 2 residues in common
        share         = shared / 3 if shared >= MIN_SHARED else 0.0   # 0, 1/3 or 2/3
        n_shared.append(shared)

        is_md_edge.append((min(a, b), max(a, b)) in md_edges)
        md_count.append(md_counts.get((min(a, b), max(a, b)), 0.0))   # how many times MD crossed it (0 = never)
        weight_pop.append(harmonic_mean)               # (1) population-only
        weight_overlap.append(share)                   # (2) overlap-only
        weight_combined.append(share * harmonic_mean)  # (3) combined = what Gz uses

    is_md_edge      = np.array(is_md_edge)
    md_count        = np.array(md_count)
    n_shared        = np.array(n_shared)
    weight_pop      = np.array(weight_pop)
    weight_overlap  = np.array(weight_overlap)
    weight_combined = np.array(weight_combined)

    n_pairs    = len(is_md_edge)
    n_md       = is_md_edge.sum()                     # number of true MD edges among these pairs
    n_overlap  = int(weight_overlap.sum())            # number of edges in the overlap graph
    base_rate  = n_md / n_pairs                       # precision of a random guess
    print(f"\n── top-N = {tag}: {len(nodes)} triplets, {n_pairs:,} pairs, "
          f"{n_md:,} MD edges (random precision {base_rate:.4f}) ──")

    # precision at k = n_md: keep the n_md highest-weighted pairs, what fraction are MD edges?
    # population-only and combined: just sort by weight.
    top_pop      = np.argsort(-weight_pop)[:n_md]
    top_combined = np.argsort(-weight_combined)[:n_md]
    prec_pop      = is_md_edge[top_pop].mean()
    prec_combined = is_md_edge[top_combined].mean()
    # overlap-only: weights are only 2/3, 1/3 or 0, so lots of ties. Its top n_md pairs are:
    # first the "share 2" pairs, then (if there is room left) random "share 1" pairs.
    # Random picks from a group give, on average, that group's MD-edge fraction.
    share_two = n_shared == 2
    share_one = n_shared == 1
    n_two     = share_two.sum()
    if n_md <= n_two:
        # all n_md picks come from the "share 2" group
        prec_overlap = is_md_edge[share_two].mean()
        flux_overlap_sum = n_md * md_count[share_two].mean()
    else:
        # every "share 2" pair, plus (n_md - n_two) random "share 1" pairs
        n_rest = n_md - n_two
        prec_overlap = (is_md_edge[share_two].sum() + n_rest * is_md_edge[share_one].mean()) / n_md
        flux_overlap_sum = md_count[share_two].sum() + n_rest * md_count[share_one].mean()

    # ── count-aware: use HOW OFTEN MD crossed each edge, not just yes/no ──
    # flux captured at k = n_md: of all MD transitions between these triplets, what
    # fraction lie on the graph's n_md highest-weighted edges? (an edge crossed 388x
    # counts 388x more than an edge crossed once)
    total_flux    = md_count.sum()
    flux_pop      = md_count[top_pop].sum() / total_flux
    flux_combined = md_count[top_combined].sum() / total_flux
    flux_overlap  = flux_overlap_sum / total_flux                  # same tie handling as precision above
    # a random graph: n_md random pairs carry n_md / n_pairs of the flux
    random_flux   = n_md / n_pairs

    # Spearman among the real MD edges only: higher weight <-> more MD transitions?
    # (overlap-only weights are only 0, 1/3, 2/3, so its rho says: are MD edges that share
    #  more residues crossed more often?)
    rho_pop      = spearmanr(weight_pop[is_md_edge],      md_count[is_md_edge]).correlation
    rho_combined = spearmanr(weight_combined[is_md_edge], md_count[is_md_edge]).correlation
    rho_overlap  = spearmanr(weight_overlap[is_md_edge],  md_count[is_md_edge]).correlation

    print(f"  random graph: precision {base_rate:.4f}   flux captured {random_flux:.4f}")
    for graph, weights, prec, flux, rho in [('population-only', weight_pop,      prec_pop,      flux_pop,      rho_pop),
                                            ('overlap-only',    weight_overlap,  prec_overlap,  flux_overlap,  rho_overlap),
                                            ('combined',        weight_combined, prec_combined, flux_combined, rho_combined)]:
        roc_auc = roc_auc_score(is_md_edge, weights)
        pr_auc  = average_precision_score(is_md_edge, weights)
        print(f"  {graph:16s} ROC-AUC {roc_auc:.3f}   PR-AUC {pr_auc:.4f}   precision@{n_md} {prec:.4f}   "
              f"flux captured@{n_md} {flux:.4f}   Spearman(weight, MD count | MD edges) {rho:.3f}")
        results.append({'top_n': tag, 'graph': graph, 'n_triplets': len(nodes), 'n_pairs': n_pairs,
                        'n_md_edges': n_md, 'random_precision': base_rate, 'random_flux': random_flux,
                        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'precision_at_n_md': prec,
                        'flux_captured_at_n_md': flux, 'spearman_weight_vs_md_count': rho})

    # do populations help ON TOP of overlap? only look at the pairs that share a residue
    inside = weight_overlap > 0
    # (use the plain harmonic mean here, so the 1/3 vs 2/3 factor doesn't leak in)
    roc_auc_inside = roc_auc_score(is_md_edge[inside], weight_pop[inside])
    print(f"  only overlapping pairs: harmonic-mean ROC-AUC {roc_auc_inside:.3f} "
          f"(0.5 = populations add nothing beyond overlap)")

    # MD edges between triplets with no shared residue: graphs (2) and (3) can never find them
    n_unreachable = (is_md_edge & ~inside).sum()
    print(f"  MD edges with no shared residue: {n_unreachable} / {n_md}")

    # keep the full set ('all') for the ROC / PR figure
    if top_n == MAIN_TOP_N:
        plot_is_md_edge = is_md_edge
        plot_weights    = {'population-only': weight_pop,
                           'overlap-only':    weight_overlap,
                           'combined':        weight_combined}
        plot_base_rate  = base_rate
        plot_tag        = tag

results = pd.DataFrame(results)
results.to_csv(f'{table_dir}/edge_rule_ablation.csv', index=False)
print(f"\nwrote {table_dir}/edge_rule_ablation.csv")


# ════════════════════════════════════════════════════════════════════════════
# 4. figures
# ════════════════════════════════════════════════════════════════════════════
GRAPHS = ['population-only', 'overlap-only', 'combined']
LABELS = {'population-only': '(1) population only',
          'overlap-only':    '(2) overlap only',
          'combined':        '(3) combined'}

# ROC and precision-recall curves for the main candidate set
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(4.6, 2.2))
for graph in GRAPHS:
    row = results[(results.top_n == plot_tag) & (results.graph == graph)].iloc[0]
    fpr, tpr, _   = roc_curve(plot_is_md_edge, plot_weights[graph])
    prec, rec, _  = precision_recall_curve(plot_is_md_edge, plot_weights[graph])

    if graph == 'overlap-only':
        # weights are only 0, 1/3, 2/3 -> two points (share 2, then share >= 1), not a curve
        ax_roc.plot(fpr, tpr, ls=':', marker='o', ms=3.5, color=GRAPH_COLORS[graph], lw=0.8,
                    label=f"{LABELS[graph]} ({row.roc_auc:.2f})")
        ax_pr.plot(rec[1:-1], prec[1:-1], ls=':', marker='o', ms=3.5, color=GRAPH_COLORS[graph], lw=0.8,
                   label=f"{LABELS[graph]} ({row.pr_auc:.3f})")
    else:
        ax_roc.plot(fpr, tpr, color=GRAPH_COLORS[graph], lw=1.0,
                    label=f"{LABELS[graph]} ({row.roc_auc:.2f})")
        ax_pr.plot(rec, prec, color=GRAPH_COLORS[graph], lw=1.0,
                   label=f"{LABELS[graph]} ({row.pr_auc:.3f})")

ax_roc.plot([0, 1], [0, 1], ls='--', color=GRAY, lw=0.7)      # random
ax_pr.axhline(plot_base_rate, ls='--', color=GRAY, lw=0.7)     # random
ax_roc.set_xlabel('False positive rate'); ax_roc.set_ylabel('True positive rate')
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.set_yscale('log')
ax_roc.legend(frameon=False, loc='lower right', title='ROC-AUC', title_fontsize=6)
ax_pr.legend(frameon=False, loc='upper right', title='PR-AUC', title_fontsize=6)
despine(ax_roc); despine(ax_pr)
fig.tight_layout()
fig.savefig(f'{fig_dir}/edge_rule_ablation_roc_pr.pdf', bbox_inches='tight')
fig.savefig(f'{fig_dir}/edge_rule_ablation_roc_pr.png', dpi=300, bbox_inches='tight')

# every metric vs. how many top-predicted triplets go into the candidate set
fig2, axes2 = plt.subplots(2, 2, figsize=(4.6, 4.0))
panels = [(axes2[0, 0], 'pr_auc',                      'PR-AUC',                          'random_precision'),
          (axes2[0, 1], 'precision_at_n_md',           'Precision at k = |MD edges|',     'random_precision'),
          (axes2[1, 0], 'flux_captured_at_n_md',       'MD flux captured at k = |MD edges|', 'random_flux'),
          (axes2[1, 1], 'spearman_weight_vs_md_count', 'Spearman(weight, MD count)\namong MD edges', None)]
for ax, column, ylabel, random_column in panels:
    for graph in GRAPHS:
        rows = results[results.graph == graph]
        ax.plot(rows['n_triplets'], rows[column], '-o', ms=2.5, lw=1.0,
                color=GRAPH_COLORS[graph], label=LABELS[graph])
    if random_column is not None:
        rows = results[results.graph == 'combined']
        ax.plot(rows['n_triplets'], rows[random_column], ls='--', color=GRAY, lw=0.8, label='random')
    else:
        ax.axhline(0, ls='--', color=GRAY, lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Top-N predicted triplets')
    ax.set_ylabel(ylabel)
    despine(ax)
axes2[0, 0].legend(frameon=False)
fig2.tight_layout()
fig2.savefig(f'{fig_dir}/edge_rule_ablation_vs_top_n.pdf', bbox_inches='tight')
fig2.savefig(f'{fig_dir}/edge_rule_ablation_vs_top_n.png', dpi=300, bbox_inches='tight')
plt.show()
