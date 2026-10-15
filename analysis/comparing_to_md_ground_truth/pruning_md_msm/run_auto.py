import sys, os, pickle, math, itertools
import numpy as np
import matplotlib
# matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
import networkx as nx
import itertools
import math
sys.path.insert(0, "/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM")
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
from collections import Counter
from MDAnalysis.lib.util import convert_aa_code

# High-quality figure settings — Science Advances style
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

sys.path.insert(0, os.getcwd())

from community_identification.population_equilibrium import *
from community_identification.clustering_uplets import *

# the popualtion from unbound is msm_full_model_paper_github/community_identification/p_eq_weighted_w_kd_unbound.ipynb

# we will be including the following for the publication:
proteins =['abeta_gabis']#,'alpha_syn_lig_12','full_alpha_syn_fau']#,'abeta_gabis',]
def sigmoid_magnification(val, A, k, x0):
    return A / (1 + math.exp(-k * (val - x0)))


plot_data = []

for protein_name in proteins:
    try:
        print(f"Processing {protein_name}...")


        # === Protein-specific settings == #
        if protein_name == 'abeta_gabis':
            pdb = '/ptmp/adlouet/camb/sequence_to_binding_paths/original_msm_model/community_identification/pickled_files/template_G5.pdb'
            label='A\u03B242 Ligand G5'
            distances_com = pickle.load(open(f'/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/abeta_pickled_files/d_24_t_com_avg.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/abeta_pickled_files/d_24_t_closest.pkl', 'rb'))
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold= 0.6
            uplet_type=3 #this is only for visauls, change it back for pop equi
            w_com = 0.7 #0.85 #this is only for visauls, change it back for pop equi
            w_closest =  0.3 #0.15 #this is only for visauls, change it back for pop equi
            pos_scaling=False

        elif protein_name == 'alpha_syn_lig_40':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_40/ligand_40_alpha_syn_c_term.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_40/distances/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_40/distances/t_resnum_closest.pkl', 'rb'))
            w_com = 0.2
            w_closest = 0.8
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=0.6
            uplet_type=5
            label='C-term \u03B1-syn Ligand 40' 
            pos_scaling=True

        elif protein_name == 'alpha_syn_lig_50':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_50/ligand_50_alpha_syn_c_term.pdb'
            # xtc = '/Users/adelielouet/Documents/science/medin/urea/urea_post_run/concatenate_urea.xtc'
            # resname_ligand = "1UNL"
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_50/distances/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/alpha_syn_lig_50/distances/t_resnum_closest.pkl', 'rb'))
            w_com = 0.2
            w_closest = 0.8
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=0.6
            uplet_type=5
            label='C-term \u03B1-syn Ligand 50' 
            pos_scaling=True

        elif protein_name == 'alpha_syn_lig_12':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_12/DESRES-Trajectory_jacs2022-5447858-no-water-glue/jacs2022-5447858-no-water-glue/lig_12_cterm.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_12/DESRES-Trajectory_jacs2022-5447858-no-water-glue/jacs2022-5447858-no-water-glue/t_resnum_com.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/alpha_syn/DE_SHAW/lig_12/DESRES-Trajectory_jacs2022-5447858-no-water-glue/jacs2022-5447858-no-water-glue/t_resnum_closest.pkl', 'rb'))
            w_com = 0.2
            w_closest = 0.8
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=0.6
            uplet_type=5
            label='C-term \u03B1-syn Ligand 12' 
            pos_scaling=True

        elif protein_name == 'full_alpha_syn_fau':
            pdb = '/Users/adelielouet/Documents/science/alpha_syn/pbmetad/DE_SHAW_traj/system.pdb'
            distances_com = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/full_alpha_syn_fau/d_24_t.pkl', 'rb'))
            distances_closest = pickle.load(open(f'/Users/adelielouet/Documents/science/dd_proj/msm_full_model_paper/pickled_files/full_alpha_syn_fau/d_24_t_closest.pkl', 'rb'))
            w_com = 0.0
            w_closest = 1.0
            w_eq=[1.0] *len(distances_com)   # this is for systems that don't need to be reweighted, assign each frame a weight of 1.0
            combined_threshold=1.0
            uplet_type=5
            label='Full Length \u03B1-syn Fasudil' 
            pos_scaling=True

        # === Create directories ===
        os.makedirs(f"/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/community_identification/pickled_files/{protein_name}", exist_ok=True)
        fig_dir = f"/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/community_identification/figures"
        os.makedirs(fig_dir, exist_ok=True)
        print(len(distances_closest))
        # === Population at equilibrium ===
        x_normed, filtered_keys,Kd_kinetic_weighted, Kd_pop_weighted,transition_matrix = transition_matrix_custom(
            pdb, distances_com, distances_closest, w_file=None, n_reps=None, 
            trim_fraction=None, combined_threshold=combined_threshold, w_com=w_com, w_closest=w_closest,uplet_type=uplet_type
        )

        equilibrium_matrix, P_eq, P_eq_keys = solving_states_at_equilirum(x_normed, filtered_keys)
        dictionary_transitions_sorted, filtered_merged_output = kd_dictionary(x_normed, filtered_keys,transition_matrix) # rather than x_normed,filtered_keys

        # === Clustered Uplets ===
        if protein_name.split('_')[0] == 'alpha':
            resolution=2

        elif protein_name.split('_')[0] == 'full':
            resolution=3

        elif protein_name.split('_')[0] == 'abeta':
            resolution=8

        parts, G, pos, values, communities,betCent = network_graph_microstates(filtered_merged_output,resolution=resolution)

        #pruned section
        original_res = mda.Universe(pdb).select_atoms("protein").residues
        original_res_lab = lambda i: f"{convert_aa_code(original_res[i].resname)}{original_res[i].resid}"  

        clean_peq = {t: float(np.ravel(P_eq_keys[t])[0]) for t in P_eq_keys}         

        core_by_macrostate = []
        for _comm in communities:
            members = [tuple(t) for t in _comm]
            pop = sum(clean_peq[t] for t in members)                              
            resids = {r for t in members for r in t}
            w = {r: sum(clean_peq[t] for t in members if r in t) for r in resids}    
            core = sorted((r for r in w if w[r] / pop >= 0.15),               
                        key=lambda r: w[r], reverse=True)                   
            core_by_macrostate.append([original_res_lab(r) for r in core])

        total = sum(clean_peq.values())
        rows = []
        total = sum(clean_peq.values())
        rows = []
        for k, _comm in enumerate(communities):
            members= [tuple(t) for t in _comm]
            pop= sum(clean_peq[t] for t in members)
            resids= {r for t in members for r in t}
            w= {r: sum(clean_peq[t] for t in members if r in t) for r in resids}
            core= sorted((r for r in w if w[r] / pop >= 0.1), key=lambda r: w[r], reverse=True)
            rows.append({
                "macrostate": k,
                "P_eq":       pop,
                "pop_pct":    100 * pop / total,
                "n":          len(members),
                "uniqR":      len(resids),
                "core":       [original_res_lab(r) for r in core],
                "core_P_eq":  [w[r] for r in core],
                "core_frac":  [w[r] / pop for r in core],
            })

        df_core = (pd.DataFrame(rows).sort_values("P_eq", ascending=False).reset_index(drop=True))

        df_core.to_pickle("/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/community_identification/df_core_010.pkl")


        #pruned section ended

        # === Microstates plot ===
        if protein_name.split('_')[0] == 'alpha':
            node_size = [(v * 500)**1.0 for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list("alpha_purple", ["#cab4f2", "#4c17a8"])  # light → dark purple

        elif protein_name.split('_')[0] == 'full':
            node_size = [(v * 500)**1.7 for v in betCent.values()]
          #  cmap = LinearSegmentedColormap.from_list("medin_green", ["#d9f0d3", "#004f18"])  # light → dark green
            cmap = LinearSegmentedColormap.from_list("medin_green", ["#d9f0d3", "#2B0342"])  # light → dark green
 
        elif protein_name.split('_')[0] == 'abeta':
            node_size = [(v * 2000)**1.25 for v in betCent.values()]
            cmap = LinearSegmentedColormap.from_list("abeta_blue", ["#d1e5f0", "#2166ac"])  # light → dark blue
        
        if pos_scaling:
            sigmoid_values_node_mass = {
                k: sigmoid_magnification(v, 1000, 200, 0.025)
                for k, v in P_eq_keys.items()}
            pos_used=nx.forceatlas2_layout(G,gravity=0.00,node_mass=sigmoid_values_node_mass)

        else:
            pos_used = pos
        
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        plt.figure(figsize=(3.5, 3.5))
        ax = plt.gca()
        for spine in ax.spines.values(): spine.set_visible(False)
        nx.draw_networkx_nodes(G, pos=pos_used, node_color=[cmap(norm(v)) for v in values], node_size=node_size, alpha=1.0, linewidths=0.01)
        nx.draw_networkx_edges(G, pos=pos_used, alpha=0.04, width=0.5)
        plt.axis("off")

        ax.text(0.5, 0.98, label, transform=ax.transAxes, fontsize=8, va='top', ha='center')
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{protein_name}_microstates.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f"{fig_dir}/{protein_name}_microstates.png", format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.show()

        # === Macrostates plot ===
        kd_centrality_ordered, inv_map, inv_map_vals, pos = network_graph_macrostates(parts, P_eq_keys, communities, G)
        print(len(kd_centrality_ordered),'number of clusters')
    #    color_cycle = itertools.cycle([plt.cm.Purples(i) for i in np.linspace(0.3, 1, len(inv_map_vals))])
        color_cycle = itertools.cycle([cmap(i) for i in np.linspace(0.3, 1, len(inv_map_vals))])

        plt.figure(figsize=(3.5, 3.5))
        for nodes,color in zip(inv_map_vals,color_cycle):
            kd = [P_eq_keys[node] for node in nodes]
            sizes_shared = [sum(kd)] * len(nodes)
            sigmoid_values = [(sigmoid_magnification(val, 1000, 200, 0.025))/1.2 for val in sizes_shared]
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                nodelist=nodes,
                node_color=[color],
                node_size=sigmoid_values,
                alpha=0.85,
                edgecolors="none"
            )

        G.remove_edges_from(list(nx.selfloop_edges(G)))
        nx.draw_networkx_edges(
            G,
            pos=pos,
            alpha=0.15,
            edge_color="gray",
            width=0.6,
            connectionstyle="arc3,rad=0.1"
        )

        ax = plt.gca()
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))
        plt.subplots_adjust(top=0.85)
        # ax.set_title(label, fontsize=8, pad=4)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{protein_name}_macrostates.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f"{fig_dir}/{protein_name}_macrostates.png", format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        # plt.close()
        plt.show()

        # # # === Sigmoid parameter exploration ===
        # inv_map = {}
        # for k, v in parts.items():
        #     inv_map[v] = inv_map.get(v, []) + [k]

        # inv_map_vals = []
        # for keys, values in inv_map.items():
        #     inv_map_vals.append(values)

        # values = [sum(list(P_eq_keys[node] for node in nodes)) for i, nodes in enumerate(inv_map_vals)]

        # params = [
        #     (1000, 500, 0.025),
        #     (1000, 600, 0.025),
        #     (1000, 400, 0.025),
        #     (1000, 200, 0.025)
        # ]

        # fig, ax = plt.subplots(figsize=(5.5, 3.5))
        # for A, k, x0 in params:
        #     sigmoid_values = [sigmoid_magnification(val, A, k, x0) for val in values]
        #     print(len(sigmoid_values))
        #     ax.scatter(values, sigmoid_values, label=f'A={A}, k={k}, x0={x0}', s=6)

        # ax.set_title("Sigmoid Function with Different Parameters", fontsize=8)
        # ax.set_xlabel("Input value", fontsize=8)
        # ax.set_ylabel("Sigmoid output", fontsize=8)
        # ax.tick_params(labelsize=7)
        # ax.legend(fontsize=7)
        # sns.despine(ax=ax)
        # plt.tight_layout()
        # plt.savefig(f"{fig_dir}/sigmoid_params.png", format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        # plt.close()

        # # === Degree centrality vs equilibrium population ===
        # from collections import defaultdict
        # from matplotlib.ticker import ScalarFormatter

        # deg_centrality = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)
        # metric = deg_centrality
        # metric_dict = {value: key for value, key in metric}
        # dd = defaultdict(list)
        # for d in (metric_dict, P_eq_keys):
        #     for key, value in d.items():
        #         dd[key].append(value)
        # values_first_ten = list(dd.values())
        # del values_first_ten[-1]
        # x = [item[1] for item in values_first_ten]
        # y = [item[0] for item in values_first_ten]
        # y_sq = np.square(y)

        # fig, ax = plt.subplots(figsize=(4.5, 3.5))
        # sns.regplot(
        #     x=x,
        #     y=y_sq,
        #     scatter_kws={'alpha': 0.6, 'color': '#4C72B0', 's': 6, 'linewidths': 0},
        #     line_kws={'color': '#1f4e79', 'linewidth': 1.5},
        #     ax=ax
        # )
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # ax.set_xlabel('Equilibrium Population', fontsize=8)
        # ax.set_ylabel('Degree Centrality²', fontsize=8)
        # ax.tick_params(labelsize=7)
        # ax.text(0.5, 0.98, label, transform=ax.transAxes, fontsize=8, va='top', ha='center')
        # sns.despine(ax=ax)
        # plt.tight_layout()
        # plt.savefig(f"{fig_dir}/{protein_name}_deg_centrality.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
        # plt.savefig(f"{fig_dir}/{protein_name}_deg_centrality.png", format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        # plt.close()

        plot_data.append({"System":protein_name,"Uplet": uplet_type,"Threshold": combined_threshold,"Number": len(filtered_keys)})

        # === Save outputs ===
        _out = f"/ptmp/adlouet/camb/sequence_to_binding_paths/pruning_msm/MSM/community_identification/pickled_files/{protein_name}"
        os.makedirs(_out, exist_ok=True)
        with open(f"{_out}/p_eq_keys.pckl", "wb") as f:
            pickle.dump(P_eq_keys, f)
        with open(f"{_out}/filtered_merged.pckl", "wb") as f:
            pickle.dump(filtered_merged_output, f)
        with open(f"{_out}/dictionary_transitions.pckl", "wb") as f:
            pickle.dump(dictionary_transitions_sorted, f)
        with open(f"{_out}/kd_centrality_ordered.pckl", "wb") as f:
            pickle.dump(kd_centrality_ordered, f)
        with open(f"{_out}/inv_map.pckl", "wb") as f:
            pickle.dump(inv_map, f)


        print(f"Finished {protein_name} successfully!\n")

    except Exception as e:
        print(f"Error processing {protein_name}: {e}")
        continue



