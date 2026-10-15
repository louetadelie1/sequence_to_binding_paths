import itertools
import math
import random
from collections import Counter, defaultdict, OrderedDict
from itertools import combinations, islice
from math import nan, isnan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.linalg import expm, eig, norm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize
import networkx as nx
from community import community_louvain
import community as c  # You may only need one of these
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
import mdtraj as md
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from community_identification.kd_calculation import *

np.set_printoptions(legacy='1.25')

def process_weights(w_file):
    KBT = 2.49  # kJ/mol
    colvar_file = w_file
    data = np.loadtxt(colvar_file, comments="#")
    bias = data[:, -1]
    weights = np.exp(bias / KBT)
    weights /= weights.sum() 
    return weights

def trim(w_eq,n_reps,trim_fraction):
    w_split = np.array_split(w_eq, n_reps)
    trimmed_chunks = [chunk[int(trim_fraction * len(chunk)):] for chunk in w_split]
    w_trimmed = np.concatenate(trimmed_chunks)
    return(w_trimmed)

def re_weighting(w_file,n_reps,trim_fraction):
    w_eq=process_weights(w_file)
    if w_eq[0]!=1:
        w_eq /= w_eq.sum()
        w_eq=trim(w_eq,n_reps,trim_fraction)
        return(w_eq)
    else:
        print ("Ensure weight file is included, if no weights pass this step")
        return(w_eq)
        pass


def calculating_threshold(
    pdb,
    distances_com,
    distances_closest,
    w_file=None,
    n_reps=None,
    trim_fraction=None,
    combined_threshold=False,
    w_com=None,
    w_closest=None,
    run_kd=True
):
    """
    Calculates thresholds for distances, with optional reweighting.

    Parameters
    ----------
    pdb : str
        Path to the PDB file (used in kd_calculation if run_kd=True).
    distances_com : array-like
        Center-of-mass distances.
    distances_closest : array-like
        Closest-atom distances.
    w_file : str or None, optional
        Weight file path. If None, function runs unweighted.
    n_reps : int, required if w_file is provided
        Number of replicates (for reweighting).
    trim_fraction : float, required if w_file is provided
        Fraction to trim (for reweighting).
    combined_threshold : bool, default False
        If True, combines distances_com and distances_closest.
    w_com, w_closest : float, required if combined_threshold=True
        Weights for COM and closest distances.
    run_kd : bool, default True
        If True, calls kd_calculation with pdb, number_contact, and weights.

    Returns
    -------
    distances_combined : np.ndarray
        Processed (and possibly trimmed) distance data.
    distance_threshold_combined : float
        The computed threshold value.
    number_contact : np.ndarray
        Count of contacts below threshold for each set of distances.
    weights : np.ndarray or None
        Reweighting factors if w_file was given, otherwise None.
    """

    weights = None
    if w_file is not None:
        # Reweighted mode
        weights = re_weighting(w_file, n_reps, trim_fraction)
        
# edits made here:
    if combined_threshold is True:
        if w_com is None or w_closest is None:
            raise ValueError("w_com and w_closest must be provided when combined_threshold=True")

        distances_combined = (w_com * np.array(distances_com)) + (w_closest * np.array(distances_closest))
        distance_threshold_combined = (0.75 * w_com) + (0.45 * w_closest)
    else:
        distances_combined = (w_com * np.array(distances_com)) + (w_closest * np.array(distances_closest))
        distance_threshold_combined=combined_threshold
    # if combined_threshold:
    #     if w_com is None or w_closest is None:
    #         raise ValueError("w_com and w_closest must be provided when combined_threshold=True")

    #     distances_combined = (w_com * np.array(distances_com)) + (w_closest * np.array(distances_closest))
    #     distance_threshold_combined = (0.75 * w_com) + (0.45 * w_closest)
    # else:
    #     distances_combined = np.array(distances_closest)
    #     distance_threshold_combined = 0.4
        
    print(distance_threshold_combined)
    number_contact = np.array([np.sum(values <= distance_threshold_combined) for values in distances_combined])

    if w_file is not None:
        distances_combined = trim(distances_combined, n_reps, trim_fraction)
        number_contact=trim(number_contact,n_reps,trim_fraction)

    if w_file is None:
        weights = [1.0] * len(distances_com)
    print('calculating_threshold may have worked',number_contact)
    # run kd_calculation only if enabled
    if run_kd:
        Kd_kinetic_weighted, Kd_pop_weighted=kd_calculation(pdb,number_contact,weights)

    return distances_combined, distance_threshold_combined, number_contact, weights, Kd_kinetic_weighted, Kd_pop_weighted


# def transition_matrix(pdb,distances_com,distances_closest,w_file=None,n_reps=None,trim_fraction=None,combined_threshold=False,w_com=None,w_closest=None):
#     distances_combined, distance_threshold_combined, number_contact, weights=calculating_threshold(distances_com,distances_closest,w_file=w_file,n_reps=n_reps,trim_fraction=trim_fraction,combined_threshold=combined_threshold,w_com=w_com,w_closest=w_closest,uplet_type=uplet_type)
    
def transition_matrix_custom(pdb, distances_com, distances_closest,w_file=None, n_reps=None, trim_fraction=None,combined_threshold=True, w_com=None, w_closest=None,uplet_type=None):
    distances_combined, distance_threshold_combined, number_contact, weights,Kd_kinetic_weighted, Kd_pop_weighted = calculating_threshold(pdb,distances_com, distances_closest,w_file=w_file, n_reps=n_reps, trim_fraction=trim_fraction,combined_threshold=combined_threshold, w_com=w_com, w_closest=w_closest)
    distances_closest = np.array(distances_closest, dtype=float)
    distances_com = np.array(distances_com, dtype=float)
    w_eq = np.array(weights, dtype=float)

    u = mda.Universe(pdb)
    protein_residues = u.select_atoms("protein").residues
    num_residues = len(protein_residues)

    residue_pairs = list(combinations(range(num_residues), uplet_type))
    contact_counts_top_uplet_type_indices = {pair: 0 for pair in residue_pairs}

    distances= distances_combined

    num_timesteps = distances.shape[0]

    contact_counts_uplet_type_timesteps = []
    contact_counts_uplet_type_timesteps_including_0 = []
    w_eq_filtered_zeros = []
    print('distance cutoff is',distance_threshold_combined)
    for t, w in zip(range(num_timesteps), w_eq):
        if w != 0.0:
            close_residues = np.where(distances[t, :] < distance_threshold_combined)[0]
            close_residues_values = [x for x in distances[t] if x < distance_threshold_combined]

            paired = list(zip(close_residues_values, close_residues))
            sorted_pairs = sorted(paired, key=lambda x: x[0])
            top_uplet_type_indices = [pair[1] for pair in sorted_pairs[:uplet_type]]

            if len(top_uplet_type_indices) == uplet_type:
                contact_counts_uplet_type_timesteps.append(top_uplet_type_indices)
                contact_counts_uplet_type_timesteps_including_0.append(top_uplet_type_indices)
                w_eq_filtered_zeros.append(w)
            else:
                contact_counts_uplet_type_timesteps_including_0.append([-1] * uplet_type)
        else:
            print('no state exists when weighted')
            contact_counts_uplet_type_timesteps_including_0.append([-1] * uplet_type)

    unique_uplets_pre_process = [tuple(sorted(sublist))
                                 for sublist in contact_counts_uplet_type_timesteps]
    frequency = Counter(unique_uplets_pre_process)
    print(f'For {uplet_type}, there are {len(frequency)} unique pairs')

    filtered_keys = list(frequency.keys())
    data_preprocessed = [tuple(x) for x in unique_uplets_pre_process]
    data = [sublist for sublist in data_preprocessed if sublist in filtered_keys]
    value_to_index = {tuple(row): index for index, row in enumerate(filtered_keys)}

    print(len(filtered_keys), len(value_to_index))

    # add unbound state
    unbound_state = tuple([0] * uplet_type)
    data.insert(0, unbound_state)
    data.append(unbound_state)
    frequency[unbound_state] = 2
    filtered_keys.append(unbound_state)
    value_to_index[unbound_state] = len(value_to_index)

    # check this below :
    unbound_count=contact_counts_uplet_type_timesteps_including_0.count([-1] * uplet_type)
    unbound_fraction=(1-(unbound_count/len(contact_counts_uplet_type_timesteps_including_0)))*100
    print(f'The ligand is bound {unbound_fraction}% of the time')
    # check this ^^^ :

    # build transition matrix
    transition_matrix = np.zeros((len(filtered_keys), len(filtered_keys)), dtype=np.float64)
    if w_eq_filtered_zeros:
        min_w = min(w_eq_filtered_zeros)
        w_eq_filtered_zeros.insert(0, min_w)
        w_eq_filtered_zeros.append(min_w)
        if len(w_eq_filtered_zeros) >= 2:
            w_eq_filtered_zeros[-2] = min_w

    for i in range(len(data) - 1):
        current_value = tuple(sorted(data[i]))
        next_value = tuple(sorted(data[i + 1]))
        weight = w_eq_filtered_zeros[i] if i < len(w_eq_filtered_zeros) else 1.0
        transition_matrix[value_to_index[current_value], value_to_index[next_value]] += weight

    # normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        x_normed = np.nan_to_num(transition_matrix / transition_matrix.sum(axis=1, keepdims=True))

    return x_normed, filtered_keys, Kd_kinetic_weighted, Kd_pop_weighted,transition_matrix


def solving_states_at_equilirum(x_normed,filtered_keys):
    eigenvalues, eigenvectors = eig(x_normed.T)
    # equilibrium_eigenvector_index=[i for i, e in enumerate(eigenvalues) if np.isclose(e, 1.00)]
    equilibrium_eigenvector_index = [np.argmin(np.abs(eigenvalues - 1.0))]
    equilibrium_eigenvector=(eigenvectors[:,equilibrium_eigenvector_index])
    P_eq = np.abs(np.real(equilibrium_eigenvector / np.sum(equilibrium_eigenvector)))
    p_eq_keys={filtered_keys[i]: P_eq[i] for i in range(len(filtered_keys))}
    P_eq_flat = P_eq.flatten()
    equilibrium_matrix = np.tile(P_eq_flat, (x_normed.shape[0], 1)) # matrix where each row is a copy of the equilibrium distribution vector P_eq_flat
    
    return equilibrium_matrix,P_eq,p_eq_keys

def kd_calcualtion_quick_spin(P_eq,filtered_keys,transition_matrix):
    kd=list(map(lambda x: (P_eq.sum()-x)/x, P_eq))
    kd_keys = {filtered_keys[i]: kd[i] for i in range(len(filtered_keys))}
    return kd_keys

def kd_dictionary(x_normed,filtered_keys,transition_matrix):
    ls_letters=filtered_keys
    coord_dict = {}
    rows = ls_letters
    cols = ls_letters

    matrix= transition_matrix #x_normed
    dictionary_transitions={}
    for idx_i,i in enumerate(rows):
        for idx_j,j in enumerate(cols):
            dictionary_transitions[i,j]=matrix[idx_i][idx_j]

    dictionary_transitions_sorted={k: v for k, v in sorted(dictionary_transitions.items(), key=lambda item: item[1],reverse=True)}
    # --> run until here for the hopping v glip_eq_keys_weightedding mechanism
    merged_data = defaultdict(int)

    for (key, value) in dictionary_transitions_sorted.items():
        key_tuple = tuple(key)
        merged_data[key_tuple] += value
    merged_output = [(list(key), value) for key, value in merged_data.items()]

    filtered_merged_output=(list(filter(lambda x: x[1] != 0, merged_output)))

    return dictionary_transitions_sorted,filtered_merged_output

