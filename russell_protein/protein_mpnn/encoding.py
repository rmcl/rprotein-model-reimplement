import torch
from typing import Tuple
from .input_process import ATOM_ORDER_MAP_FOUR_ATOMS

CA_INDEX = ATOM_ORDER_MAP_FOUR_ATOMS['CA']
N_INDEX = ATOM_ORDER_MAP_FOUR_ATOMS['N']
C_INDEX = ATOM_ORDER_MAP_FOUR_ATOMS['C']
O_INDEX = ATOM_ORDER_MAP_FOUR_ATOMS['O']

def get_atom_positions(X):
    """Return the positions of the atoms in the input tensor X including virtual atom "Cb"."""

    # b is distance between Ca and N
    b = X[:,:,CA_INDEX,:] - X[:,:,N_INDEX,:]

    # c is distance between C and Ca
    c = X[:,:, C_INDEX,:] - X[:,:, CA_INDEX,:]
    a = torch.cross(b, c, dim=-1)

    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,CA_INDEX,:]
    Ca = X[:,:,CA_INDEX,:]
    N = X[:,:,N_INDEX,:]
    C = X[:,:,C_INDEX,:]
    O = X[:,:,O_INDEX,:]

    return Ca, Cb, N, C, O

def compute_pairwise_distances(
    atom_positions: torch.Tensor,
    mask: torch.Tensor,
    top_k : int,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise Euclidean distances between residues based on their Cα atom positions.

    Args:
        atom_positions (torch.Tensor): Tensor of shape (B, L, 3) representing the Cα atomic coordinates.
        mask (torch.Tensor): Tensor of shape (B, L) indicating valid residues (1 for valid, 0 for padding).
        top_k (int): Number of nearest neighbors to consider.
        eps (float, optional): Small epsilon value to prevent division by zero in sqrt operation. Default is 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Distance matrix of shape (B, L, K) containing the top-K nearest neighbor distances.
            - Index matrix of shape (B, L, K) containing the indices of the nearest neighbors.
    """
    # Create a 2D mask to exclude invalid residues
    pairwise_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (B, L, L)

    # Compute pairwise distance matrix
    displacement = atom_positions.unsqueeze(1) - atom_positions.unsqueeze(2)  # Shape: (B, L, L, 3)
    distance_matrix = pairwise_mask * torch.sqrt(torch.sum(displacement ** 2, dim=-1) + eps)  # Shape: (B, L, L)

    # Adjust distances for padded residues to ensure they are ignored
    max_distance, _ = torch.max(distance_matrix, dim=-1, keepdim=True)
    adjusted_distances = distance_matrix + (1.0 - pairwise_mask) * max_distance

    # Select top-K nearest neighbors
    k_neighbors = min(top_k, atom_positions.shape[1])
    top_k_distances, neighbor_indices = torch.topk(adjusted_distances, k_neighbors, dim=-1, largest=False)

    return top_k_distances, neighbor_indices

def gather_edges(edges, neighbor_idx):
    """This function extracts edge features from a full adjacency matrix (edges)
    based on neighbor indices (neighbor_idx).

    edges [B,N,N,C] at
    Neighbor indices [B,N,K] =>

    Returns
        Neighbor features [B,N,K,C]
    """

    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def get_rbf_for_distances(D, num_rbf):
    """This function takes D, a tensor of distances, and transforms them into a higher-dimensional RBF space."""

    # WHY 2-22?
    # ChatGPT thinks "2 Å to 22 Å (angstroms) is a reasonable range for interatomic or residue distances in proteins."
    D_min, D_max, D_count = 2., 22., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device) # Generate RBF centers

    D_mu = D_mu.view([1,1,1,-1])  # Reshape for broadcasting
    D_sigma = (D_max - D_min) / D_count  # Compute standard deviation
    D_expand = torch.unsqueeze(D, -1)  # Expand D for broadcasting
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)  # Compute RBF transformation

    return RBF

def get_rbf(A, B, E_idx):
    """"""

    # probably should make sure this is a good idea
    num_rbf = E_idx.shape[-1]

    D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
    D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
    RBF_A_B = get_rbf_for_distances(D_A_B_neighbors, num_rbf)
    return RBF_A_B


def get_rbf_for_top_k_neighbors(
    top_k_distances,
    top_k_neighbor_indexes,
    Ca, Cb, N, C, O,
    num_rbfs
):
    rbfs = []
    rbfs.append(get_rbf_for_distances(top_k_distances, num_rbfs))       #Ca-Ca
    rbfs.append(get_rbf(N, N, top_k_neighbor_indexes))                  #N-N
    rbfs.append(get_rbf(C, C, top_k_neighbor_indexes))                  #C-C
    rbfs.append(get_rbf(O, O, top_k_neighbor_indexes))                  #O-O
    rbfs.append(get_rbf(Cb, Cb, top_k_neighbor_indexes))                #Cb-Cb
    rbfs.append(get_rbf(Ca, N, top_k_neighbor_indexes))                 #Ca-N
    rbfs.append(get_rbf(Ca, C, top_k_neighbor_indexes))                 #Ca-C
    rbfs.append(get_rbf(Ca, O, top_k_neighbor_indexes))                 #Ca-O
    rbfs.append(get_rbf(Ca, Cb, top_k_neighbor_indexes))    #Ca-Cb
    rbfs.append(get_rbf(N, C, top_k_neighbor_indexes))      #N-C
    rbfs.append(get_rbf(N, O, top_k_neighbor_indexes))      #N-O
    rbfs.append(get_rbf(N, Cb, top_k_neighbor_indexes))     #N-Cb
    rbfs.append(get_rbf(Cb, C, top_k_neighbor_indexes))     #Cb-C
    rbfs.append(get_rbf(Cb, O, top_k_neighbor_indexes))     #Cb-O
    rbfs.append(get_rbf(O, C, top_k_neighbor_indexes))      #O-C
    rbfs.append(get_rbf(N, Ca, top_k_neighbor_indexes))     #N-Ca
    rbfs.append(get_rbf(C, Ca, top_k_neighbor_indexes))     #C-Ca
    rbfs.append(get_rbf(O, Ca, top_k_neighbor_indexes))     #O-Ca
    rbfs.append(get_rbf(Cb, Ca, top_k_neighbor_indexes))    #Cb-Ca
    rbfs.append(get_rbf(C, N, top_k_neighbor_indexes))      #C-N
    rbfs.append(get_rbf(O, N, top_k_neighbor_indexes))      #O-N
    rbfs.append(get_rbf(Cb, N, top_k_neighbor_indexes))     #Cb-N
    rbfs.append(get_rbf(C, Cb, top_k_neighbor_indexes))     #C-Cb
    rbfs.append(get_rbf(O, Cb, top_k_neighbor_indexes))     #O-Cb
    rbfs.append(get_rbf(C, O, top_k_neighbor_indexes))      #C-O
    rbfs = torch.cat(tuple(rbfs), dim=-1)

    return rbfs



def build_features_for_network_input(X, mask, residue_indexes, chain_encodings, num_rbfs, top_k):
    Ca = X[:,:,CA_INDEX,:]
    top_k_distances, top_k_neighbor_indexes = compute_pairwise_distances(Ca, mask, top_k)

    rbfs = get_rbf_for_top_k_neighbors(
        top_k_distances,
        top_k_neighbor_indexes,
        *get_atom_positions(X),
        num_rbfs
    )

    # so (B, L_max) and the None's add another dimension
    # then we broadcast subtract so we end up with (B, L_max, L_max)
    offset = residue_indexes[:,:,None] - residue_indexes[:,None,:]
    # we add another dimension so offset becomes (B, L_max, L_max, 1)
    offset = gather_edges(offset[:,:,:,None], top_k_neighbor_indexes)[:,:,:,0]

    d_chains = ((chain_encodings[:,:,None] - chain_encodings[:,None,:]) == 0).long()
    E_chains = gather_edges(d_chains[:,:,:,None], top_k_neighbor_indexes)[:,:,:,0]

    return top_k_neighbor_indexes, rbfs, offset.long(), E_chains
