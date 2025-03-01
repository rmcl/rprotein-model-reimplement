import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class ProteinMPNNInput:
    """Input parameters for a ProteinMPNN run"""

    # boolean - True if only the position of the Carbon alpha atoms should be consider
    ca_only : Optional[bool] = False

    # A dictionary of design_name to a dictionary of chain names to lists of fixed residue indexes.
    #    Example: {"D0": {"A": [0, 1, 2,...]}}
    fixed_positions_details : Dict[str, Dict] = None

    # A dictionary of design_name to list of dictionaries of chain names to lists of tied residue indexes.
    # For some reason each list only has a single residue and there is a dictionary for each residue.
    #    Example: {"D0": [{'A': [1], 'C': [1]}..]}}
    tied_positions_details : Dict[str, List[Dict]] = None

    chain_data : Optional[Dict] = None

@dataclass
class ProcessedRun:

    X : np.array = None                    # [B, L_max, 4, 3] # this would be [B, L_max, 1, 3] for Ca only

    residue_indexes : np.array = None # [B, L_max]

    # If a design sequence is shorter than the longest sequence in the batch, the mask is used to pad the sequence.
    # 0 - for padding and 1 for regions with data
    # its unclear to me if mask and mask_pos are different or if we only need one
    mask : np.array = None              # [B, L_max]    - aka chain_M
    mask_pos : np.array = None          # [B, L_max]    - aka chain_M_pos

    chain_encodings : np.array = None    # [B, L_max]    - chain index to residue position

    omit_AA_mask : List[np.array] = None

    def __init__(self, num_batches : int, max_sequence_length : int, ca_only : bool):
        if ca_only:
            self.X = np.zeros([num_batches, max_sequence_length, 1, 3])
        else:
            self.X = np.zeros([num_batches, max_sequence_length, 4, 3])

        self.residue_indexes = np.zeros([num_batches, max_sequence_length], dtype=np.int32)
        self.chain_encodings = np.zeros([num_batches, max_sequence_length], dtype=np.int32)

        self.mask = np.zeros([num_batches, max_sequence_length])
        self.mask_pos = np.zeros([num_batches, max_sequence_length])


@dataclass
class ProcessedBatch:
    batch_index : int = None
    max_sequence_length : int = None # aka L_max - the longest sequence of any chain in the batch

    designs : List['ProcessedDesign'] = None

    def __init__(self, num_designs : int, max_sequence_length):
        """

        """
        self.max_sequence_length = max_sequence_length

        self.designs = []

@dataclass
class ProcessedDesign:
    # the position of the design in the batch
    design_index : int = None
    design_name : str = None

    all_chains : List[str] = field(default_factory=list)
    visible_chains : List[str] = field(default_factory=list)
    masked_chains : List[str] = field(default_factory=list)
    chain_letters : List[str] = field(default_factory=list)

    residue_indexes : np.array = None

    # aka global_idx_start_list
    # this is a list of indexes in the combined sequence for the entire protein where chains start.
    chain_start_indexes : List[int] = field(default_factory=lambda: [0])

    #masked_chain_length_list
    # a list of the lengths of all masked chains
    masked_chain_lengths : List[int] = field(default_factory=list)

    # a list of chain tensors (aa len, num atoms, )
    chain_Xs : List[np.array] = field(default_factory=list)

    chain_masks : List[np.array] = field(default_factory=list)
    chain_sequences : List[str] = field(default_factory=list)

    # chain index for each reside position
    chain_encodings : List[np.array] = field(default_factory=list)

    # a list of length num chains of numpy arrays of length chain length containing 0 or 1 based on if the residue is fixed.
    # 1 - for not fixed
    # 0 - for fixed
    chain_fixed_positions_masks : List[np.array] = field(default_factory=list)

    omitted_AA_masks : List[np.array] = field(default_factory=list)

    # Position-Specific Scoring Matrix (PSSM) - i think!
    pssm_coefs : List[np.array] = field(default_factory=list)
    pssm_biases : List[np.array] = field(default_factory=list)
    pssm_log_odds : List[np.array] = field(default_factory=list)

    bias_by_residues : List[np.array] = field(default_factory=list)

    tied_beta : np.array = None
    tied_positions : List[List[int]] = None
