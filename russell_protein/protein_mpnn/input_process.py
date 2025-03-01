import re
import numpy as np
import torch
from .input_types import (
    ProteinMPNNInput,
    ProcessedRun,
    ProcessedBatch,
    ProcessedDesign
)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_DICT = dict(zip(ALPHABET, range(21)))

ATOM_ORDER_FOUR_ATOMS =['N', 'CA', 'C', 'O']
ATOM_ORDER_CA_ONLY = ['Ca']

ATOM_ORDER_MAP_FOUR_ATOMS = dict([
    (atom_name, index)
    for index, atom_name in enumerate(ATOM_ORDER_FOUR_ATOMS)
])
ATOM_ORDER_MAP_CA_ONLY = dict([
    (atom_name, index)
    for index, atom_name in enumerate(ATOM_ORDER_CA_ONLY)
])


def process_chain(
    protein_mpnn_input : ProteinMPNNInput,
    batch_details : ProcessedBatch,
    design_details : ProcessedDesign,
    input_design_data : dict, # maybe we make the batch have a dataclass someday.
    chain_index : int,
    chain_letter : str,
):
    """Process a single chain of a design"""

    design_details.chain_letters.append(chain_letter)

    # Replace "-" in sequence with "X"
    chain_sequence = input_design_data[f'seq_chain_{chain_letter}']
    chain_sequence = chain_sequence.replace('-', 'X')
    design_details.chain_sequences.append(chain_sequence)

    chain_length = len(chain_sequence)

    chain_start_index = design_details.chain_start_indexes[-1]
    chain_end_index = chain_start_index + chain_length

    design_details.chain_start_indexes.append(chain_end_index)
    design_details.masked_chain_lengths.append(chain_length)

    # Handle masked chains
    if chain_letter in design_details.masked_chains:
        #design_details.masked_chains.append(chain_letter)
        design_details.chain_masks.append(np.ones(chain_length))

    # Handle visible chains
    if chain_letter in design_details.visible_chains:
        # TODO make sure we dont need a seperate list for this.. i don think so
        #design_details.visible_chains.append(chain_letter)
        design_details.chain_masks.append(np.zeros(chain_length))

    chain_coordinates = input_design_data[f'coords_chain_{chain_letter}']
    if protein_mpnn_input.ca_only:
        atom_keys = [
            f'{a}_chain_{chain_letter}'
            for a in ATOM_ORDER_CA_ONLY
        ]
    else:
        atom_keys = [
            f'{a}_chain_{chain_letter}'
            for a in ATOM_ORDER_FOUR_ATOMS
        ]

    x_chain = np.stack([
        chain_coordinates[c]
        for c in atom_keys
    ], 1) #[chain length,4,3]

    design_details.chain_Xs.append(x_chain)
    design_details.chain_encodings.append(
        chain_index * np.ones(chain_length))

    fixed_position_mask = np.ones(chain_length)
    if protein_mpnn_input.fixed_positions_details:
        design_fixed_positions = protein_mpnn_input.fixed_positions_details.get(design_details.design_name, {})
        chain_fixed_position_indexes = np.array(design_fixed_positions.get(chain_letter, [])) - 1 # TODO/QUESTION: THIS -1 is in the ProteinMPNN code, does that mean these are specified in jsonl file as 1-index

        # update the mask and make the fixed positions 0
        fixed_position_mask[chain_fixed_position_indexes] = 0.0

    design_details.chain_fixed_positions_masks.append(fixed_position_mask)

    # TODO TODO TODO - omit_AA not fully implemented. Currently ignore the input file
    # and always returning a zero mask
    design_details.omitted_AA_masks.append(
        np.zeros([chain_length, len(ALPHABET)], np.int32)
    )

    # Position-Specific Scoring Matrix (PSSM) - i think!
    # TODO TODO TODO: Not going to implement this right now
    design_details.pssm_coefs.append(np.zeros(chain_length))
    design_details.pssm_biases.append(np.zeros([chain_length, 21]))
    design_details.pssm_log_odds.append(10000.0 * np.ones([chain_length, 21]))

    # TODO: Not implementing Bias by residue either
    design_details.bias_by_residues.append(np.zeros([chain_length, 21]))


chain_pattern = re.compile(r'seq_chain_(\w)')

def get_masked_and_visible_chains_for_design(
    protein_mpnn_input : ProteinMPNNInput,
    design: ProcessedDesign,
    design_data
):
    chain_letters = []
    for key in design_data.keys():
        result = chain_pattern.match(key)
        if result:
            chain_letters.append(result.group(1))

    if protein_mpnn_input.chain_data is not None:
        raise Exception('not implemented.')
        # todo figure this out
        #design_name = design_data['name']
        #masked_chains, visible_chains = chain_data[design_name] #masked_chains a list of chain letters to predict [A, D, F]
    else:
        masked_chains = list(chain_letters)
        visible_chains = []

    masked_chains.sort() #sort masked_chains
    visible_chains.sort() #sort visible_chains
    all_chains = masked_chains + visible_chains

    design.masked_chains = masked_chains
    design.visible_chains = visible_chains
    design.all_chains = all_chains


def process_design(
    protein_mpnn_input : ProteinMPNNInput,
    batch_details : ProcessedBatch,
    design_index : int,
    design_data,
):

    design_name = design_data.get('name', None)
    if not design_name:
        raise Exception('design name not specified.')

    design_details = ProcessedDesign(
        design_index=design_index,
        design_name=design_name
    )

    get_masked_and_visible_chains_for_design(
        protein_mpnn_input,
        design_details,
        design_data)

    for chain_index, chain_letter in enumerate(design_details.all_chains):
        print(chain_index, chain_letter)
        process_chain(
            protein_mpnn_input,
            batch_details,
            design_details,
            design_data,
            chain_index,
            chain_letter)


    total_design_len = sum(design_details.masked_chain_lengths)
    design_details.residue_indexes = np.zeros(total_design_len, dtype=np.int32)

    for chain_index, chain_letter in enumerate(design_details.all_chains):
        chain_start_index = design_details.chain_start_indexes[chain_index]
        chain_length = design_details.masked_chain_lengths[chain_index]
        chain_end_index = chain_start_index + chain_length

        design_details.residue_indexes[chain_start_index:chain_end_index] = \
            100 * chain_index + np.arange(chain_start_index, chain_end_index)


    batch_details.designs.append(design_details)
    return design_details

def compute_max_sequence_length(batch):
    """Compute the maximum sequence length of a design with all chains combined"""
    max_length = 0
    for _, design_data in enumerate(batch):
        design_seq_len = 0
        for key, value in design_data.items():
            if key.find('seq_chain_') == 0:
                design_seq_len += len(value)

        max_length = max(max_length, design_seq_len)

    return max_length


def process_batch(protein_mpnn_input, batch_index, batch) -> ProcessedBatch:

    num_designs = len(batch)
    max_sequence_length = compute_max_sequence_length(batch)
    batch_details = ProcessedBatch(num_designs, max_sequence_length)

    # I MAY HAVE MISUNDERSTOOD. IS THERE ALWAYS ONLY ONE DESIGN IN A BATCH?
    assert num_designs == 1, 'Only one design per batch is supported? Revisit this if this assertion is false'

    for design_index, design_data in enumerate(batch):
        print(f'Processing Design: {design_index}')
        processed_design = process_design(
            protein_mpnn_input,
            batch_details,
            design_index,
            design_data)

        process_tied_positions_for_design(
            protein_mpnn_input,
            batch_details,
            processed_design
        )

    return batch_details

def process_run(protein_mpnn_input, batches):
    """Process a list of batches"""

    max_sequence_length = 0
    process_batches = []
    for batch_index, batch in enumerate(batches):
        batch_result = process_batch(protein_mpnn_input, batch_index, batch)
        process_batches.append(batch_result)

        max_sequence_length = max(
            max_sequence_length,
            batch_result.max_sequence_length)

    run = ProcessedRun(
        len(batches),
        max_sequence_length,
        protein_mpnn_input.ca_only
    )

    # Populate values for run based on batches
    for batch_index, batch in enumerate(process_batches):
        for design_index, design_data in enumerate(batch.designs):

            assert design_index == 0, 'Only one design per batch is supported? Revisit this if this assertion is false'

            design_x = np.concatenate(design_data.chain_Xs, 0 ) #[L, 4, 3]
            design_all_sequence = "".join(design_data.chain_sequences)
            design_all_sequence_length = len(design_all_sequence)

            # This is the difference in the maximum length sequence in the batch versus the length of the current design sequence
            sequence_pad_len = max_sequence_length - design_all_sequence_length

            # not confident about this padding.
            all_chain_encodings = np.concatenate(design_data.chain_encodings, 0) #[L,]
            run.chain_encodings[batch_index, :] = np.pad(all_chain_encodings, [0, sequence_pad_len], 'constant', constant_values=(0, ))

            run.X[batch_index,:,:,:] = np.pad(design_x, [[0, sequence_pad_len], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))

            run.residue_indexes[batch_index,:] = np.pad(design_data.residue_indexes, [[0, sequence_pad_len]], 'constant', constant_values=(0, ))

            design_chain_masks = np.concatenate(design_data.chain_masks, 0) #[L,], 1.0 for places that need to be predicted
            run.mask[design_index,:] = np.pad(design_chain_masks, [[0, sequence_pad_len]], 'constant', constant_values=(0.0, ))
            run.mask_pos[design_index,:] = np.pad(design_chain_masks, [[0,sequence_pad_len]], 'constant', constant_values=(0.0, ))

            # not certain about this
            run.omit_AA_mask = np.pad(np.concatenate(design_data.omitted_AA_masks, 0), [[0,sequence_pad_len]], 'constant', constant_values=(0.0, ))

    run.chain_encodings = torch.from_numpy(run.chain_encodings).long()
    run.X = torch.from_numpy(run.X).float()
    run.residue_indexes = torch.from_numpy(run.residue_indexes).long()
    run.mask = torch.from_numpy(run.mask).float()
    run.mask_pos = torch.from_numpy(run.mask_pos).float()
    run.omit_AA_mask = torch.from_numpy(run.omit_AA_mask).float()
    run.tied_beta = torch.from_numpy(np.array([design.tied_beta for batch in process_batches for design in batch.designs])).float()
    run.tied_positions = torch.from_numpy(np.array([design.tied_positions for batch in process_batches for design in batch.designs])).long()

    return run

def process_tied_positions_for_design(
    protein_mpnn_input: ProteinMPNNInput,
    batch_details : ProcessedBatch,
    design_details: ProcessedDesign
):
    """Process tied positions for a given design.

    Updates the processed design details with tied positions and beta values.
    """
    tied_positions = protein_mpnn_input.tied_positions_details[design_details.design_name]

    tied_beta = np.ones(batch_details.max_sequence_length)
    all_tied_residues_absolute_indexes = []

    for tied_item in tied_positions:
        tied_residue_absolute_indexes = []
        for chain_letter, residue_indexes in tied_item.items():
            chain_index = get_design_chain_index(design_details, chain_letter)
            sequence_chain_start_index = design_details.chain_start_indexes[chain_index]

            if not isinstance(residue_indexes, list):
                raise TypeError("residue_indexes in tied input should be a list")

            for residue_index in residue_indexes:
                if isinstance(residue_index, int):
                    residue_index = residue_index
                    # Assuming positive design so weight is set to 1
                    weight = 1.0

                elif isinstance(residue_index, list):
                    if len(residue_indexes) != 2:
                        raise Exception(
                            'List of ints should contain two values: [residue_index, weight] where weight '
                            'is +1 for positive design and -1 for negative design'
                        )

                    residue_index = residue_index[0]
                    weight = residue_index[1]

                else:
                    raise Exception(
                        'Each tied residue index should be an int or a list of ints. List of ints should contain'
                        ' two values: [residue_index, weight] where weight is +1 for positive design'
                        ' and -1 for negative design'
                    )

                tied_residue_index = sequence_chain_start_index + residue_index - 1
                tied_beta[tied_residue_index] = weight
                tied_residue_absolute_indexes.append(tied_residue_index)

        all_tied_residues_absolute_indexes.append(tied_residue_absolute_indexes)

    design_details.tied_beta = tied_beta
    design_details.tied_positions = all_tied_residues_absolute_indexes

# utilities

def get_design_chain_index(
    design_details : ProcessedDesign,
    chain_letter : str
):
    """Get the index of a chain in a design given the chain letter"""
    return design_details.chain_letters.index(chain_letter)
