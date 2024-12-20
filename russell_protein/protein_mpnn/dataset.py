import os
import torch
import random
from typing import Optional
from datetime import datetime
from torch.utils.data import Dataset
from csv import DictReader


class AnishchankoMultiChainDataset(Dataset):
    """Training set for ProteinMPNN curated by Ivan Anishchanko.

    More information at: https://github.com/dauparas/ProteinMPNN/tree/main/training
    """

    # PyTorch Dataset Methods

    def __len__(self):
        """Return the number of proteins in the dataset."""
        return len(self._selected_pdb_ids)

    def __getitem__(self, index):
        """Return the protein at the specified index."""
        pdb_id = self._selected_pdb_ids[index]

        #
        # WARNING WARNING WARNING
        # THIS MAY REALLY BIAS THE DATASET!?
        # for some reason, in the original MPNN implementation
        # they select a random chain here to compute similarity to
        # i don't know why they do this, but it would be good to
        # get some clarification on this
        #
        selected_chain = 'A'

        assemblies = load_assembly_from_pdb_id_containing_chain(
            pdb_id,
            selected_chain,

            self._dataset_path,
            self._percent_homology
        )

        selected_assembly = random.choice(assemblies)

        input_dict = build_protein_mpnn_input_structure_dict(selected_assembly)
        return input_dict

    # Helper Methods

    def __init__(
        self,
        pdb_set_name : str,
        dataset_path : str,

        resolution_cutoff : Optional[float] = None,
        pdb_deposition_data : Optional[datetime] = None,
        percent_homology : float = 0.9
    ):
        """Initialize the dataset.

        resolution_cutoff:
            Filter out structures with resolution in Angstroms below this value.

        pdb_deposition_data:
            Filter out proteins deposited after this date.
        """

        self._dataset_path = dataset_path
        self._percent_homology = percent_homology

        if pdb_set_name not in ['valid', 'test']:
            raise Exception('Unknown pdb_set_name. Options are "valid" or "test"')

        valid_pdb_ids, test_pdb_ids = load_cluster_lists(self._dataset_path)
        pdb_details = load_pdb_chain_detail_csv(self._dataset_path)

        if pdb_set_name == 'valid':
            self._selected_pdb_ids = filter_pdbs_by_cluster_and_params(
                valid_pdb_ids,

                pdb_details['chain_details_by_id'],
                pdb_details['cluster_id_to_chain_ids'],
                resolution_cutoff,
                pdb_deposition_data
            )

        elif pdb_set_name == 'test':
            self._selected_pdb_ids = filter_pdbs_by_cluster_and_params(
                test_pdb_ids,

                pdb_details['chain_details_by_id'],
                pdb_details['cluster_id_to_chain_ids'],
                resolution_cutoff,
                pdb_deposition_data
            )


def get_assembly_xyz_coords(pdb_id, chains, assembly_index, xform, chain_data):
    """Return the xyz coordinates of the chains in the assembly."""

    # asmb_xformIDX  - (one per biounit) xforms to be applied to chains from asmb_chains[IDX], [n,4,4]
    # [n,:3,:3] - rotation matrices
    # [n,3,:3] - translation vectors
    rotation_matrix_u = xform[:,:3,:3]
    translation_matrix_r = xform[:,:3,3]

    assembly_xyz_data = {}

    for chain_name in chains:
        chain_details = chain_data[chain_name]

        xyz = chain_details['xyz']
        xyz_ru = torch.einsum('bij,raj->brai', rotation_matrix_u, xyz) + translation_matrix_r[:,None,None,:]

        assembly_xyz_data.update({
            (chain_name, assembly_index, i): xyz_i
            for i, xyz_i in enumerate(xyz_ru)
        })

    return assembly_xyz_data


def load_assembly_from_pdb_id_containing_chain(
    pdb_id,
    chain_id,
    dataset_path,
    percent_homology : float,
):
    """Build all protein assemblies from a PDB ID containing the specified specific chain."""

    prefix = pdb_id[1:3]
    structure_metadata_path = f"{dataset_path}/pdb/{prefix}/{pdb_id}.pt"

    if not os.path.isfile(structure_metadata_path):
        raise FileNotFoundError(f"Could not find structure metadata for {pdb_id}")

    meta = torch.load(structure_metadata_path)
    assembly_ids = meta['asmb_ids']
    assembly_chains = meta['asmb_chains']
    chain_ids = meta['chains']

    # Return the percent homology to other chains
    # relative to chain_id chain
    homologous_chains = []

    chain_index = chain_ids.index(chain_id)
    chain_similarities = meta['tm'][chain_index,:,1]
    for i, chain_homology in enumerate(chain_similarities):
        if chain_homology > percent_homology:
            homologous_chains.append(chain_ids[i])

    assemblies = []

    chain_data = {}
    for chain_name in chain_ids:
        chain_details = load_chain_data(pdb_id, chain_name, dataset_path)
        chain_data[chain_name] = chain_details

    for assembly_index, asembly_id in enumerate(assembly_ids):

        # Add the chains that are part of the assembly if they are also in
        # the metadata chains list
        chains = []
        assembly_chains = assembly_chains[assembly_index].split(',')
        for c in assembly_chains:
            if c in meta['chains']:
                chains.append(c)

        # OMIT ALL ASSEMBLIES THAT DO NOT CONTAIN THE DESIRED CHAIN
        if chain_id not in chains:
            continue

        xform = meta[f'asmb_xform{assembly_index}']

        assembly_xyz_data = get_assembly_xyz_coords(
            pdb_id,
            chains,
            assembly_index,
            xform,
            chain_data)

        chain_details = []
        seq, masked = "", []
        for counter, (assembly_key, assembly_xyz_coords) in enumerate(assembly_xyz_data.items()):
            (chain_name, assembly_index, i) = assembly_key

            chain_sequence = chain_data[chain_name]['seq']

            seq += chain_sequence

            if chain_name in homologous_chains:
                masked.append(counter)

            chain_detail = {
                'name': chain_name,
                'seq' : chain_sequence,
                'xyz' : assembly_xyz_coords,
                'idx' : torch.full((assembly_xyz_coords.shape[0],),counter),
            }
            chain_details.append(chain_detail)

        assemblies.append({
            'label'  : pdb_id,
            'chains' : chain_details,
            'masked' : torch.Tensor(masked).int(),
        })

    return assemblies


def filter_pdbs_by_cluster_and_params(
    desired_cluster_ids,

    chain_details_by_id,
    cluster_id_to_chain_ids,
    resolution_cutoff: Optional[float],
    pdb_deposition_date : Optional[datetime]
):
    """Filter clusters by resolution and creation date of their chains."""

    filtered_pdb_ids = []
    for desired_cluster_id in desired_cluster_ids:

        include_cluster = True
        pdb_id = None

        chain_ids = cluster_id_to_chain_ids.get(desired_cluster_id, [])

        if len(chain_ids) == 0:
            include_cluster = False

        for chain_id in chain_ids:
            chain_details = chain_details_by_id[chain_id]

            if resolution_cutoff:
                if chain_details['resolution'] > resolution_cutoff:
                    include = False

            if pdb_deposition_date:
                if chain_details['deposition_date'] > pdb_deposition_date:
                    include = False

            pdb_id = chain_details['pdb_id']

        if include_cluster:
            filtered_pdb_ids.append(pdb_id)

    return filtered_pdb_ids


def load_cluster_lists(dataset_path):
    """Return the "valid" and "test" clusters as defined by the dataset."""
    VALID_CLUSTER_PATH = f'{dataset_path}/valid_clusters.txt'
    TEST_CLUSTER_PATH = f'{dataset_path}/test_clusters.txt'

    with open(VALID_CLUSTER_PATH, 'r') as f:
        valid_clusters = f.read().splitlines()

    with open(TEST_CLUSTER_PATH, 'r') as f:
        test_clusters = f.read().splitlines()

    return valid_clusters, test_clusters


def load_pdb_chain_detail_csv(dataset_path):
    """The dataset contains a list.csv file with the following data:

    CHAINID    - chain label, PDBID_CHAINID
    DEPOSITION - deposition date
    RESOLUTION - structure resolution
    HASH       - unique 6-digit hash for the sequence
    CLUSTER    - sequence cluster the chain belongs to (clusters were generated at seqID=30%)
    SEQUENCE   - reference amino acid sequence
    """

    csv_path = f'{dataset_path}/list.csv'

    # the "Cluster" column seems to group chains together into complexes
    # it seems that all chains in a complex are included in the dataset
    # so we will capture this information in a dictionary by cluster_id
    # for later use in filtering
    cluster_id_to_chain_ids = {}
    chain_detail_by_id = {}

    with open(csv_path, 'r') as read_obj:
        csv_reader = DictReader(read_obj)
        for row in csv_reader:
            chain_id = row['CHAINID']
            cluster_id = row['CLUSTER']

            pdb_id = chain_id.split('_')[0]

            chain_detail_by_id[chain_id] = {
                'deposition_date': datetime.strptime(row['DEPOSITION'], '%Y-%m-%d'),
                'resolution': float(row['RESOLUTION']),
                'hash': row['HASH'],
                'sequence': row['SEQUENCE'],
                'cluster_id': cluster_id,
                'pdb_id': pdb_id
            }
            cluster_id_to_chain_ids.setdefault(cluster_id, []).append(chain_id)

    return {
        'chain_details_by_id': chain_detail_by_id,
        'cluster_id_to_chain_ids': cluster_id_to_chain_ids
    }


#cached_chain_data = {}
def load_chain_data(pdb_id, chain_id, data_path):
    """Load the chain data from the dataset."""

    #global cached_chain_data

    #key = f'{pdb_id}_{chain_id}'
    #if key in cached_chain_data:
    #    return cached_chain_data[key]

    path = f'{data_path}/pdb/l3/{pdb_id}_{chain_id}.pt'
    data = torch.load(path)

    #cached_chain_data[key] = data
    return data

def remove_6xHis_tags(chain_seq, xyz_coords):
    """Search and remove 6xHis tags from the beginning and end of chains

    Search for 6xHis tags in the first 5 residues of the chain as well as
    the last 5 residues of the chain. If found, remove the tag and corresponding
    coordinates from the chain.

    """
    tag = 'HHHHHH'
    pos = chain_seq.find(tag)
    if pos > -1 and pos < 5:
        chain_seq = chain_seq[pos+len(tag):]
        xyz_coords = xyz_coords[pos+len(tag):,:,:]

    if pos > -1 and pos > len(chain_seq) - 5:
        chain_seq = chain_seq[:pos]
        xyz_coords = xyz_coords[:pos,:,:]

    return chain_seq, xyz_coords

final_alphabet = 'ABCDEFGHIKLMNPQRSTVWXYZ'

def build_protein_mpnn_input_structure_dict(input_data):
    """Build the input structure for the Protein MPNN model"""
    result = {
        'name': input_data['label'],
        'num_of_chains': len(input_data['chains']),
    }

    concat_seq = ''
    masked_chain_list = []
    visible_chain_list = []
    for chain_index, chain_details in enumerate(input_data['chains']):
        chain_id = final_alphabet[chain_index]
        chain_seq = chain_details['seq']

        # Filter 6xHis tags
        chain_seq, chain_xyz = remove_6xHis_tags(chain_seq, chain_details['xyz'])

        concat_seq += chain_seq

        print(input_data['masked'])
        if chain_index in input_data['masked']:
            masked_chain_list.append(chain_id)
        else:
            visible_chain_list.append(chain_id)

        result[f'seq_chain_{chain_id}'] = chain_seq
        result[f'coords_chain_{chain_id}'] = {
            f'N_chain_{chain_id}': chain_xyz[:,0,:],
            f'CA_chain_{chain_id}': chain_xyz[:,1,:],
            f'C_chain_{chain_id}': chain_xyz[:,2,:],
            f'O_chain_{chain_id}': chain_xyz[:,3,:],
        }

    result['masked_list'] = masked_chain_list
    result['visible_list']= visible_chain_list
    result['seq'] = concat_seq
    return result
