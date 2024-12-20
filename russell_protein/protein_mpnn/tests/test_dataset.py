from datetime import datetime
from russell_protein.protein_mpnn.dataset import (
    load_cluster_lists,
    load_pdb_chain_detail_csv,
    filter_chains_by_cluster_and_params,
    AnishchankoMultiChainDataBackend
)

DATA_PATH = './russell_protein/protein_mpnn/tests/fixtures/pdb_2021aug02_sample'


def test_multichain_dataset_load_clusters():
    """Test loading the cluster lists."""

    valid_clusters, test_clusters = load_cluster_lists(DATA_PATH)

    assert len(valid_clusters) == 1481
    assert len(test_clusters) == 1554

def test_multichain_dataset_load_pdb_chain_detail_csv():
    """Test loading the PDB chain details CSV."""

    pdb_details = load_pdb_chain_detail_csv(DATA_PATH)

    assert len(pdb_details['chain_details_by_id']) == 555785
    assert len(pdb_details['cluster_id_to_chain_ids']) == 27798

    print(pdb_details['cluster_id_to_chain_ids']['12123'])

    assert pdb_details['cluster_id_to_chain_ids']['12123'] == ['5naf_A', '5naf_B', '5naf_C', '5naf_D', '4lg9_A']

    x5naf_D = pdb_details['chain_details_by_id']['5naf_D']
    assert x5naf_D['sequence'][0:5] == 'MGSSH'
    del x5naf_D['sequence']
    assert x5naf_D['cluster_id'] == '12123'
    assert x5naf_D['deposition_date'] == datetime(2017, 2, 27, 0, 0)
    assert x5naf_D['resolution'] == 2.493

def test_filter_chains_by_cluster_and_params():

    chain_details_by_id = {
        '5naf_D': {
            'sequence': 'MGSSH',
            'cluster_id': '12123',
            'deposition_date': datetime(2017, 2, 27, 0, 0),
            'resolution': 2.493,
            'pdb_id': '5naf'
        },
        '5naf_C': {
            'sequence': 'MGSSH',
            'cluster_id': '12123',
            'deposition_date': datetime(2017, 2, 27, 0, 0),
            'resolution': 2.493,
            'pdb_id': '5naf'
        },
        'X123_A': {
            'sequence': 'MGSSH',
            'cluster_id': '12345',
            'deposition_date': datetime(2017, 2, 27, 0, 0),
            'resolution': 2.493,
            'pdb_id': 'X123'
        }
    }
    cluster_id_to_chain_ids = {
        '12123': ['5naf_D', '5naf_C'],
        '12345': ['X123_A']
    }

    desired_cluster_ids = ['12123']

    results = filter_chains_by_cluster_and_params(
        desired_cluster_ids,
        chain_details_by_id,
        cluster_id_to_chain_ids,
        resolution_cutoff = 2.5,
        pdb_deposition_date = datetime(2017, 2, 28, 0, 0)
    )

    assert set(results) == set([
        '5naf',
    ])

#dataset = AnishchankoMultiChainDataBackend(
    #    'valid',
    #
    #)
