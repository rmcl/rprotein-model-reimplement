import torch
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

def data_to_pdb(data, output_file="output.pdb"):
    """
    Convert ProteinMPNN input data to a PDB file.

    You can then use nglview to visualize the generated PDB file in a Jupyter notebook.
    ```
    import nglview as nv
    view = nv.show_structure_file("output.pdb")
    view
    ```


    Args:
        data (dict): A dictionary containing protein data with sequence and coordinates.
                     Expected keys: 'name', 'seq_chain_A', 'coords_chain_A', 'seq_chain_B', 'coords_chain_B'.
        output_file (str): Path to save the generated PDB file.
    """
    # Create structure hierarchy: Structure > Model > Chains > Residues > Atoms
    structure = Structure.Structure(data['name'])
    model = Model.Model(0)  # Single model

    # Function to add a chain
    def add_chain(seq, coords_dict, chain_id):
        chain = Chain.Chain(chain_id)  # Create a chain with the specified ID

        # Iterate over residues in the sequence and their coordinates
        for i, res in enumerate(seq):
            res_id = (" ", i + 1, " ")  # Residue ID must be unique
            residue = Residue.Residue(res_id, res, "")

            # Add backbone atoms (N, CA, C, O) if coordinates exist
            for atom_name, atom_coords in coords_dict.items():

                base_atom_name = atom_name.split("_")[0]  # Extract 'N', 'CA', 'C', 'O' from 'N_chain_A'
                xyz = atom_coords[i]
                if not torch.any(torch.isnan(xyz)):
                    atom = Atom.Atom(base_atom_name, atom_coords[i], 1.0, 1.0, " ", base_atom_name, i + 1, element=base_atom_name[0])
                    residue.add(atom)

            chain.add(residue)

        return chain

    possible_chains = 'ABCDEFGHIJKLMNOP'
    for chain_name in possible_chains:
        if f'seq_chain_{chain_name}' in data and f'coords_chain_{chain_name}' in data:
            model.add(
                add_chain(
                    data[f'seq_chain_{chain_name}'],
                    data[f'coords_chain_{chain_name}'],
                    chain_name)
            )

    structure.add(model)

    # Write to a PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)
