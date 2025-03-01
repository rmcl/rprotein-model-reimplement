import py3Dmol
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


def get_pdb_from_atom_positions(points):
    """
    Converts a NumPy array of atomic positions into a PDB-formatted string.

    Args:
        points (np.ndarray): A (N, 4, 3) array where:
            - N = Number of residues
            - 4 = Atoms per residue (N, CA, C, O)
            - 3 = Cartesian coordinates (x, z, y)

        TODO: MAKE THIS ALSO SUPPORT 1 atom per residue - Ca atom only

    Returns:
        str: PDB-formatted string.
    """
    atom_names = ["N", "CA", "C", "O"]  # Atoms in a glycine residue
    pdb_str = ""
    atom_index = 1  # Atom serial number

    for res_id, residue in enumerate(points, start=1):  # Iterate over residues
        for atom_id, (x, z, y) in enumerate(residue):  # Iterate over atoms, preserving (x, z, y) order
            pdb_str += (
                f"HETATM{atom_index:5d}  {atom_names[atom_id]:<2}  GLY A{res_id:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
            )
            atom_index += 1  # Increment global atom index

    return pdb_str



def view_atom_position_tensor(X, residue_colors = None):
    """Given a (aa len, # atoms, xyz coords) tensor, use py3Dmol to display

    Input:
        X: (aa len, num atoms per aa, xyz)
        residue_colors: Dict[residue pos, color string]

    Example:
        residue_colors = {
            1: "red",
            10: "blue",
            20: "green",
            30: "purple",
            40: "orange",
            50: "pink"
        }


    """
    pdb_str = get_pdb_from_atom_positions(X)

    # Create 3Dmol viewer
    view = py3Dmol.view(width=500, height=500)
    view.addModel(pdb_str, "pdb")

    # Define colors for specific residue numbers


    # Default color for all other residues
    view.setStyle({"cartoon": {"color": "white"}})

    if residue_colors:
        # Apply different colors to specific residues
        for res_id, color in residue_colors.items():
            view.setStyle({"resi": str(res_id)}, {"cartoon": {"color": color}})


    #view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.show()
