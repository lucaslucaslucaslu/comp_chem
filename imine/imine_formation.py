import pandas as pd
from rdkit.Chem import AllChem as Chem
import os
import autode as ade

ade.Config.n_cores = 16
ade.Config.max_core = 4000

# for small value testing
ds_1 = pd.DataFrame({
    'SMILES': ['CC(C)=O', 'C=O', 'CC(C)C=O'],
    'Name': ['Acetone', 'Formaldehyde', 'Isobutyraldehyde']
})

ds_2 = pd.DataFrame({
    'SMILES': ['CC(N)C', 'CN', 'CCC(N)C'],
    'Name': ['Isopropylamine', 'Methylamine', 'Isobutylamine']
})


# # real set of aldehydes and methyl ketones
# ds_1 = pd.DataFrame({
#     'SMILES': [
#         'CC(=O)C', 'CCC(=O)C', 'CCCC(=O)C', 'CC(C)=O', 'CC(=O)CC',
#         'CCC(=O)CC', 'CC(=O)C(C)C', 'CCC(=O)C(C)C', 'CC(C)C(=O)C(C)C',
#         'CCC(C)C(=O)C(C)C', 'CCCC(C)C(=O)C(C)C', 'CC(C)(C)C(=O)C(C)C',
#         'CCC(C)(C)C(=O)C(C)C', 'CCCC(C)(C)C(=O)C(C)C', 'CC(C)C(=O)C',
#         'CCC(C)C(=O)C', 'CCCC(C)C(=O)C', 'CC(C)(C)C(=O)C(C)C',
#         'CCC(C)(C)C(=O)C(C)C', 'CCCC(C)(C)C(=O)C(C)C', 'CC(=O)CC(C)C',
#         'CCC(=O)CC(C)C', 'CCCC(=O)CC(C)C', 'CC(=O)CCC(C)C', 'CCC(=O)CCC(C)C',
#         'CCCC(=O)CCC(C)C', 'CC(=O)CCCC(C)C', 'CCC(=O)CCCC(C)C', 'CCCC(=O)CCCC(C)C',
#         'CC(C)CC(=O)C(C)C', 'CCC(C)CC(=O)C(C)C', 'CCCC(C)CC(=O)C(C)C',
#         'CC(C)(C)CC(=O)C(C)C', 'CCC(C)(C)CC(=O)C(C)C', 'CCCC(C)(C)CC(=O)C(C)C',
#         'CC(C)CC(=O)CC(C)C', 'CCC(C)CC(=O)CC(C)C', 'CCCC(C)CC(=O)CC(C)C',
#         'CC(C)(C)CC(=O)CC(C)C', 'CCC(C)(C)CC(=O)CC(C)C', 'CCCC(C)(C)CC(=O)CC(C)C'
#     ]
# })

# #set of primary amines
# ds_2 = pd.DataFrame({
#     'SMILES': [
#         'CC(N)C', 'CCC(N)C', 'CC(C)CN', 'CC(C)(C)CN', 'CCCC(N)C',
#         'CC(CCC)N', 'CC(CC)CN', 'CC(N)CC', 'CC(NC)C', 'CC(CN)C',
#         'C(CN)C', 'CC(C)NCC', 'CC(CCC)NCC', 'CC(CCC)N(C)C', 'CC(CCC)NC',
#         'CCC(CC)N', 'CC(CC)(C)CN', 'CCC(CC)NC', 'CC(CCN)CC', 'CC(CNCC)CC',
#         'CC(C)(CN)CC', 'CC(C)N(C)CC', 'CC(N(C)C)CC', 'CCCCC(N)C', 'CCCCC(C)N',
#         'CC(CC)CCN', 'CC(CCC)CCN', 'CC(CC)(CC)CN', 'CCC(CC)CCN', 'CC(CC)(C)CCN',
#         'CCC(CC)CCN', 'CC(CCN)CC', 'CC(C)(CC)CCN', 'CCC(CC)CCN', 'CC(CCNCC)CC',
#         'CC(CNCC)CC', 'CC(C)(CN)CC', 'CC(C)N(C)CC', 'CC(N(C)C)CC', 'CCCCC(N)C',
#         'CCCCC(C)N', 'CC(CC)CC(C)N', 'CCC(CC)CC(C)N', 'CCCC(CC)N'
#     ]
# })


def form_imine(aldehyde_ketone, amine):
    #data preprocessing
    aldehyde_ketone_smiles = ds_1.loc[aldehyde_ketone, 'SMILES']
    amine_smiles = ds_2.loc[amine, 'SMILES']
    aldehyde_ketone_mol = Chem.MolFromSmiles(aldehyde_ketone_smiles)
    
    # Convert amine SMILES to RDKit molecule
    amine_mol = Chem.MolFromSmiles(amine_smiles)
    imine_mol = Chem.RWMol(aldehyde_ketone_mol)

    #search for C=O
    pattern = Chem.MolFromSmarts('C=O')
    matches = imine_mol.GetSubstructMatches(pattern)

    #set the bond order to 1 (single bond)
    bond_idx = imine_mol.GetBondBetweenAtoms(matches[0][0], matches[0][1])
    bond_idx.SetBondType(Chem.BondType.SINGLE)

    #give oxygen a negative charge
    imine_mol.GetAtomWithIdx(matches[0][1]).SetFormalCharge(-1)

    imine_mol.UpdatePropertyCache(strict=False)

    #add the amine part to the same mol object but not yet connect them
    imine_mol = Chem.CombineMols(imine_mol,amine_mol)
    
    # Get the atom index for the nitrogen in the amine
    nitrogen_idx = amine_mol.GetSubstructMatch(Chem.MolFromSmarts('N'))[0] + len(aldehyde_ketone_mol.GetAtoms())

    #change status to an editable mol object
    ed_imine = Chem.EditableMol(imine_mol)

    #connect the carbonyl carbon with the nitrogen
    ed_imine.AddBond(matches[0][0], nitrogen_idx, order=Chem.BondType.SINGLE)
    
    #change back to mol object
    imine_mol = ed_imine.GetMol()

    # Assign a positive charge to the nitrogen
    imine_mol.GetAtomWithIdx(nitrogen_idx).SetFormalCharge(1)

    imine_mol.UpdatePropertyCache(strict=False)  
    
    # Convert the modified imine molecule back to SMILES
    imine_smiles = Chem.MolToSmiles(imine_mol)
    return imine_smiles

rows = len(ds_1) * len(ds_2)
columns = ['ketone_aldehyde', 'amine','SMILES_intermediate_1', 'Energy_amine', 'Energy_ketone_aldehyde', 'Energy_intermediate', 'Energy']

df_res = pd.DataFrame(index=range(rows), columns=columns)


for i in range(0, len(ds_1)):
    for j in range(0, len(ds_2)):
        HOME = os.getcwd()
        aldehyde_ketone_index = i
        amine_index = j
        imine_smiles = form_imine(aldehyde_ketone_index, amine_index)
        df_res.at[i * len(ds_2) + j, 'ketone_aldehyde'] = ds_1.loc[i, 'SMILES']
        df_res.at[i * len(ds_2) + j, 'amine'] = ds_2.loc[j, 'SMILES']
        df_res.at[i * len(ds_2) + j, 'SMILES_intermediate_1'] = imine_smiles
        
        #create xyz files for each amine + aldehyde/ketone + intermediate combo
        smiles_strings = [ds_1.loc[i, 'SMILES'], ds_2.loc[j, 'SMILES'], imine_smiles]
        os.system(f'mkdir -p rxn_{i}_{j}')
        os.chdir(f'rxn_{i}_{j}')
        # make xyz_files
        for k in range(3):
            mol = Chem.MolFromSmiles(smiles_strings[k])
            mol = Chem.AddHs(mol)
            Chem.EmbedMolecule(mol)  # Generate a 3D conformation
            Chem.UFFOptimizeMolecule(mol)  # Optimize the geometry using the UFF force field
            if k == 0:
                Chem.rdmolfiles.MolToXYZFile(mol, f'rxn_{i}_{j}_ketone_aldehyde.xyz')
                m = ade.Molecule(f'rxn_{i}_{j}_ketone_aldehyde.xyz')
                m.optimise(method=ade.methods.XTB())
                # m.calc_thermo()
                df_res.at[i * len(ds_2) + j, 'Energy_ketone_aldehyde'] = m.energy.to('kcal')
            elif k == 1:
                Chem.rdmolfiles.MolToXYZFile(mol, f'rxn_{i}_{j}_amine.xyz')
                m = ade.Molecule(f'rxn_{i}_{j}_amine.xyz')
                m.optimise(method=ade.methods.XTB())
                # m.calc_thermo()
                df_res.at[i * len(ds_2) + j, 'Energy_amine'] = m.energy.to('kcal')
            else:
                Chem.rdmolfiles.MolToXYZFile(mol, f'rxn_{i}_{j}_intermediate.xyz')
                m = ade.Molecule(f'rxn_{i}_{j}_intermediate.xyz')
                m.optimise(method=ade.methods.XTB())
                # m.calc_thermo()
                df_res.at[i * len(ds_2) + j, 'Energy_intermediate'] = m.energy.to('kcal')
        reaction_energy = df_res.loc[i * len(ds_2) + j, 'Energy_amine'] - (df_res.loc[i * len(ds_2) + j, 'Energy_ketone_aldehyde'] + df_res.loc[i * len(ds_2) + j, 'Energy_amine'])
        df_res.at[i * len(ds_2) + j, 'Energy'] = reaction_energy
        os.chdir(HOME)

print(df_res)