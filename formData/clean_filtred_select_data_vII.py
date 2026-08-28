import rdkit
import pandas as pd
import numpy as np

# ******************************************************************************************************** #
# ------------------------------------ GENERATE EXPERIMENTAL DATASET ------------------------------------- #
# ________________________________________________________________________________________________________ #

data = pd.read_csv(r"C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\singlet-tripletEnergy.csv",
                   encoding='latin1')
#print(data.head())

exp_est = data[data["experimental"] == 1]
print(exp_est)

exp_est.to_csv("OnlyExperimentalData.csv", index=False)
ss = exp_est["state"].unique()
print(ss)
molecule_in_solution = exp_est[(exp_est["state"] != "Film") & (exp_est["state"] != "Solid")]
print(molecule_in_solution)
print(molecule_in_solution["state"].unique())

smile = pd.read_csv(r"C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\smilesMolecules.csv",
                   encoding='latin1')
print(smile)

# Adding solvent properties as descriptors to the dataset (dielectric constant, Reichardt Polarity ET(30), and Kamlet-Taft polarity parameter)
solvent_properties = {
    "CC1=CC=CC=C1":       {"eps": 2.38,  "ET30": 33.9, "kamlet_pi": 0.55}, # Toluene
    "CC1CCCO1":           {"eps": 7.58,  "ET30": 37.4, "kamlet_pi": 0.58}, # 2-MeTHF
    "C1CCCO1":            {"eps": 4.33,  "ET30": 37.4, "kamlet_pi": 0.58}, # THF
    "C1CCCO2":            {"eps": 4.33,  "ET30": 37.4, "kamlet_pi": 0.58}, # THF
    "ClC([H])([H])Cl":    {"eps": 9.08,  "ET30": 39.1, "kamlet_pi": 0.58}, # DCM
    "CC1OCCC1":           {"eps": 7.58,  "ET30": 37.4, "kamlet_pi": 0.58}, # 1-MeTHF
    "CC1CCOC1":           {"eps": 7.58,  "ET30": 37.4, "kamlet_pi": 0.58}, # 2-MeTHF
    "CCCCCC":             {"eps": 1.92,  "ET30": 31.0, "kamlet_pi": 0.55}, # n-Hexane
    "C1CCCCC1":           {"eps": 2.02,  "ET30": 31.0, "kamlet_pi": 0.55}, # Cyclohexane
    "CC1=CC=CC=C2":       {"eps": 2.38,  "ET30": 33.9, "kamlet_pi": 0.55}, # Benzene
    "C1CCCO3":            {"eps": 4.33,  "ET30": 37.4, "kamlet_pi": 0.58}, # THF
    "C1CCCO4":            {"eps": 4.33,  "ET30": 37.4, "kamlet_pi": 0.58}, # THF
    "C1CCCO5":            {"eps": 4.33,  "ET30": 37.4, "kamlet_pi": 0.58}, # THF
    "C1CCCO6":            {"eps": 4.33,  "ET30": 37.4, "kamlet_pi": 0.58}, # THF
    "CC1=CC=CC=C3":       {"eps": 2.38,  "ET30": 33.9, "kamlet_pi": 0.55}, # Toluene
    "CC1=CC=CC=C4":       {"eps": 2.38,  "ET30": 33.9, "kamlet_pi": 0.55}, # Toluene
    "CC(Cl)Cl":           {"eps": 9.08,  "ET30": 39.1, "kamlet_pi": 0.58}, # DCE
}

df_solvent = molecule_in_solution.copy()
df_solvent = df_solvent["state"].map(solvent_properties).apply(pd.Series)
#df_solvent = df_solvent.drop(columns=[0])

molecule_in_solution = pd.concat([molecule_in_solution, df_solvent], axis=1)

col_with_came = molecule_in_solution.columns[12] 
bring_column = ['raw_value','raw_units','experimental', 'eps', 'ET30', 'kamlet_pi']
bring_column.append(col_with_came)

# Join the two datasets based on the 'nickname' column
merge_smiles = pd.merge(
    smile,
    molecule_in_solution[['nickname']+bring_column],
    left_on = 'nickname',
    right_on = 'nickname',
    how = 'inner'
)
merge_name = pd.merge(
    smile,
    molecule_in_solution[['nickname']+bring_column],
    left_on = 'name.compound',
    right_on = 'nickname',
    how = 'inner'
)
#print(merge_smiles)
#print(merge_name)

# Drop duplicates and unnecessary columns, then save the final dataset
df_finil = pd.concat([merge_smiles, merge_name], ignore_index=True)
df = df_finil.drop(columns=['Unnamed: 0','nickname_x','nickname_y'])
print(df_finil)
df = df[df['raw_value'] < 0.3]
df.to_csv("SolutionDataII.csv", index=False)
