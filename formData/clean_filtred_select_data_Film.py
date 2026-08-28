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

exp_est.to_csv("OnlyExpData.csv", index=False)
ss = exp_est["state"].unique()
print(ss)
solvents = ["CC1=CC=CC=C1","CC1CCCO1","C1CCCO1","C1CCCO2","ClC([H])([H])Cl","CC1OCCC1", "CC1CCOC1",
            "CCCCCC", "C1CCCCC1", "CC1=CC=CC=C2", "C1CCCO3", "C1CCCO4", "C1CCCO5", "C1CCCO6", "CC1=CC=CC=C3",
            "CC1=CC=CC=C4", "CC(Cl)Cl"]

molecule_in_film = exp_est[(exp_est["state"] == "Film")]

print(molecule_in_film)
print(molecule_in_film["state"].unique())

smile = pd.read_csv(r"C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\smilesMolecules.csv",
                   encoding='latin1')
print(smile)

col_with_came = molecule_in_film.columns[12] 
bring_column = ['raw_value','raw_units','experimental', 'state']
bring_column.append(col_with_came)

# Join the two datasets based on the 'nickname' column
merge_smiles = pd.merge(
    smile,
    molecule_in_film[['nickname']+bring_column],
    left_on = 'nickname',
    right_on = 'nickname',
    how = 'inner'
)
merge_name = pd.merge(
    smile,
    molecule_in_film[['nickname']+bring_column],
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
df.to_csv("FilmData.csv", index=False)
