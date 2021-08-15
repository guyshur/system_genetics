import pandas as pd
import numpy as np

from scipy.stats import linregress
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as bh_procedure
import pickle
import os
import preprocessing
from sklearn.linear_model import LinearRegression

hypothalamus_path = 'hypothalamus.txt'
liver_path = 'liver.txt'
genotypes_path = 'BXD.geno'
phenotypes_path = 'phenotypes.xls'
hypothalamus_expression_df = pd.read_csv(hypothalamus_path,
                                         sep='\t',
                                         comment='#',
                                         index_col=0,
                                         )
liver_expression_df = pd.read_csv(liver_path,
                                         sep='\t',
                                         comment='#',
                                         index_col=0,
                                         )
phenotypes_df = pd.read_excel(phenotypes_path,index_col=1)
phenotypes_df = phenotypes_df[phenotypes_df.columns[4:]]



liver_expression_df = preprocessing.preprocess_liver_data(liver_expression_df)
hypothalamus_expression_df = preprocessing.preprocess_hypo_data(hypothalamus_expression_df)

genotypes_df = pd.read_csv(genotypes_path,
                           sep='\t',
                           comment='#',
                           index_col=1
                           )
curr = []
redundant = []
for index,row in genotypes_df.iterrows():
    row = row.tolist()[3:]
    if row == curr:
        redundant.append(index)
    else:
        curr = row
genotypes_df.drop(index=redundant,inplace=True)
liver_expression_df.rename(
    columns={strain: '_'.join(strain.strip("'").split('_')[:2]) for strain in liver_expression_df.columns},
inplace=True)
liver_expression_df.rename(
    columns={strain: '_'.join(strain.strip('"').split('_')[:2]) for strain in liver_expression_df.columns},
inplace=True)

phenotypes = [x for x in phenotypes_df.index if 'ethanol' in x.lower() and 'male' in x.lower() and 'female' not in x.lower()]
phenotypes_df = phenotypes_df.loc[phenotypes,:]
hypothalamus_expression_df = hypothalamus_expression_df[
    [column for column in hypothalamus_expression_df if column.startswith('BXD') and column.endswith('_M')]]
liver_expression_df = liver_expression_df[
    [column for column in liver_expression_df if column.startswith('BXD') and column.endswith('_M')]]
liver_expression_df.rename(
    columns={strain: strain.split('_')[0] for strain in liver_expression_df.columns},
inplace=True)
hypothalamus_expression_df.rename(
    columns={strain: strain.split('_')[0] for strain in hypothalamus_expression_df.columns},
inplace=True)
genotypes_numeric = genotypes_df.iloc[:,3:].applymap(lambda genotype: 0 if genotype == 'B' else 1 if genotype == 'H' else 2)

genotype_liver_strains = genotypes_numeric.columns.intersection(liver_expression_df.columns)
genotype_hypothalamus_strains = genotypes_numeric.columns.intersection(hypothalamus_expression_df.columns)
liver_expression_df = liver_expression_df[genotype_liver_strains]
hypothalamus_expression_df = hypothalamus_expression_df[genotype_hypothalamus_strains]

try:
    os.mkdir("./data/")
except FileExistsError:
    pass
liver_expression_df.to_csv('./data/liver_expression_preproc_df.csv')
hypothalamus_expression_df.to_csv('./data/hypothalamus_expression_preproc_df.csv')
genotypes_df.to_csv('./data/genotypes_preproc_df.csv')

exit()

# linear regression
if os.path.isfile('bin/liver_eqtl.pkl'):
    with open('bin/liver_eqtl.pkl','rb') as f:
        liver_eqtl_results = pickle.load(f)
else:
    liver_eqtl_results: pd.DataFrame = \
        genotypes_numeric[genotype_liver_strains].apply(lambda x: liver_expression_df.apply(lambda y: linregress(x, y)[3], axis=1),
                                  axis=1)
    with open('bin/liver_eqtl.pkl','wb+') as f:
        pickle.dump(liver_eqtl_results,f)

if os.path.isfile('bin/hypothalamus_eqtl.pkl'):
    with open('bin/hypothalamus_eqtl.pkl','rb') as f:
        hypothalamus_eqtl_results = pickle.load(f)
else:
    hypothalamus_eqtl_results: pd.DataFrame = \
        genotypes_numeric[genotype_hypothalamus_strains].apply(lambda x: hypothalamus_expression_df.apply(lambda y: linregress(x, y)[3], axis=1),
                                  axis=1)
    with open('bin/hypothalamus_eqtl.pkl','wb+') as f:
        pickle.dump(hypothalamus_eqtl_results,f)

# multiple test correction
flattened_results = []
for _, p_values in liver_eqtl_results.iterrows():
    flattened_results.extend(p_values)
bools, flattened_results = bh_procedure(flattened_results)
num_genes = len(liver_eqtl_results.columns)
for i in range(len(liver_eqtl_results.index)):
    liver_eqtl_results.iloc[i, :] = flattened_results[i * num_genes:(i + 1) * num_genes]
flattened_results = []
for _, p_values in hypothalamus_eqtl_results.iterrows():
    flattened_results.extend(p_values)
bools, flattened_results = bh_procedure(flattened_results)
num_genes = len(hypothalamus_eqtl_results.columns)
for i in range(len(hypothalamus_eqtl_results.index)):
    hypothalamus_eqtl_results.iloc[i, :] = flattened_results[i * num_genes:(i + 1) * num_genes]

# boolean (is statistically significant) table
liver_significant: pd.DataFrame = liver_eqtl_results <= 0.05
hypothalamus_significant: pd.DataFrame = hypothalamus_eqtl_results <= 0.05

# dropping genes without eQTLs
#TODO