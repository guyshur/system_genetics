import pandas as pd
import numpy as np

from scipy.stats import linregress
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as bh_procedure

import preprocessing


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


liver_expression_df = preprocessing.remove_liver_metadata(liver_expression_df)
liver_expression_df = preprocessing.preprocess_expression_data(liver_expression_df)

hypothalamus_expression_df = preprocessing.remove_hyppthalamus_metadata(hypothalamus_expression_df)
hypothalamus_expression_df = preprocessing.preprocess_expression_data(hypothalamus_expression_df)
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
hypothalamus_expression_df_no_metadata = hypothalamus_expression_df[
    [column for column in hypothalamus_expression_df if column.startswith('BXD') and column.endswith('_M')]]
liver_expression_df_no_metadata = liver_expression_df[
    [column for column in liver_expression_df if column.startswith('BXD') and column.endswith('_M')]]
liver_expression_df_no_metadata.rename(
    columns={strain: strain.split('_')[0] for strain in liver_expression_df_no_metadata.columns},
inplace=True)
hypothalamus_expression_df_no_metadata.rename(
    columns={strain: strain.split('_')[0] for strain in hypothalamus_expression_df_no_metadata.columns},
inplace=True)
genotypes_numeric = genotypes_df.iloc[:,3:].applymap(lambda genotype: 0 if genotype == 'B' else 1 if genotype == 'H' else 2)

