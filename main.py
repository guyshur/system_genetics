import pandas as pd
import numpy as np

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
hypothalamus_expression_df_no_metadata = hypothalamus_expression_df.iloc[34:,:]


liver_expression_df_no_metadata = liver_expression_df.iloc[47:-1,:]
# Set minimal maximal expression value to be the .025 percentile of all maximal values
# Liver data
thr = liver_expression_df_no_metadata.max(axis = 1).quantile(.025)
liver_expression_df_no_metadata = liver_expression_df_no_metadata.apply(lambda col: col.astype(float))
mask = liver_expression_df_no_metadata.apply(lambda row: row.max() > thr, axis = 1)
liver_expression_df_no_metadata = liver_expression_df_no_metadata.loc[mask]
# Hypo data
thr = hypothalamus_expression_df_no_metadata.max(axis = 1).quantile(.025)
hypothalamus_expression_df_no_metadata = hypothalamus_expression_df_no_metadata.apply(lambda col: col.astype(float))
mask = hypothalamus_expression_df_no_metadata.apply(lambda row: row.max() > thr, axis = 1)
hypothalamus_expression_df_no_metadata = hypothalamus_expression_df_no_metadata.loc[mask]
phenotypes_df = phenotypes_df.astype(float)

#hypothalamus_expression_df.columns = hypothalamus_expression_df.iloc[0]
hypothalamus_expression_df = hypothalamus_expression_df[1:]
liver_expression_df = liver_expression_df[1:]
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
hypothalamus_expression_df.rename(
    columns={strain: '_'.join(strain.strip('"').split('_')[:2]) for strain in liver_expression_df.columns},
inplace=True)
print(hypothalamus_expression_df_no_metadata)