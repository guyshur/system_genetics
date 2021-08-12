import pandas as pd
import numpy as np
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
