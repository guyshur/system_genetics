import pandas as pd
import numpy as np

hypothalamus_path = 'hypothalamus.txt'
genotypes_path = 'BXD.geno'

hypothalamus_expression_df = pd.read_csv(hypothalamus_path,
                                         sep='\t',
                                         comment='#',
                                         index_col=0,
                                         )
#hypothalamus_expression_df.columns = hypothalamus_expression_df.iloc[0]
hypothalamus_expression_df = hypothalamus_expression_df[1:]

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