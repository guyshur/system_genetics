import os

import pandas as pd
from statsmodels.stats.multitest import fdrcorrection as bh_procedure

def fdr_analysis(results_df):
    index_prev_name = results_df.index.name
    vec = results_df.copy()

    vec.index = vec.index.set_names(['index'])
    vec = vec.reset_index().melt(id_vars=['index'], var_name='cols', value_name='pval')

    vec_for_bh = vec[~vec.pval.isna()]
    vec_corrected = vec_for_bh.copy()
    bools, corrected = bh_procedure(vec_for_bh.pval)
    vec_corrected.pval = corrected

    # wide=vec.pivot(index='Locus', columns='pheno', values='pval')
    wide_corrected = vec_corrected.pivot(index='index', columns='cols', values='pval')
    wide_corrected.index = wide_corrected.index.set_names([index_prev_name])

    return wide_corrected


def get_fdr_corrected_qtl(qtl_raw):

    path = './data/qtls_fdr.csv'
    if os.path.isfile(path):
        print("Reading existing qtls fdr file")
        qtls_fdr = pd.read_csv(path, index_col=0)
    else:
        print("qtls fdr corrected file does not exist, running fdr")
        qtls_fdr = fdr_analysis(qtl_raw)
        qtls_fdr.to_csv(path)

    return qtls_fdr



def get_fdr_corrected_eqtl(hypo_eqtl_raw, liver_eqtl_raw):

    path = './data/hypothalamus_eqtls_fdr.csv'
    if os.path.isfile(path):
        print("Reading existing hypo eqtl fdr file")
        hypo_eqtl_fdr = pd.read_csv(path, index_col=0)
    else:
        print("hypo eqtl fdr corrected file does not exist, running fdr")
        hypo_eqtl_fdr = fdr_analysis(hypo_eqtl_raw)
        hypo_eqtl_fdr.to_csv(path)

    path = './data/liver_eqtls_fdr.csv'
    if os.path.isfile(path):
        print("Reading existing liver eqtl fdr file")
        liver_eqtl_fdr = pd.read_csv(path, index_col=0)
    else:
        print("liver eqtl fdr corrected file does not exist, running fdr")
        liver_eqtl_fdr = fdr_analysis(liver_eqtl_raw)
        liver_eqtl_fdr.to_csv(path)

    return hypo_eqtl_fdr, liver_eqtl_fdr