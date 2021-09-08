import os

import pandas as pd
from main import fdr_analysis
from main import multiple_test_correction
from main import  num_significant_genes_per_snp, filter_genes_without_associations
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 100
NUM_SAMPLES = 10
def create_eqtl_sampling_based_distribution(qtl, eqtl1, eqtl2):

    dist = []

    for i in range(NUM_SIMULATIONS):
        sample_qtls = qtl.sample(n=NUM_SAMPLES).index
        _eqtl1 = eqtl1.loc[sample_qtls, :]
        _eqtl2 = eqtl2.loc[sample_qtls, :]

        res = (
            (_eqtl1<= 0.05).sum().sum(),
            (_eqtl2 <= 0.05).sum().sum()
        )

        dist.append(res)

    return dist


def get_fdr_corrected_eqtl(hypo_eqtl_raw, liver_eqtl_raw):

    path = './data/hypo_eqtl_fdr.csv'
    if os.path.isfile(path):
        print("Reading existing hypo eqtl fdr file")
        hypo_eqtl_fdr = pd.read_csv(path, index_col=0)
    else:
        print("hypo eqtl fdr corrected file does not exist, running fdr")
        hypo_eqtl_fdr = fdr_analysis(hypo_eqtl_raw)
        hypo_eqtl_fdr.to_csv('./data/hypo_eqtl_fdr.csv')

    path = './data/liver_eqtl_fdr.csv'
    if os.path.isfile(path):
        print("Reading existing liver eqtl fdr file")
        liver_eqtl_fdr = pd.read_csv(path, index_col=0)
    else:
        print("liver eqtl fdr corrected file does not exist, running fdr")
        liver_eqtl_fdr = fdr_analysis(liver_eqtl_raw)
        hypo_eqtl_fdr.to_csv('./data/liver_eqtl_fdr.csv')

    return hypo_eqtl_fdr, liver_eqtl_fdr


if __name__ == '__main__':
    hypo_eqtl_raw = pd.read_pickle('/Users/d_private/_git/system_genetics/bin/hypothalamus_eqtl.pkl')
    liver_eqtl_raw = pd.read_pickle('/Users/d_private/_git/system_genetics/bin/liver_eqtl.pkl')
    qtl_raw = pd.read_pickle('/Users/d_private/_git/system_genetics/bin/qtl.pkl')

    # Preform fdr on qtl data
    qtl_fdr = fdr_analysis(qtl_raw)
    is_significant = qtl_fdr <= 0.05

    reduced_qtl = qtl_fdr[is_significant].dropna(axis=1, how='all').dropna(axis=0, how='all')
    num_sig_qtls = reduced_qtl.notna().sum().sum()
    sig_qtl_idx = list(reduced_qtl.index)

    print(f"After filtering non significant QTLs in the FDR-corrected data, {num_sig_qtls} snp-pehnotypes remain significant.")
    print(f"Filtering eqtl fdr corrected data ")

    hypo_eqtl_fdr, liver_eqtl_fdr = get_fdr_corrected_eqtl(hypo_eqtl_raw, liver_eqtl_raw)









    # Create sampling based distribution of significant eqtls from QTL sub sampling
    # dist = create_eqtl_sampling_based_distribution(qtl=qtl_fdr, eqtl1=hypo_eqtl_raw, eqtl2=liver_eqtl_raw)
    # print(dist)



    # print(hypo_eqtl)