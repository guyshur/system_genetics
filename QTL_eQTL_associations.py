import os
from typing import List
import seaborn as sns

import pandas as pd
from utils import fdr_analysis
from main import multiple_test_correction
from main import num_significant_genes_per_snp, filter_genes_without_associations
import matplotlib.pyplot as plt
import main

NUM_SIMULATIONS = 500

def create_eqtl_sampling_based_distribution(qtl, eqtl1, eqtl2, NUM_SAMPLES=10, RE_FDR=False):

    sig_eqtl_dist: List = []

    print(f"Running {NUM_SIMULATIONS} simulations")
    for i in range(1, NUM_SIMULATIONS+1):
        sample_qtls = qtl.sample(n=NUM_SAMPLES).index

        _eqtl1 = eqtl1.loc[sample_qtls, :]
        _eqtl2 = eqtl2.loc[sample_qtls, :]

        if RE_FDR:
            _eqtl1 = fdr_analysis(_eqtl1)
            _eqtl2 = fdr_analysis(_eqtl2)

        res = (
            (_eqtl1 <= 0.05).sum().sum(),
            (_eqtl2 <= 0.05).sum().sum()
        )

        sig_eqtl_dist.append(res)

        if i % 100 == 0:
            print(f"Simulation {i}/{NUM_SIMULATIONS} completed.")

    return sig_eqtl_dist


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


def get_fdr_corrected_qtl(qtl_raw):

    path = './data/qtls.csv'
    if os.path.isfile(path):
        print("Reading existing qtls fdr file")
        qtls_fdr = pd.read_csv(path, index_col=0)
    else:
        print("qtls fdr corrected file does not exist, running fdr")
        qtls_fdr = fdr_analysis(hypo_eqtl_raw)
        qtls_fdr.to_csv(path)

    return qtls_fdr


def plot_empirical_distributions(dist_df,sig_eqtl_hypo_from_qtl, sig_eqtl_liver_from_qtl, num_snps,  RE_FDR=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.kdeplot(dist_df.liver_eqtl_sig, fill=True, ax=ax1)
    ax1.axvline(dist_df.liver_eqtl_sig.quantile(.95), color='red', alpha=0.3)
    ax1.text(sig_eqtl_liver_from_qtl, 0.0015, f"{sig_eqtl_liver_from_qtl}", multialignment='center', size='small', color='black', )
    ax1.scatter(sig_eqtl_liver_from_qtl, 0.0005, marker='o', s=40, edgecolors='black', alpha=.6, linewidths=0.5)
    ax1.set_xlabel('num of significant eQTLs - Liver data')

    sns.kdeplot(dist_df.hypo_eqtl_sig, fill=True, label='eQTLs distribution', ax=ax2)
    ax2.axvline(dist_df.hypo_eqtl_sig.quantile(.95), color='red', alpha=0.3, label='.95 percentile')
    ax2.text(sig_eqtl_hypo_from_qtl, 0.0015, f"{sig_eqtl_hypo_from_qtl}", horizontalalignment='center',
             verticalalignment='baseline', size='small', color='black')
    ax2.scatter(sig_eqtl_hypo_from_qtl, 0.0005, marker='o', s=40, edgecolors='black',
                alpha=.6, linewidths=0.5,  label='eQTL based on QTL')

    ax2.set_xlabel('num of significant eQTLs - Hypothalamus data')
    plt.suptitle(f'Empirical distribution of the number of significant eQTLs per {num_snps} SNPs')
    plt.legend(loc='upper right')
    plt.legend()
    if RE_FDR:
        plt.savefig('empirical_eQTLs_density_RE_FDR.png')

    else:
        plt.savefig('empirical_eQTLs_density.png')

    return

if __name__ == '__main__':
    hypo_eqtl_raw = pd.read_pickle('/Users/d_private/_git/system_genetics/bin/hypothalamus_eqtl.pkl')
    liver_eqtl_raw = pd.read_pickle('/Users/d_private/_git/system_genetics/bin/liver_eqtl.pkl')
    qtl_raw = pd.read_pickle('/Users/d_private/_git/system_genetics/bin/qtl.pkl')

    # Preform fdr on qtl data
    qtl_fdr_only_sig = get_fdr_corrected_qtl(qtl_raw)
    sig_qtl_snp_names = set(qtl_fdr_only_sig.SNP)
    num_snps = len(sig_qtl_snp_names)
    print(sig_qtl_snp_names)

    print(f"After filtering non significant QTLs in the FDR-corrected data, {qtl_fdr_only_sig.shape[0]} snp-pehnotypes pairs remain significant.\n"
          f"Number of SNPs: {len(sig_qtl_snp_names)}")

    print("Preforming fdr on eqtl datasets")
    hypo_eqtl_fdr, liver_eqtl_fdr = get_fdr_corrected_eqtl(hypo_eqtl_raw, liver_eqtl_raw)

    sig_eqtl_liver_from_qtl = liver_eqtl_fdr.loc[sig_qtl_snp_names,:]
    sig_eqtl_liver_from_qtl = (sig_eqtl_liver_from_qtl <= 0.05).sum().sum()
    print(sig_eqtl_liver_from_qtl)

    sig_eqtl_hypo_from_qtl = hypo_eqtl_fdr.loc[sig_qtl_snp_names,:]
    sig_eqtl_hypo_from_qtl = (sig_eqtl_hypo_from_qtl <= 0.05).sum().sum()
    print(sig_eqtl_hypo_from_qtl)

    # Create sampling based distribution of significant eqtls from QTL sub sampling
    print("eQTL sampling without RE-FDR")
    dist = create_eqtl_sampling_based_distribution(qtl=qtl_raw, eqtl1=hypo_eqtl_fdr, eqtl2=liver_eqtl_fdr,
                                                   NUM_SAMPLES=num_snps)

    dist_df = pd.DataFrame(data=dist, columns=['hypo_eqtl_sig', 'liver_eqtl_sig'])
    dist_df.to_csv('./data/eqtl_dist_dist.csv')

    plot_empirical_distributions(dist_df, sig_eqtl_hypo_from_qtl, sig_eqtl_liver_from_qtl, num_snps)

    # Repeat the same proccess only this time preform FDR only on the subset of 'num_snps' in the eqtl data
    print("eQTL sampling with RE-FDR")
    dist = create_eqtl_sampling_based_distribution(qtl=qtl_raw, eqtl1=hypo_eqtl_raw, eqtl2=liver_eqtl_raw,
                                                   NUM_SAMPLES=num_snps, RE_FDR=True)

    dist_df_FDR = pd.DataFrame(data=dist, columns=['hypo_eqtl_sig', 'liver_eqtl_sig'])
    dist_df_FDR.to_csv('./data/re_fdr_eqtl_dist_dist.csv')

    sig_eqtl_hypo_from_qtl_FDR = (fdr_analysis(hypo_eqtl_raw.loc[sig_qtl_snp_names,:]) <= 0.05).sum().sum()
    sig_eqtl_liver_from_qtl_FDR = (fdr_analysis(liver_eqtl_raw.loc[sig_qtl_snp_names,:]) <= 0.05).sum().sum()

    print(sig_eqtl_hypo_from_qtl_FDR, sig_eqtl_liver_from_qtl_FDR)
    plot_empirical_distributions(dist_df_FDR, sig_eqtl_hypo_from_qtl_FDR, sig_eqtl_liver_from_qtl_FDR, num_snps,
                                 RE_FDR=True)


    # print(hypo_eqtl)