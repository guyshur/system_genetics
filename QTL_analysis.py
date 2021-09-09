import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
import seaborn as sns

# Local imports
from utils import fdr_analysis, get_fdr_corrected_qtl, get_fdr_corrected_eqtl
CHOSEN_PHENOTYPE = 'Blood chemistry, cardiovascular system: Hematocrit of 14-week old males [%]'

def local_regress(x, y):
    x = x.dropna()
    common = x.index.intersection(y.dropna().index)
    y = y[common]
    return stats.linregress(x, y)


def run_QTL_analysis(genotypes_numeric, phenotypes_df):
    geno_df = genotypes_numeric.copy()
    pheno_df = phenotypes_df.copy()
    # Filter out hetrozygous samples and encode genotypes as either 1 or 2.
    mask = geno_df != 1
    genotypes_homozygous = geno_df[mask]
    pheno_vs_geno_df = pd.DataFrame(index=genotypes_homozygous.index, columns=pheno_df.index)
    print("Now running QTL regressions")
    for _, pheno in pheno_df.iterrows():
        y = pheno.dropna()
        common = genotypes_homozygous.columns.intersection(y.index)
        y = y[common]
        X = genotypes_homozygous[common]
        X = X.astype(float)
        reg_models = X.apply(lambda x: local_regress(x, y), axis=1)
        pvals = reg_models.apply(lambda model: model.pvalue)
        pheno_vs_geno_df[pheno.name] = pvals

    print("Finished QTL analysis")
    return pheno_vs_geno_df

def plot_qtl_analysis(qtl_fdr, genotypes_df):
    CHOSEN_PHENOTYPE = 'Blood chemistry, cardiovascular system: Hematocrit of 14-week old males [%]'
    plt.figure(figsize=(16, 8))
    p_values = -1 * np.log10(qtl_fdr.loc[:, CHOSEN_PHENOTYPE])
    # genotypes_df.set_index("Locus", inplace=True)
    genotypes_df['p_val'] = p_values
    genotypes_df.reset_index(inplace=True)
    chr_max_position = genotypes_df.groupby("Chr_Build37").max()["Build37_position"]

    genotypes_df['plot_pos'] = genotypes_df.apply(
        lambda snp: snp["Build37_position"] / chr_max_position.loc[snp["Chr_Build37"]] + snp["Chr_Build37"] - 0.5,
        axis=1)

    for chrom, chrom_snps in genotypes_df.groupby("Chr_Build37"):
        ax = sns.scatterplot(data=chrom_snps, x=chrom_snps.plot_pos, y=chrom_snps.p_val)

    ax.axhline(y=-np.log10(0.05), linewidth=1.5, color='gray', linestyle="--")
    plt.text(20.15, -np.log10(0.05) + 0.1, r"$\alpha = 0.05$", fontsize=10, color='gray')

    plt.title('Manhattan plot for phenotype:\n{}'.format(CHOSEN_PHENOTYPE))
    plt.xlabel("Chromosome number", fontsize=11)
    plt.ylabel("-log10 p_val", fontsize=11)
    ax.set_xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    plt.savefig('./qtl_manhattan_plot.png')
    plt.show()

if __name__ == '__main__':
    genotypes_df = pd.read_csv('./data/genotypes_preproc_df.csv', index_col=0)
    genotypes_numeric = pd.read_csv('./data/genotypes_preproc_numeric_df.csv', index_col=0)
    phenotypes_df = pd.read_csv('data/phenotypes_filtered_df.csv', index_col=0)

    path = './bin/qtl.pkl'
    if os.path.isfile(path):
        print("Using existing qtl file")
        qtls = pd.read_pickle(path)
    else:
        print("No existing qtl file, running qtl analysis")
        qtls = run_QTL_analysis(genotypes_numeric, phenotypes_df)
        qtls.to_pickle(path)

    qtl_fdr = qtls.copy()
    qtl_fdr = fdr_analysis(qtl_fdr)

    is_significant = qtl_fdr <= 0.05

    reduced_qtl = qtl_fdr[is_significant].dropna(axis = 1, how='all').dropna(axis = 0, how='all')

    num_significant_qtls = reduced_qtl.notna().sum().sum()

    print(f"After filtering non significant QTLs in the FDR-corrected data, {num_significant_qtls} snp-traits remain significant.")

    print("Plotting an exmaple manhattan plot for a selected phenotype:\n"
          f"{CHOSEN_PHENOTYPE}")
    plot_qtl_analysis(qtl_fdr, genotypes_df)

