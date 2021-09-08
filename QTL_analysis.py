import pandas as pd
from scipy.stats import stats
from main import fdr_analysis

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


if __name__ == '__main__':
    genotypes_numeric = pd.read_csv('./data/genotypes_preproc_numeric_df.csv', index_col=0)
    phenotypes_df = pd.read_csv('data/phenotypes_filtered_df.csv', index_col=0)

    # qtls = run_QTL_analysis(genotypes_numeric, phenotypes_df)

    # qtls.to_csv('./data/temp_qtl.csv')
    qtls = pd.read_csv('./data/temp_qtl.csv', index_col=0)

    qtl_fdr = qtls.copy()
    qtl_fdr = fdr_analysis(qtl_fdr)

    is_significant = qtl_fdr <= 0.05

    reduced_qtl = qtl_fdr[is_significant].dropna(axis = 1, how='all').dropna(axis = 0, how='all')
    num_significant_qtls = reduced_qtl.notna().sum().sum()

    print(f"After filtering non significant QTLs in the FDR-corrected data, {num_significant_qtls} snp-traits remain significant.")

