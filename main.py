import pandas as pd
import numpy as np

from scipy.stats import linregress, stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as bh_procedure
import pickle
import os
import preprocessing


def run_QTL_analysis(genotypes_numeric, phenotypes_df):
    geno_df = genotypes_numeric.copy()
    pheno_df = phenotypes_df.copy()

    # Filter out hetrozygous samples and encode genotypes as either 1 or 2.
    mask = geno_df != 1
    genotypes_homozygous = geno_df[mask]

    pheno_vs_geno_df = pd.DataFrame(index=genotypes_homozygous.index, columns=pheno_df.index)

    for _, pheno in pheno_df.iterrows():
        y = pheno
        y = y.dropna()
        common = genotypes_homozygous.columns.intersection(y.index)
        y = y[common]
        X = genotypes_homozygous[common]
        reg_models = X.apply(lambda x: stats.linregress(x, y), axis=1)
        pvals = reg_models.apply(lambda model: model.pvalue)
        pheno_vs_geno_df[pheno.name] = pvals

    return pheno_vs_geno_df

def fdr_QTL_analysis(QTL_results_df):
    vec = QTL_results_df.reset_index().melt(id_vars=['Locus'], var_name='pheno', value_name='pval')
    vec_for_bh = vec[~vec.pval.isna()]
    vec_corrected = vec_for_bh.copy()
    bools, corrected = bh_procedure(vec_for_bh.pval)
    vec_corrected.pval = corrected

    # wide=vec.pivot(index='Locus', columns='pheno', values='pval')
    wide_corrected = vec_corrected.pivot(index='Locus', columns='pheno', values='pval')
    return wide_corrected



def multiple_test_correction(df: pd.DataFrame) -> None:
    flattened_results = []
    for _, p_values in df.iterrows():
        flattened_results.extend(p_values)
    bools, flattened_results = bh_procedure(flattened_results)
    num_genes = len(df.columns)
    for i in range(len(df.index)):
        df.iloc[i, :] = flattened_results[i * num_genes:(i + 1) * num_genes]


def filter_genes_without_associations(results: pd.DataFrame, is_significant: pd.DataFrame) -> None:
    """
    removes genes that are not associated with any SNP from the dataframe
    :param results: a num_snps X num_genes dataframe of multiple-testing-corrected linear regression p values
    :param is_significant: a boolean dataframe with the same shape as results, with True if the p value is significant
    :return: Nothing - the change is in place
    """
    weak_genes = []
    for gene in is_significant:
        if not any(is_significant[gene].tolist()):
            weak_genes.append(gene)
    results.drop(columns=weak_genes, inplace=True)


def linear_regression(genotypes: pd.DataFrame, expression: pd.DataFrame,
                      shared_strains: list, path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            results = pickle.load(f)
    else:
        results: pd.DataFrame = \
            genotypes[shared_strains].apply(
                lambda x: expression[shared_strains].apply(lambda y: linregress(x, y)[3], axis=1),
                axis=1)
        with open(path, 'wb+') as f:
            pickle.dump(results, f)
    return results


def num_significant_genes_per_snp(df_without_weakly_associated_genes: pd.DataFrame) -> pd.Series:
    """
    returns a series of SNPs and the number of genes significantly associated with each snp
    :param df_without_weakly_associated_genes: the expression dataframe after filtering genes without significant
    eQTLs
    :return: a pandas series with index = snp names and values = number of genes associated with the SNPs
    """
    index = []
    num_genes = []
    is_significant: pd.DataFrame = df_without_weakly_associated_genes <= 0.05
    for snp,genes in is_significant.iterrows():
        index.append(snp)
        try:
            num_genes.append(genes.value_counts()[True])
        except KeyError:
            num_genes.append(0)
    return pd.Series(index=index,data=num_genes)

hypothalamus_path = 'hypothalamus.txt'
liver_path = 'liver.txt'
genotypes_path = 'genotypes.xls'
phenotypes_path = 'phenotypes.xls'
genotypes_df: pd.DataFrame = pd.read_excel(genotypes_path)
genotypes_df.columns = genotypes_df.loc[0, :]
genotypes_df = genotypes_df.iloc[1:, :-3]
genotypes_df = genotypes_df.set_index('Locus')
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
phenotypes_df.to_csv('data/phenotypes_filtered_df.csv')

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

# Write intermediate files, after preprocessing

liver_expression_df.to_csv('./data/liver_expression_preproc_df.csv')
hypothalamus_expression_df.to_csv('./data/hypothalamus_expression_preproc_df.csv')
genotypes_df.to_csv('./data/genotypes_preproc_df.csv')
genotypes_numeric.to_csv('./data/genotypes_preproc_numeric_df.csv')

QTL_results_not_corrected_df = run_QTL_analysis(genotypes_numeric, phenotypes_df)
QTL_results_not_corrected_df.to_csv('./data/QTL_results_not_corrected_df.csv')
QTL_results__corrected_df = fdr_QTL_analysis(QTL_results_not_corrected_df)

liver_eqtl_results = linear_regression(genotypes_numeric, liver_expression_df,
                                       genotype_liver_strains, 'bin/liver_eqtl.pkl')

multiple_test_correction(liver_eqtl_results)
liver_significant: pd.DataFrame = liver_eqtl_results <= 0.05
print('number of liver eQTLs: {}'.format(liver_significant.stack().value_counts()[True]))
filter_genes_without_associations(liver_eqtl_results, liver_significant)
num_of_significant_genes = num_significant_genes_per_snp(liver_eqtl_results)
plt.hist(num_of_significant_genes,bins=30, label='liver eQTL distribution', alpha=0.5)

hypothalamus_eqtl_results = linear_regression(genotypes_numeric, hypothalamus_expression_df,
                                       genotype_hypothalamus_strains, 'bin/hypothalamus_eqtl.pkl')
multiple_test_correction(hypothalamus_eqtl_results)
hypothalamus_significant: pd.DataFrame = hypothalamus_eqtl_results <= 0.05
print('number of hypothalamus eQTLs: {}'.format(hypothalamus_significant.stack().value_counts()[True]))
filter_genes_without_associations(hypothalamus_eqtl_results, hypothalamus_significant)
num_of_significant_genes = num_significant_genes_per_snp(hypothalamus_eqtl_results)
plt.hist(num_of_significant_genes,bins=30,label='hypothalamus eQTL distribution',alpha=0.5)
plt.legend()
plt.title('Distribution of eQTLs')
plt.xlabel('number of SNPs')
plt.ylabel('number of significantly associated genes')
plt.savefig('dist.png')