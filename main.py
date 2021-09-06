import pandas as pd
import numpy as np

from scipy.stats import linregress, stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as bh_procedure
import pickle
import os
import preprocessing

np.seterr('raise')
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


# nearby SNPs table for causality
path = 'bin/cis_snps.pkl'
if os.path.isfile(path):
    with open(path, 'rb') as f:
         is_cis = pickle.load(f)
else:
    genotypes_df['Chr_Build37'] = genotypes_df['Chr_Build37'].astype(int)
    genotypes_df['Build37_position'] = genotypes_df['Build37_position'].astype(int)
    dist = 2000000
    is_cis = pd.DataFrame(index=genotypes_df.index, columns=genotypes_df.index,dtype=bool, data=[
        [
            False for _ in range(len(genotypes_df.index))
        ] for _ in range(len(genotypes_df.index))
    ])
    for snp_1 in is_cis.index:
        for snp_2 in is_cis.columns:
            if genotypes_df.at[snp_1,'Chr_Build37'] == genotypes_df.at[snp_2,'Chr_Build37'] and \
                    genotypes_df.at[snp_1, 'Build37_position'] - dist <= \
                    genotypes_df.at[snp_2, 'Build37_position'] <= \
                    genotypes_df.at[snp_1, 'Build37_position'] + dist:
                is_cis.at[snp_1, snp_2] = True
            else:
                is_cis.at[snp_1, snp_2] = False
    with open(path, 'wb+') as f:
        pickle.dump(is_cis, f)
print('number of cis snps: {}'.format(is_cis.stack().value_counts()[True]))

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
phenotypes_df = phenotypes_df.loc[~phenotypes_df.index.duplicated(),:]
phenotypes = [x for x in phenotypes_df.index if 'blood' in x.lower() and 'male' in x.lower() and 'female' not in x.lower()]
phenotypes_df = phenotypes_df.loc[phenotypes,:]
print('num phenotypes: {}'.format(len(phenotypes)))

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



path = 'bin/qtl.pkl'
if os.path.isfile(path):
        with open(path, 'rb') as f:
            qtls = pickle.load(f)
else:
    common_geno_pheno = [x for x in phenotypes_df.columns if x in genotypes_numeric.columns]
    shared_geno = genotypes_numeric.loc[:,common_geno_pheno]
    shared_pheno = phenotypes_df.loc[:,common_geno_pheno]
    assert shared_geno.columns.tolist() == shared_pheno.columns.tolist()
    qtls = pd.DataFrame(index=shared_geno.index.tolist(), columns=shared_pheno.index.tolist(),dtype=float, data=np.zeros((
        len(shared_geno.index), len(shared_pheno.index)
    )))
    assert all([isinstance(x, str) for x in shared_geno.index.tolist()])
    assert all([isinstance(x, str) for x in shared_pheno.index.tolist()])
    to_drop = set()
    for snp, genotypes in shared_geno.iterrows():
        for phenotype, phenotype_values in shared_pheno.iterrows():
            mask = phenotype_values.notna()
            assert phenotype_values.index.tolist() == genotypes.index.tolist()
            phenotypes_cp = phenotype_values.copy()[mask]
            genotypes_cp = genotypes.copy()[mask]
            assert phenotypes_cp.index.tolist() == genotypes_cp.index.tolist()
            try:
                p_val = linregress(genotypes_cp, phenotypes_cp)[3]
                try:
                    assert 0 <= p_val <= 1
                except AssertionError:
                    print(p_val)
                    raise AssertionError
                assert isinstance(p_val, float)
                qtls.at[snp,phenotype] = p_val
            except FloatingPointError:  # at least one of the groups (B or D) is empty, can't do regression
                print(phenotype)
                to_drop.add(phenotype)

    qtls.drop(columns=list(to_drop),inplace=True)
    assert not any(qtls.isna().values.flatten().tolist())
    assert all([isinstance(float(x), float) for x in qtls.values.flatten().tolist()])
    with open(path, 'wb+') as f:
        pickle.dump(qtls, f)
for x in qtls.index:
    for y in qtls.columns:
        try:
            assert isinstance(qtls.at[x,y], float)
        except AssertionError:
            print(x,y)
            print(qtls.at[x,y])
            exit()
multiple_test_correction(qtls)
qtls_significant = qtls <= 0.05



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

cis_phenotype_liver = pd.DataFrame(index=qtls.columns,columns=liver_significant.columns, dtype=bool,data=[
    [
        False for _ in range(len(liver_significant.columns))
    ] for _ in range(len(qtls.columns))
])
cis_phenotype_hypothalamus = pd.DataFrame(index=qtls.columns,columns=hypothalamus_significant.columns, dtype=bool,data=[
    [
        False for _ in range(len(hypothalamus_significant.columns))
    ] for _ in range(len(qtls.columns))
])

print('Finding cis significant qtl and liver eqtl')
for snp_1 in qtls_significant.index:
    for snp_2 in liver_significant.index:
        if is_cis.at[snp_1, snp_2]:
            genes = liver_significant.loc[snp_1,:]
            genes = genes[genes == True]
            phenos = qtls_significant.loc[snp_2,:]
            phenos = phenos[phenos == True]
            for gene in genes.index:
                for pheno in phenos.index:
                    cis_phenotype_liver.at[pheno, gene] = True
print('number of pairs of cis significant qtls and liver eQTLs: {}'.format(cis_phenotype_liver.stack().value_counts()[True]))
print(len(cis_phenotype_liver.index)*len(cis_phenotype_liver.columns))
print('Finding cis significant qtl and hypothalamus eqtl')
for snp_1 in qtls_significant.index:
    for snp_2 in hypothalamus_significant.index:
        if is_cis.at[snp_1, snp_2]:
            genes = hypothalamus_significant.loc[snp_1,:]
            genes = genes[genes == True]
            phenos = qtls_significant.loc[snp_2,:]
            phenos = phenos[phenos == True]
            for gene in genes.index:
                for pheno in phenos.index:
                    cis_phenotype_hypothalamus.at[pheno, gene] = True
print('number of pairs of cis significant qtls and hypothalamus eQTLs: {}'.format(cis_phenotype_hypothalamus.stack().value_counts()[True]))
