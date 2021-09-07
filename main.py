import pandas as pd
import numpy as np

from scipy.stats import linregress, stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as bh_procedure
import pickle
import os
import preprocessing
from scipy.stats import pearsonr, norm

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
    for snp, genes in is_significant.iterrows():
        index.append(snp)
        try:
            num_genes.append(genes.value_counts()[True])
        except KeyError:
            num_genes.append(0)
    return pd.Series(index=index, data=num_genes)


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
    dist = 0
    is_cis = pd.DataFrame(index=genotypes_df.index, columns=genotypes_df.index, dtype=bool, data=[
        [
            False for _ in range(len(genotypes_df.index))
        ] for _ in range(len(genotypes_df.index))
    ])
    for snp_1 in is_cis.index:
        for snp_2 in is_cis.columns:
            if genotypes_df.at[snp_1, 'Chr_Build37'] == genotypes_df.at[snp_2, 'Chr_Build37'] and \
                    genotypes_df.at[snp_1, 'Build37_position'] == genotypes_df.at[snp_2, 'Build37_position']:
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
phenotypes_df = pd.read_excel(phenotypes_path, index_col=1)
phenotypes_df = phenotypes_df[phenotypes_df.columns[4:]]

liver_expression_df = preprocessing.preprocess_liver_data(liver_expression_df)
hypothalamus_expression_df = preprocessing.preprocess_hypo_data(hypothalamus_expression_df)

curr = []
redundant = []
for index, row in genotypes_df.iterrows():
    row = row.tolist()[3:]
    if row == curr:
        redundant.append(index)
    else:
        curr = row
genotypes_df.drop(index=redundant, inplace=True)
liver_expression_df.rename(
    columns={strain: '_'.join(strain.strip("'").split('_')[:2]) for strain in liver_expression_df.columns},
    inplace=True)
liver_expression_df.rename(
    columns={strain: '_'.join(strain.strip('"').split('_')[:2]) for strain in liver_expression_df.columns},
    inplace=True)
phenotypes_df = phenotypes_df.loc[~phenotypes_df.index.duplicated(), :]
phenotypes = [x for x in phenotypes_df.index if
              'blood' in x.lower() and 'male' in x.lower() and 'female' not in x.lower()]
phenotypes_df = phenotypes_df.loc[phenotypes, :]
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
genotypes_numeric = genotypes_df.iloc[:, 3:].applymap(
    lambda genotype: 0 if genotype == 'B' else 1 if genotype == 'H' else 2)

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
    shared_geno = genotypes_numeric.loc[:, common_geno_pheno]
    shared_pheno = phenotypes_df.loc[:, common_geno_pheno]
    assert shared_geno.columns.tolist() == shared_pheno.columns.tolist()
    qtls = pd.DataFrame(index=shared_geno.index.tolist(), columns=shared_pheno.index.tolist(), dtype=float,
                        data=np.zeros((
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
                qtls.at[snp, phenotype] = p_val
            except FloatingPointError:  # at least one of the groups (B or D) is empty, can't do regression
                to_drop.add(phenotype)

    qtls.drop(columns=list(to_drop), inplace=True)
    assert not any(qtls.isna().values.flatten().tolist())
    assert all([isinstance(float(x), float) for x in qtls.values.flatten().tolist()])
    with open(path, 'wb+') as f:
        pickle.dump(qtls, f)
for x in qtls.index:
    for y in qtls.columns:
        try:
            assert isinstance(qtls.at[x, y], float)
        except AssertionError:
            print(x, y)
            print(qtls.at[x, y])
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
plt.hist(num_of_significant_genes, bins=30, label='liver eQTL distribution', alpha=0.5)

hypothalamus_eqtl_results = linear_regression(genotypes_numeric, hypothalamus_expression_df,
                                              genotype_hypothalamus_strains, 'bin/hypothalamus_eqtl.pkl')
multiple_test_correction(hypothalamus_eqtl_results)
hypothalamus_significant: pd.DataFrame = hypothalamus_eqtl_results <= 0.05

print('number of hypothalamus eQTLs: {}'.format(hypothalamus_significant.stack().value_counts()[True]))
filter_genes_without_associations(hypothalamus_eqtl_results, hypothalamus_significant)
num_of_significant_genes = num_significant_genes_per_snp(hypothalamus_eqtl_results)
plt.hist(num_of_significant_genes, bins=30, label='hypothalamus eQTL distribution', alpha=0.5)
plt.legend()
plt.title('Distribution of eQTLs')
plt.xlabel('number of SNPs')
plt.ylabel('number of significantly associated genes')
plt.savefig('dist.png')

# causality analysis - finding cis snp-gene-pheno triplets
cis_phenotype_liver = pd.DataFrame(index=qtls.columns, columns=liver_significant.columns, dtype=bool, data=[
    [
        False for _ in range(len(liver_significant.columns))
    ] for _ in range(len(qtls.columns))
])
cis_phenotype_hypothalamus = pd.DataFrame(index=qtls.columns, columns=hypothalamus_significant.columns, dtype=bool,
                                          data=[
                                              [
                                                  False for _ in range(len(hypothalamus_significant.columns))
                                              ] for _ in range(len(qtls.columns))
                                          ])

liver_triplets = set()
print('Finding cis significant qtl and liver eqtl')
for snp in qtls_significant.index:
    if is_cis.at[snp, snp]:
        genes = liver_significant.loc[snp, :]
        genes = genes[genes == True]
        phenos = qtls_significant.loc[snp, :]
        phenos = phenos[phenos == True]
        for gene in genes.index:
            for pheno in phenos.index:
                liver_triplets.add((snp, gene, pheno))
print('number of pairs of cis significant qtls and liver eQTLs: {}'.format(len(liver_triplets)))
print('Finding cis significant qtl and hypothalamus eqtl')
hypo_triplets = set()
for snp in qtls_significant.index:
    if is_cis.at[snp, snp]:
        genes = hypothalamus_significant.loc[snp, :]
        genes = genes[genes == True]
        phenos = qtls_significant.loc[snp, :]
        phenos = phenos[phenos == True]
        for gene in genes.index:
            for pheno in phenos.index:
                hypo_triplets.add((snp, gene, pheno))
print('number of pairs of cis significant qtls and liver eQTLs: {}'.format(len(hypo_triplets)))
liver_triplets = list(liver_triplets)
hypo_triplets = list(hypo_triplets)
# shared strains only
shared_strains_liver = set(genotypes_numeric.columns) & set(phenotypes_df.columns) & set(liver_expression_df.columns)
shared_strains_hypo = set(genotypes_numeric.columns) & set(phenotypes_df.columns) & set(
    hypothalamus_expression_df.columns)
genotypes_numeric = genotypes_numeric[shared_strains_liver | shared_strains_hypo]
phenotypes_df = phenotypes_df[shared_strains_liver | shared_strains_hypo]
liver_expression_df = liver_expression_df[shared_strains_liver]
hypothalamus_expression_df = hypothalamus_expression_df[shared_strains_hypo]


def generate_triplet_dataframe(snp: str, gene: str, pheno: str, expression_matrix: pd.DataFrame, strains: set) -> pd.DataFrame:
    snp_col = genotypes_numeric.loc[snp, strains]
    expression_col = expression_matrix.loc[gene, strains]
    pheno_col = phenotypes_df.loc[pheno, strains]
    df = pd.DataFrame(index=strains)
    df['genotype'] = snp_col
    df['expression'] = expression_col
    df['phenotype'] = pheno_col
    df.dropna(inplace=True)
    df.drop(index=df[df['genotype'] == 1].index, inplace=True)
    df['mu_R|L'] = np.nan
    df['sigma_R|L'] = np.nan
    df['p_R|L'] = np.nan
    df['p_C|R'] = np.nan
    df['mu_C|L'] = np.nan
    df['sigma_C|L'] = np.nan
    df['p_C|L'] = np.nan
    df['mu_R|C'] = np.nan
    df['sigma_R|C'] = np.nan
    df['mu_C|R'] = np.nan
    df['sigma_C|R'] = np.nan
    df['p_R|C'] = np.nan
    df['p_C|R'] = np.nan
    df['m1_likelihood'] = np.nan
    df['m2_likelihood'] = np.nan
    df['m3_likelihood'] = np.nan
    return df


# calculating parameters
def get_R_variable_params_given_L(triplet_dataframe: pd.DataFrame):
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'mu_R|L'] = \
        np.mean(triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'expression'].values)
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'mu_R|L'] = \
        np.mean(triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'expression'].values)
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'sigma_R|L'] = \
        np.std(triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'expression'].values)
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'sigma_R|L'] = \
        np.std(triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'expression'].values)


def get_C_variable_params_given_L(triplet_dataframe: pd.DataFrame):
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'mu_C|L'] = \
        np.mean(triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'phenotype'].values)
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'mu_C|L'] = \
        np.mean(triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'phenotype'].values)
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'sigma_C|L'] = \
        np.std(triplet_dataframe.loc[triplet_dataframe['genotype'] == 2, 'phenotype'].values)
    triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'sigma_C|L'] = \
        np.std(triplet_dataframe.loc[triplet_dataframe['genotype'] == 0, 'phenotype'].values)


def get_rho(triplet_dataframe: pd.DataFrame):
    pheno_data = triplet_dataframe['phenotype'].values
    mask = ~np.isnan(pheno_data)
    pheno_data = pheno_data[mask]
    if len(pheno_data) < 3:
        return None
    expression_data = triplet_dataframe['expression'].values[mask]
    rho, p = pearsonr(expression_data, pheno_data)
    return rho


def get_C_variable_params_given_R(triplet_dataframe: pd.DataFrame):
    mu_r = np.mean(triplet_dataframe['expression'])
    sigma_r = np.std(triplet_dataframe['expression'])
    mu_c = np.mean(triplet_dataframe['phenotype'])
    sigma_c = np.std(triplet_dataframe['phenotype'])
    rho = get_rho(triplet_dataframe)
    def calc_mu(r): return mu_c + rho * (sigma_c / sigma_r) * (r - mu_r)
    triplet_dataframe['sigma_C|R'] = np.sqrt((sigma_c ** 2) * (1 - (rho ** 2)))
    for strain, data in triplet_dataframe.iterrows():
        triplet_dataframe.at[strain, 'mu_C|R'] = calc_mu(triplet_dataframe.at[strain, 'expression'])


def get_R_variable_params_given_C(triplet_dataframe: pd.DataFrame):
    mu_r = np.mean(triplet_dataframe['expression'])
    sigma_r = np.std(triplet_dataframe['expression'])
    mu_c = np.mean(triplet_dataframe['phenotype'])
    sigma_c = np.std(triplet_dataframe['phenotype'])
    rho = get_rho(triplet_dataframe)
    def calc_mu(c): return mu_r + rho * (sigma_r / sigma_c) * (c - mu_c)
    triplet_dataframe['sigma_R|C'] = np.sqrt((sigma_r ** 2) * (1 - (rho ** 2)))
    for strain, data in triplet_dataframe.iterrows():
        triplet_dataframe.at[strain, 'mu_R|C'] = calc_mu(triplet_dataframe.at[strain, 'phenotype'])


def get_p_R_given_L(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_R|L'] = triplet_dataframe.apply(
        lambda row: norm(row['mu_R|L'], row['sigma_R|L']).pdf(row['expression']), axis=1)


def get_p_C_given_L(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_C|L'] = triplet_dataframe.apply(
        lambda row: norm(row['mu_C|L'], row['sigma_C|L']).pdf(row['phenotype']), axis=1)


def get_p_R_given_C(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_R|C'] = triplet_dataframe.apply(
        lambda row: norm(row['mu_R|C'], row['sigma_R|C']).pdf(row['expression']), axis=1)


def get_p_C_given_R(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_C|R'] = triplet_dataframe.apply(
        lambda row: norm(row['mu_C|R'], row['sigma_C|R']).pdf(row['phenotype']), axis=1)


def get_likelihoods(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['m1_likelihood'] = triplet_dataframe.apply(
        lambda row: 0.5 * row['p_R|L'] * row['p_C|R'], axis=1)
    triplet_dataframe['m2_likelihood'] = triplet_dataframe.apply(
        lambda row: 0.5 * row['p_C|L'] * row['p_R|C'], axis=1)
    triplet_dataframe['m3_likelihood'] = triplet_dataframe.apply(
        lambda row: 0.5 * row['p_R|L'] * row['p_C|L'], axis=1)

def get_likelihood_ratio(triplet_dataframe: pd.DataFrame):
    complete_likelihoods = [np.prod(triplet_dataframe['m1_likelihood']),
                            np.prod(triplet_dataframe['m2_likelihood']),
                            np.prod(triplet_dataframe['m3_likelihood'])]
    most_likely = np.max(complete_likelihoods)
    most_likely_index = complete_likelihoods.index(most_likely)
    complete_likelihoods.remove(most_likely)
    likelihood_ratio = most_likely/np.max(complete_likelihoods)
    return likelihood_ratio, most_likely_index+1


def _calculate_hypothesis_for_table(triplet_dataframe:pd.DataFrame):
    get_R_variable_params_given_L(triplet_dataframe)
    get_C_variable_params_given_L(triplet_dataframe)
    get_C_variable_params_given_R(triplet_dataframe)
    get_R_variable_params_given_C(triplet_dataframe)
    get_p_R_given_L(triplet_dataframe)
    get_p_C_given_L(triplet_dataframe)
    get_p_R_given_C(triplet_dataframe)
    get_p_C_given_R(triplet_dataframe)
    get_likelihoods(triplet_dataframe)
    return get_likelihood_ratio(triplet_dataframe)


def calculate_hypothesis_for_triplet(triplet: tuple, expression_dataframe: pd.DataFrame, strains: set):
    """
    returns the likelihood ratio and selected model number for a snp with associated eQTL and QTL.
    :param triplet: the snp, gene and phenotype of the triplet
    :param expression_dataframe: the epxression matrix that the gene belongs to
    :param strains: the BXD strains that will be used to the calculations
    :return: tuple of (the likelihood ratio, number of the selected model [1,2,3])
    """
    snp = triplet[0]
    gene = triplet[1]
    pheno = triplet[2]
    triplet_table: pd.DataFrame = generate_triplet_dataframe(
        snp=snp,gene=gene,pheno=pheno,expression_matrix=expression_dataframe,strains=strains)
    return _calculate_hypothesis_for_table(triplet_dataframe=triplet_table)


def get_likelihood_ratio_and_permutation_p_value(triplet: tuple, expression_dataframe: pd.DataFrame, strains: set):
    """
    returns the likelihood ratio, selected model number and permutation p value for a snp with associated eQTL and QTL.
    :param triplet: the snp, gene and phenotype of the triplet
    :param expression_dataframe: the epxression matrix that the gene belongs to
    :param strains: the BXD strains that will be used to the calculations
    :return: tuple of (the likelihood ratio, number of the selected model [1,2,3], p_value from permutation)
    """
    snp = triplet[0]
    gene = triplet[1]
    pheno = triplet[2]
    triplet_table: pd.DataFrame = generate_triplet_dataframe(
        snp=snp,gene=gene,pheno=pheno,expression_matrix=expression_dataframe,strains=strains)
    ratio, selected_model = _calculate_hypothesis_for_table(triplet_table)
    ratios = []
    for _ in range(100):
        triplet_table: pd.DataFrame = generate_triplet_dataframe(
            snp=snp, gene=gene, pheno=pheno, expression_matrix=expression_dataframe, strains=strains)
        expression = triplet_table['expression'].values
        np.random.shuffle(expression)
        triplet_table['expression'] = expression
        phenotype = triplet_table['phenotype'].values
        np.random.shuffle(phenotype)
        triplet_table['phenotype'] = phenotype
        ratios.append(_calculate_hypothesis_for_table(triplet_table)[0])
    return ratio, selected_model, len([x for x in ratios if x >= ratio])/100


def calculate_hypotheses_and_export(triplets, expression_dataframe, strains, dest):
    output = {'SNP': [], 'gene': [], 'phenotype': [], 'likelihood_ratio':[], 'model': [], 'p_value': []}
    for triplet in triplets:
        output['SNP'].append(triplet[0])
        output['gene'].append(triplet[1])
        output['phenotype'].append(triplet[2])
        ratio, model, p_val = get_likelihood_ratio_and_permutation_p_value(triplet, expression_dataframe, strains)
        output['likelihood_ratio'].append(ratio)
        output['model'].append(model)
        output['p_value'].append(p_val)
    pd.DataFrame.from_dict(output).to_csv(dest,sep='\t')


calculate_hypotheses_and_export(liver_triplets, liver_expression_df, shared_strains_liver, 'data/liver_causality.tsv')
calculate_hypotheses_and_export(hypo_triplets, hypothalamus_expression_df, shared_strains_hypo, 'data/hypothalamus_causality.tsv')
