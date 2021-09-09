import math

import pandas as pd
import numpy as np

from scipy.stats import linregress, stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection as bh_procedure
import pickle
import os
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

## Local modules
import preprocessing
import QTL_analysis
import utils

np.seterr('raise')


def generate_random_triplets(geno_df, expression_df, pheno_df, number_of_triplets=30):
    possible_snps = geno_df.index.tolist()
    possible_genes = expression_df.index.tolist()
    possible_phenos = pheno_df.index.tolist()
    snps_for_triplets = np.random.choice(possible_snps, size=number_of_triplets, replace=True)
    genes_for_triplets = np.random.choice(possible_genes, size=number_of_triplets, replace=True)
    phenos_for_triplets = np.random.choice(possible_phenos, size=number_of_triplets, replace=True)
    while True:
        valid = True
        for snp in snps_for_triplets:
            gen_data = geno_df.loc[snp,:].dropna().tolist()
            if all([x == 2 for x in gen_data]) or all ([x == 0 for x in gen_data]):
                valid = False
                break
        print(valid)
        if valid:
            break
        snps_for_triplets = np.random.choice(possible_snps, size=number_of_triplets, replace=True)
    return list(zip(snps_for_triplets, genes_for_triplets, phenos_for_triplets))



def norm_pdf(mu, sigma):
    return lambda x: (math.exp(-0.5*(((x-mu)/sigma)**2)))/(sigma*math.sqrt(2*math.pi))


def generate_triplet_dataframe(triplet_snp: str, triplet_gene: str, triplet_pheno: str,
                               expression_matrix: pd.DataFrame, strains: set) -> pd.DataFrame:
    snp_col = genotypes_numeric.loc[triplet_snp, strains]
    expression_col = expression_matrix.loc[triplet_gene, strains]
    pheno_col = phenotypes_df.loc[triplet_pheno, strains]
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
    return df.sort_values(by="genotype")


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
    expression_data = triplet_dataframe['expression'].values
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
    triplet_dataframe['mu_C|R'] = triplet_dataframe.apply(
        lambda row: calc_mu(row['expression']), axis=1)


def get_R_variable_params_given_C(triplet_dataframe: pd.DataFrame):
    mu_r = np.mean(triplet_dataframe['expression'])
    sigma_r = np.std(triplet_dataframe['expression'])
    mu_c = np.mean(triplet_dataframe['phenotype'])
    sigma_c = np.std(triplet_dataframe['phenotype'])
    rho = get_rho(triplet_dataframe)
    def calc_mu(c): return mu_r + rho * (sigma_r / sigma_c) * (c - mu_c)
    triplet_dataframe['sigma_R|C'] = np.sqrt((sigma_r ** 2) * (1 - (rho ** 2)))
    triplet_dataframe['mu_R|C'] = triplet_dataframe.apply(
        lambda row: calc_mu(row['phenotype']), axis=1)


def get_p_R_given_L(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_R|L'] = triplet_dataframe.apply(
        lambda row: norm_pdf(row['mu_R|L'], row['sigma_R|L'])(row['expression']), axis=1)


def get_p_C_given_L(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_C|L'] = triplet_dataframe.apply(
        lambda row: norm_pdf(row['mu_C|L'], row['sigma_C|L'])(row['phenotype']), axis=1)


def get_p_R_given_C(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_R|C'] = triplet_dataframe.apply(
        lambda row: norm_pdf(row['mu_R|C'], row['sigma_R|C'])(row['expression']), axis=1)


def get_p_C_given_R(triplet_dataframe: pd.DataFrame):
    triplet_dataframe['p_C|R'] = triplet_dataframe.apply(
        lambda row: norm_pdf(row['mu_C|R'], row['sigma_C|R'])(row['phenotype']), axis=1)


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
    triplet_dataframe.to_csv('triplet.tsv',sep='\t')
    return get_likelihood_ratio(triplet_dataframe)


def calculate_hypothesis_for_triplet(triplet: tuple, expression_dataframe: pd.DataFrame, strains: set):
    """
    returns the likelihood ratio and selected model number for a snp with associated eQTL and QTL.
    :param triplet: the snp, gene and phenotype of the triplet
    :param expression_dataframe: the epxression matrix that the gene belongs to
    :param strains: the BXD strains that will be used to the calculations
    :return: tuple of (the likelihood ratio, number of the selected model [1,2,3])
    """
    triplet_snp = triplet[0]
    triplet_gene = triplet[1]
    triplet_pheno = triplet[2]
    triplet_table: pd.DataFrame = generate_triplet_dataframe(
        triplet_snp=triplet_snp,triplet_gene=triplet_gene,triplet_pheno=triplet_pheno,
        expression_matrix=expression_dataframe,strains=strains)
    return _calculate_hypothesis_for_table(triplet_dataframe=triplet_table)


def get_likelihood_ratio_and_permutation_p_value(triplet: tuple, expression_dataframe: pd.DataFrame, strains: set):
    """
    returns the likelihood ratio, selected model number and permutation p value for a snp with associated eQTL and QTL.
    :param triplet: the snp, gene and phenotype of the triplet
    :param expression_dataframe: the epxression matrix that the gene belongs to
    :param strains: the BXD strains that will be used to the calculations
    :return: tuple of (the likelihood ratio, number of the selected model [1,2,3], p_value from permutation)
    """
    triplet_snp = triplet[0]
    triplet_gene = triplet[1]
    triplet_pheno = triplet[2]
    triplet_table: pd.DataFrame = generate_triplet_dataframe(
        triplet_snp=triplet_snp, triplet_gene=triplet_gene, triplet_pheno=triplet_pheno,
        expression_matrix=expression_dataframe, strains=strains)
    ratio, selected_model = _calculate_hypothesis_for_table(triplet_table)
    ratios = []
    for _ in range(100):
        triplet_table: pd.DataFrame = generate_triplet_dataframe(
            triplet_snp=triplet_snp, triplet_gene=triplet_gene, triplet_pheno=triplet_pheno,
            expression_matrix=expression_dataframe, strains=strains)
        expression = triplet_table['expression'].values
        np.random.shuffle(expression)
        triplet_table['expression'] = expression
        phenotype_vec = triplet_table['phenotype'].values
        np.random.shuffle(phenotype_vec)
        triplet_table['phenotype'] = phenotype_vec
        ratios.append(_calculate_hypothesis_for_table(triplet_table)[0])
    return ratio, selected_model, len([x for x in ratios if x >= ratio])/100


def calculate_hypotheses_and_export(triplets, expression_dataframe, strains, dest, organ):
    output = {'SNP': [], 'gene': [], 'phenotype': [], 'likelihood_ratio':[], 'model': [], 'p_value': []}
    total = len(triplets)
    i = 0
    for triplet in triplets:
        print('done {} / {} triplets'.format(i, total))
        i += 1
        output['SNP'].append(triplet[0])
        output['gene'].append(triplet[1])
        output['phenotype'].append(triplet[2])
        ratio, model, model_p_val = get_likelihood_ratio_and_permutation_p_value(triplet, expression_dataframe, strains)
        output['likelihood_ratio'].append(ratio)
        output['model'].append(model)
        output['p_value'].append(model_p_val)
    output['p_value'] = bh_procedure(output['p_value'])[1]
    print('exporting to csv..')
    density = gaussian_kde(output['p_value'])
    xs = np.linspace(0, 1, 2000)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.plot(xs, density(xs))
    plt.title('Causality analysis model selection\np-value distribution for significant triplets\n{} genes'.format(organ))
    plt.xlabel('p value')
    plt.savefig('{}_causality_p_value_distribution.png'.format(organ))
    plt.clf()
    pd.DataFrame.from_dict(output).to_csv(dest)


def calculate_mock_hypotheses_and_export(triplets, expression_dataframe, strains, dest):
    output = {'SNP': [], 'gene': [], 'phenotype': [], 'likelihood_ratio':[], 'model': [], 'p_value': []}
    total = len(triplets)
    i = 0
    for triplet in triplets:
        print('done {} / {} triplets'.format(i, total))
        i += 1
        output['SNP'].append(triplet[0])
        output['gene'].append(triplet[1])
        output['phenotype'].append(triplet[2])
        ratio, model, model_p_val = get_likelihood_ratio_and_permutation_p_value(triplet, expression_dataframe, strains)
        output['likelihood_ratio'].append(ratio)
        output['model'].append(model)
        output['p_value'].append(model_p_val)
    output['p_value'] = bh_procedure(output['p_value'])[1]
    print('exporting to csv..')
    pd.DataFrame.from_dict(output).to_csv(dest)


def qtls_to_csv(qtls: pd.DataFrame, dest):
    output = {'SNP': [], 'phenotype': [], 'p_value': []}
    for snp, data in qtls.iterrows():
        data = data[data <= 0.05]
        for pheno, pval in data.iteritems():
            output['SNP'].append(snp)
            output['phenotype'].append(pheno)
            output['p_value'].append(pval)
    pd.DataFrame.from_dict(output).to_csv(dest)


def eqtls_to_csv(eqtls: pd.DataFrame, dest):
    output = {'SNP': [], 'gene': [], 'p_value': []}
    for snp, data in eqtls.iterrows():
        data = data[data <= 0.05]
        for gene, pval in data.iteritems():
            output['SNP'].append(snp)
            output['gene'].append(gene)
            output['p_value'].append(pval)
    pd.DataFrame.from_dict(output).to_csv(dest)


def run_QTL_analysis(genotypes_numeric, phenotypes_df):
    geno_df = genotypes_numeric.copy()
    pheno_df = phenotypes_df.copy()
    #Filter out hetrozygous samples and encode genotypes as either 1 or 2.
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

if __name__ == '__main__':

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
    for pheno in phenotypes:
        print(pheno)
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
    genotypes_numeric = genotypes_df.iloc[:,3:].applymap(
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

    # Run QTL analysis
    qtls = QTL_analysis.run_QTL_analysis(genotypes_numeric, phenotypes_df)
    qtl_fdr = utils.fdr_analysis(qtls.copy())
    qtls_significant = qtl_fdr <= 0.05
    reduced_qtl = qtl_fdr[qtls_significant].dropna(axis=1, how='all').dropna(axis=0, how='all')
    num_significant_qtls = reduced_qtl.notna().sum().sum()
    print(f"After filtering non significant QTLs in the FDR-corrected data, {num_significant_qtls} snp-traits remain significant.")
    p_vals = qtl_fdr.values.flatten()
    density = gaussian_kde(p_vals)
    xs = np.linspace(0, 1, 2000)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.plot(xs, density(xs))
    plt.title('QTL analysis p value density')
    plt.xlabel('p value')

    plt.savefig('qtl_p_density.png')
    plt.clf()


    liver_eqtl_results = linear_regression(genotypes_numeric, liver_expression_df,
                                           genotype_liver_strains, 'bin/liver_eqtl.pkl')

    multiple_test_correction(liver_eqtl_results)
    liver_significant: pd.DataFrame = liver_eqtl_results <= 0.05
    print('number of liver eQTLs: {}'.format(liver_significant.stack().value_counts()[True]))
    print('liver eqtl candidates: {}'.format(len(liver_eqtl_results.index)*len(liver_eqtl_results.columns)))
    filter_genes_without_associations(liver_eqtl_results, liver_significant)
    hypothalamus_eqtl_results = linear_regression(genotypes_numeric, hypothalamus_expression_df,
                                                  genotype_hypothalamus_strains, 'bin/hypothalamus_eqtl.pkl')
    multiple_test_correction(hypothalamus_eqtl_results)
    hypothalamus_significant: pd.DataFrame = hypothalamus_eqtl_results <= 0.05
    print('number of hypothalamus eQTLs: {}'.format(hypothalamus_significant.stack().value_counts()[True]))
    print('hypothalamus eqtl candidates: {}'.format(len(hypothalamus_eqtl_results.index)*len(hypothalamus_eqtl_results.columns)))

    filter_genes_without_associations(hypothalamus_eqtl_results, hypothalamus_significant)

    qtls_to_csv(qtl_fdr, 'data/qtls.csv')
    eqtls_to_csv(liver_eqtl_results, 'data/liver_eqtls.csv')
    eqtls_to_csv(hypothalamus_eqtl_results, 'data/hypothalamus_eqtls.csv')
    print('liver eqtl candidates: {}'.format(len(liver_eqtl_results.index)*len(liver_eqtl_results.columns)))
    print('hypothalamus eqtl candidates: {}'.format(len(hypothalamus_eqtl_results.index)*len(hypothalamus_eqtl_results.columns)))

    p_vals = liver_eqtl_results.values.flatten()
    density = gaussian_kde(p_vals)
    density._compute_covariance()
    plt.plot(xs, density(xs), label='liver eQTL p value density')
    p_vals = hypothalamus_eqtl_results.values.flatten()
    density = gaussian_kde(p_vals)
    density._compute_covariance()
    plt.plot(xs, density(xs), label='hypothalamus eQTL p value density')
    plt.legend()
    plt.title('eQTL analysis p value density')
    plt.xlabel('p value')
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.savefig('eQTL_p_density.png')
    plt.clf()

    num_of_significant_genes = num_significant_genes_per_snp(liver_eqtl_results)
    plt.hist(num_of_significant_genes,bins=30, label='liver eQTL distribution', alpha=0.5)
    num_of_significant_genes = num_significant_genes_per_snp(hypothalamus_eqtl_results)
    plt.hist(num_of_significant_genes,bins=30,label='hypothalamus eQTL distribution',alpha=0.5)
    plt.legend()
    plt.title('number of genes associated to each SNP')
    plt.xlabel('number of SNPs')
    plt.ylabel('number of genes')
    plt.savefig('eQTL_dist.png')
    plt.clf()

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
    print('doin hypothesis')
    # shared strains only
    shared_strains_liver = set(genotypes_numeric.columns) & set(phenotypes_df.columns) & set(liver_expression_df.columns)
    shared_strains_hypo = set(genotypes_numeric.columns) & set(phenotypes_df.columns) & set(
        hypothalamus_expression_df.columns)
    genotypes_numeric = genotypes_numeric[shared_strains_liver | shared_strains_hypo]
    phenotypes_df = phenotypes_df[shared_strains_liver | shared_strains_hypo]
    liver_expression_df = liver_expression_df[shared_strains_liver]
    hypothalamus_expression_df = hypothalamus_expression_df[shared_strains_hypo]
    calculate_hypotheses_and_export(hypo_triplets, hypothalamus_expression_df, shared_strains_hypo, 'data/hypothalamus_causality.csv', 'hypothalamus')
    calculate_hypotheses_and_export(liver_triplets, liver_expression_df, shared_strains_liver, 'data/mock_liver_causality.csv', 'mock_liver')

    # generate random triplets to test pvaluue distribution
    # liver_triplets = generate_random_triplets(genotypes_df, liver_expression_df, phenotypes_df, 50)
    # calculate_hypotheses_and_export(liver_triplets, liver_expression_df, shared_strains_liver,
    #                                 'data/mock_liver_causality.csv', 'mock_liver')

