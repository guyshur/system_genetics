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