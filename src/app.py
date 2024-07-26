import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, callback, State
from astropy.stats import sigma_clipped_stats as scs
from dash.exceptions import PreventUpdate
import warnings
import time
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore", message = ".*FITS standard.*")
warnings.filterwarnings("ignore", message = ".*RADECSYS.*")
warnings.filterwarnings("ignore", message = ".*invalid value encountered in sqrt.*")
warnings.filterwarnings("ignore", message = ".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message = ".*scalar divide.*")
warnings.filterwarnings("ignore", message = ".*PerformanceWarning.*")
warnings.filterwarnings("ignore", message = ".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", message = ".*Input data contains invalid values.*")


######################## defs #############################################
####################### ROTATE ############################################            
def rotate(qutheta,theta_key = 'theta'):
    rotated_q,rotated_u = np.around(np.dot(rotmat(qutheta[('across_offsets',theta_key)]),\
                                           qutheta.loc[pd.IndexSlice[:,['q_scm','u_scm']]]),decimals = 10)
    return rotated_q,rotated_u
def rotmat(theta):
    return [[np.cos(2*theta),np.sin(2*theta)],[-np.sin(2*theta),np.cos(2*theta)]]
def rotate_exp(df_to_be_rotated,tk = 'theta'):
    rotation_mind = pd.MultiIndex.from_product([['across_offsets'],['rotated_q','rotated_u']])
    rotateddf = df_to_be_rotated.loc[:,pd.IndexSlice[:,['q_scm','u_scm',tk]]].apply(rotate,axis = 1,result_type = 'expand',**{'theta_key' : tk})
    rotateddf.columns = rotation_mind
    return rotateddf

####################### CLEAN & EXPAND DF #################################
def expand_df(combined_df,sig = 2):
    # print(sig)

    # # combine all positions
    # df_combined_frag = pd.concat([pd.concat([combined_df[chip][pos]], keys=[(chip,pos)], names=['CHIP','POS','row_index']) for chip in chiplist for pos in poslist], axis = 0)

    # calculate ellipticity,q/u_scm (sub inlau, inlaq) and avg phi_l
    combined_df.loc[:,('across_offsets','ellipticity')] = combined_df.loc[:,('across_offsets','b')]/combined_df.loc[:,('across_offsets','a')]

    qstats = scs(combined_df.loc[:,pd.IndexSlice[:,['q1','q2']]],axis = 1, sigma = sig)
    ustats = scs(combined_df.loc[:,pd.IndexSlice[:,['u1','u2']]],axis = 1, sigma = sig)
    combined_df.loc[:,('across_offsets','q_scm')] = qstats[0]
    combined_df.loc[:,('across_offsets','u_scm')] = ustats[0]

    combined_df.loc[:,('across_offsets','q_scerr')] = qstats[2]
    combined_df.loc[:,('across_offsets','u_scerr')] = ustats[2]

    combined_df.loc[:,('across_offsets','q_weights')] = 1/(qstats[2]**2)
    combined_df.loc[:,('across_offsets','u_weights')] = 1/(ustats[2]**2)

    combined_df.loc[:,('across_offsets','avg_phil')] = 0.5 * np.arctan2(combined_df.loc[:,('across_offsets','u_scm')],combined_df.loc[:,('across_offsets','q_scm')])

    return combined_df.copy()
def droprows_bin(df_to_be_cleaned,stars = True,ellipticity = True,npixthresh = 20,higherror = None):
    if stars:
        df_to_be_cleaned = df_to_be_cleaned.drop(df_to_be_cleaned[df_to_be_cleaned[('across_offsets','object_type')]=='star'].index,inplace = False)
    if higherror is not None:
        df_to_be_cleaned = df_to_be_cleaned.drop(df_to_be_cleaned[df_to_be_cleaned[('across_offsets','q_scerr')]>higherror].index,inplace = False)
        df_to_be_cleaned = df_to_be_cleaned.drop(df_to_be_cleaned[df_to_be_cleaned[('across_offsets','u_scerr')]>higherror].index,inplace = False)
    if ellipticity:
        df_to_be_cleaned = df_to_be_cleaned.drop(df_to_be_cleaned[df_to_be_cleaned[('across_offsets','ellipticity')].isnull()].index,inplace = False)
    if npixthresh is not None:
        df_to_be_cleaned = df_to_be_cleaned.drop(df_to_be_cleaned[df_to_be_cleaned[('across_offsets','npix')]<npixthresh].index,inplace = False)
    return df_to_be_cleaned.copy()
def add_bins(df_to_be_binned,binsize = 100):
    df_to_be_binned = df_to_be_binned.sort_values(('across_offsets','ellipticity'),ascending = False)
    df_to_be_binned.loc[:,('across_offsets','ell_bins')] = np.floor(np.arange(df_to_be_binned.shape[0])/binsize).astype(int).astype(str)
    return df_to_be_binned.copy()

####################### WEIGHTED CALCS ####################################
def expand_averages(rotated_df_for_exp):
    # print(rotated_df_for_exp['across_offsets'])
    rotated_df_for_exp.loc[:,('across_offsets','q_x_weights')] = rotated_df_for_exp.loc[:,('across_offsets','rotated_q')]*rotated_df_for_exp.loc[:,('across_offsets','q_weights')]
    rotated_df_for_exp.loc[:,('across_offsets','u_x_weights')] = rotated_df_for_exp.loc[:,('across_offsets','rotated_u')]*rotated_df_for_exp.loc[:,('across_offsets','u_weights')]
    binned_stats = rotated_df_for_exp['across_offsets'].groupby('ell_bins')[['q_x_weights','q_weights','u_x_weights','u_weights']].sum()
    # binned_stats.loc[:,'q_wa'] =binned_stats['q_x_weights']/binned_stats['q_weights']
    # binned_stats.loc[:,'u_wa'] =binned_stats['u_x_weights']/binned_stats['u_weights']
    binned_stats['q_wa'] =binned_stats['q_x_weights']/binned_stats['q_weights']
    binned_stats['u_wa'] =binned_stats['u_x_weights']/binned_stats['u_weights']
    rotated_df_for_exp.loc[:,('across_offsets','binavg_q')] = binned_stats['q_wa'].loc[rotated_df_for_exp.loc[:,('across_offsets','ell_bins')].to_numpy()].to_numpy()
    rotated_df_for_exp.loc[:,('across_offsets','binavg_u')] = binned_stats['q_wa'].loc[rotated_df_for_exp.loc[:,('across_offsets','ell_bins')].to_numpy()].to_numpy()
    rotated_df_for_exp.loc[:,('across_offsets','weighted_q_diff_sq')] = ((rotated_df_for_exp.loc[:,('across_offsets','rotated_q')]-rotated_df_for_exp.loc[:,('across_offsets','binavg_q')])**2)*(rotated_df_for_exp.loc[:,('across_offsets','q_weights')])
    rotated_df_for_exp.loc[:,('across_offsets','weighted_u_diff_sq')] = ((rotated_df_for_exp.loc[:,('across_offsets','rotated_u')]-rotated_df_for_exp.loc[:,('across_offsets','binavg_u')])**2)*(rotated_df_for_exp.loc[:,('across_offsets','u_weights')])
    binned_stats_err = rotated_df_for_exp['across_offsets'].groupby('ell_bins')[['weighted_q_diff_sq','q_weights','weighted_u_diff_sq','u_weights']].sum()
    binned_stats_err['weighted_err_q'] =np.sqrt(binned_stats_err['weighted_q_diff_sq']/binned_stats_err['q_weights'])
    binned_stats_err['weighted_err_u'] =np.sqrt(binned_stats_err['weighted_u_diff_sq']/binned_stats_err['u_weights'])
    binned_stats['x_center'] = rotated_df_for_exp['across_offsets'].groupby('ell_bins')['ellipticity'].median()
    
    binned_stats = pd.concat([binned_stats,binned_stats_err[['weighted_err_q','weighted_err_u']]],axis = 1)
    
    return binned_stats

###########################################################################

####################### setup init ########################################
# init_t,init_npix,init_errorthresh,init_bins,init_sigexp,init_tkinput = '2p25',None,None,100,2,'theta'

####################### get data ##########################################
def calc_scstats_and_avgphil(t,sig_exp):
    # get_df
    basedf_to_expand = dict_basedf[t]
    # create scm columns
    exp = expand_df(basedf_to_expand,sig_exp)
    return exp
def rotate(df_exp,tk_input = 'theta'):
    # rotate
    rotated_qu = rotate_exp(df_exp, tk = tk_input)
    df_exp[('across_offsets','rotated_q')],df_exp[('across_offsets','rotated_u')] = \
        rotated_qu[('across_offsets','rotated_q')],rotated_qu[('across_offsets','rotated_u')]
    return df_exp
def clean_and_bin(exp_df,npix,errorthresh,bins):
    # clean and bin
    exp_clean = droprows_bin(exp_df, npixthresh=npix, higherror = errorthresh,binsize = bins)
    # calculate binned stats
    binned_exp_clean = add_bins(exp_clean,bins)
    binned_stats = expand_averages(binned_exp_clean)
    return exp_clean, binned_stats
def read_json_sbl(dict_of_dicts):
    return {k:pd.DataFrame.from_dict(dict_df) for k,dict_df in dict_of_dicts.items()}
def get_basedf(t_pull):
    with open("dict_dfpoldata_rot_"+t_pull+".pkl",'rb') as inp:
        basedf = pickle.load(inp)
    # combine all positions
    df_combined_frag = pd.concat([pd.concat([basedf[chip][pos]], keys=[(chip,pos)], names=['CHIP','POS','row_index'])\
                                  for chip in chiplist for pos in poslist], axis = 0)
    return df_combined_frag
def get_qudf(c,p,ri,df_reset):
    qs = df_reset[(df_reset[('across_offsets','CHIP')]==c)&(df_reset[('across_offsets','POS')]==p)&(df_reset[('across_offsets','row_index')] == ri)].loc[:,pd.IndexSlice[:,['CHIP','POS','row_index','q1','q2','errq1','errq2']]].iloc[0].unstack(level = 1)#['u_scm'].iloc[0]
    us = df_reset[(df_reset[('across_offsets','CHIP')]==c)&(df_reset[('across_offsets','POS')]==p)&(df_reset[('across_offsets','row_index')] == ri)].loc[:,pd.IndexSlice[:,['CHIP','POS','row_index','u1','u2','erru1','erru2']]].iloc[0].unstack(level = 1)
    return {'q':qs.to_dict(), 'u':us.to_dict()}

# dict_basedf = {t:get_basedf(t_pull = t) for t in tlist}

# get init datasets
# init_rotated_df = rotate(calc_scstats_and_avgphil(init_t,init_sigexp),init_tkinput)
# init_combined_rows = init_rotated_df.index
# init_combined_columns = init_rotated_df.columns
# init_rotated_df_for_store = init_rotated_df.reset_index(drop = True).T.reset_index(drop = True).T
# init_df_poldata_exp_forplot, init_df_binned_stats_forplot = clean_and_bin(init_rotated_df,init_npix,init_errorthresh,init_bins)
# init_chip,init_pos,init_rowindex = init_df_binned_stats_forplot.iloc[0].name

# init_reset_df_for_functions = init_df_poldata_exp_forplot.reset_index(col_level = 1,col_fill= 'across_offsets')
# init_col_names = init_reset_df_for_functions.columns
# init_reset_df_for_store = init_reset_df_for_functions.T.reset_index(drop = True).T
# init_qudict_for_store = get_qudf(init_chip, init_pos, init_rowindex, init_reset_df_for_store)

app = dash.Dash(__name__, use_pages = True,) # external_stylesheets=[dbc.themes.QUARTZ]

server = app.server

app.layout = html.Div(
    [   
        # main app store
        ## dcc.Store(id = 'store_data_clickcount'),
        ## dcc.Store(id = 'store_object_clickcount'),
        
        dcc.Store(id = 'global_df_rotated',),
        dcc.Store(id = 'global_rotated_rows',),
        dcc.Store(id = 'global_rotated_cols',),
        dcc.Store(id = 'global_df_filtered',),
        dcc.Store(id = 'global_filtered_rows',),
        dcc.Store(id = 'global_filtered_cols',),
        dcc.Store(id = 'global_qudict',),
        dcc.Store(id = 'global_df_binned',),
        dcc.Store(id = 'global_df_statedict',),
        dcc.Store(id = 'global_object_statedict',),

        # main app framework
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink(page['name'], href=page['path'])) for page in dash.page_registry.values()
            ],
            brand="AanalitiQ",
            brand_href="https://www.linkedin.com/in/ambar-qadeer",
            color= 'rgba(130, 83, 18, 0.8)',
            # width = 'auto',
            dark=True,
            fluid = True,
            class_name = 'd-flex w-100'
        ),

        html.Div(
            html.H1("Object Statistics and Deepdive",
                    style = {'width':'100vh', 'textAlign':'center', 'paddingTop':'60px',},className="position-absolute top-50 start-50 translate-middle pb-5",
            ),
            style = {'height':200,'width':'100vw',}, className="position-relative text-uppercase text-light text-center",
        ),
        # html.Div([
        #     dcc.Link(page['name']+"  |  ", href=page['path'])
        #     for page in dash.page_registry.values()
        # ]),
        # html.Hr(),

        # content of each page
        dash.page_container
    ], style = {'width':'100%', 'padding':'0px', 'margin':'0px'}
)


if __name__ == "__main__":
    app.run(debug=True, port = 8003)
