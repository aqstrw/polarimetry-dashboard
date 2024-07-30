import numpy as np
import pandas as pd
import pickle
import plotly
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Input, Output, callback, State, clientside_callback
from astropy.stats import sigma_clipped_stats as scs
from dash.exceptions import PreventUpdate
import warnings
import time
from plotly.subplots import make_subplots
import json
warnings.filterwarnings("ignore", message = ".*FITS standard.*")
warnings.filterwarnings("ignore", message = ".*RADECSYS.*")
warnings.filterwarnings("ignore", message = ".*invalid value encountered in sqrt.*")
warnings.filterwarnings("ignore", message = ".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message = ".*scalar divide.*")
warnings.filterwarnings("ignore", message = ".*PerformanceWarning.*")
warnings.filterwarnings("ignore", message = ".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", message = ".*Input data contains invalid values.*")

chiplist = ["CHIP1","CHIP2"]
poslist = ["POS1","POS2"]
offsetlist = ["Offset0","Offset1","Offset2","Offset3","Offset4","Offset5","Offset6","Offset7"]
tlist = ['2','2p25','2p5','2p75','3']
# with open("dict_dfpoldata_rot_2p25.pkl",'rb') as inp:
#     dict_dfpoldata_rot_2p25 = pickle.load(inp)
# with open('./df_poldata_2p25_exp_clean_forplotly.pkl','rb') as inp:
#     df_poldata_2p25_final = pickle.load(inp)
# with open('./binned_stats_forplot.pkl','rb') as inp:
#     df_binned_stats_forplot = pickle.load(inp)



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

####################### PLOT SUPPORT ######################################
def get_rls(cleaned_df, column_name = 'q_scerr'):
    # logseries = np.log(cleaned_df[('across_offsets',column_name)]/np.nanmax(cleaned_df[('across_offsets',column_name)]))
    logseries = np.log(cleaned_df[column_name]/np.nanmax(cleaned_df[column_name]))
    return (logseries/np.nanmin(logseries))

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
init_t,init_npix,init_errorthresh,init_bins,init_sigexp,init_tkinput = '2p25',None,None,100,2,'theta'
init_df_statedict = {
    't':'2p25',
    'npix':None,
    'errorthresh':None,
    'bins':100,
    'sigexp':2,
    'tk':'theta',
}

####################### get data ##########################################
def calc_scstats_and_avgphil(t,sig_exp):
    # get_df
    basedf_to_expand = dict_basedf[t]
    # create scm columns
    exp = expand_df(basedf_to_expand,sig_exp)
    return exp
def rotate_df(df_exp,tk_input = 'theta'):
    # rotate
    rotated_qu = rotate_exp(df_exp, tk = tk_input)
    df_exp[('across_offsets','rotated_q')],df_exp[('across_offsets','rotated_u')] = \
        rotated_qu[('across_offsets','rotated_q')],rotated_qu[('across_offsets','rotated_u')]
    return df_exp
def clean_and_bin(exp_df,npix,errorthresh,bins):
    # clean and bin
    exp_clean = droprows_bin(exp_df, npixthresh=npix, higherror = errorthresh)
    # calculate binned stats
    binned_exp_clean = add_bins(exp_clean,bins)
    binned_stats = expand_averages(binned_exp_clean)
    return binned_exp_clean, binned_stats
def read_from_store_wrapper(dicts,rows = None,cols = None,levels = 2):
    # if dict of dicts, then iterate (in the case of qu dicts)
    if levels == 2:
        return {k:pd.DataFrame.from_dict(dict_df) for k,dict_df in dicts.items()}
    # if multiindex cols and rows, add them to the data
    elif levels == 1:
        df = pd.DataFrame.from_dict(dicts)
        if cols is not None:
            df.columns = cols
        else:
            print("columns not added")
        if rows is not None:
            df.index = rows
        else:
            print("rows not added")
        return df
def jsonify(df):
    df_reset = df.reset_index(drop = True).T.reset_index(drop=True).T
    return df_reset.to_dict()
def mindtojson(mind):
    return mind.to_frame().reset_index(drop=True).T.reset_index(drop=True).T.to_dict()
def jsontomind(json,rows = False):
    if rows:
        return pd.MultiIndex.from_frame(pd.DataFrame.from_records(json), names = ["CHIP","POS","row_index"])
    else:
        return pd.MultiIndex.from_frame(pd.DataFrame.from_records(json))


def get_qudf(c,p,ri,df_reset):
    qs = df_reset.loc[(c,p,ri),pd.IndexSlice[:,['q1','q2','errq1','errq2']]].unstack(level = 1)
    us = df_reset.loc[(c,p,ri),pd.IndexSlice[:,['u1','u2','erru1','erru2']]].unstack(level = 1)
    return {'q':qs, 'u':us}
def get_basedf(t_pull):
    with open("dict_dfpoldata_rot_"+t_pull+".pkl",'rb') as inp:
        basedf = pickle.load(inp)
    # combine all positions
    df_combined_frag = pd.concat([pd.concat([basedf[chip][pos]], keys=[(chip,pos)], names=['CHIP','POS','row_index'])\
                                  for chip in chiplist for pos in poslist], axis = 0)
    return df_combined_frag

dict_basedf = {t:get_basedf(t_pull = t) for t in tlist}

###################################### Figure Functions ######################################################
def make_stats_scatter(c,p,ri,df_reset):
    fig = make_subplots(rows=2, cols=2, shared_yaxes='rows',shared_xaxes = 'columns',vertical_spacing=0.02, horizontal_spacing=0.02, column_widths=[0.9,0.1], row_heights=[0.1,0.9]) #subplot_titles=("distribution in q-space","distribution in u-space")
    # xbins = np.histogram_bin_edges(df_reset['q_scm'])
    # fig_xhist = px.histogram(df_reset.reset_index(),x='q_scm',bins = xbins)
    counts, bins = np.histogram(df_reset['q_scm'][df_reset['q_scm'].between(-0.3,0.3)], bins='sqrt')
    # print(bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig_xhist = px.bar(x=bins, y=counts, text_auto = True, opacity = 0.5,color_discrete_sequence=['white'])

    fig_xhist.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # font_family="Courier New",
        font_color="white",
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white",
        legend=dict(
            title = 'Bin #',
            orientation='v',
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.85,
            bgcolor = 'rgba(184, 171, 178, 0.28)',
            bordercolor = 'white'),
    )

    # fig_yhist = px.histogram(df_reset.reset_index(),y='u_scm',)
    counts, bins = np.histogram(df_reset['u_scm'][df_reset['u_scm'].between(-0.3,0.3)], bins='sqrt')
    # print(bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig_yhist = px.bar(y=bins, x=counts, orientation='h')
    fig_yhist.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # font_family="Courier New",
        font_color="white",
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white",
        legend=dict(
            title = 'Bin #',
            orientation='v',
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.85,
            bgcolor = 'rgba(184, 171, 178, 0.28)',
            bordercolor = 'white'),
    )
    fig.add_trace(fig_xhist["data"][0], col = 1, row = 1,)
    fig.add_trace(fig_yhist["data"][0], col = 2, row = 2,)

    figspread = px.scatter(
        df_reset.reset_index(),
        x='q_scm',
        y ='u_scm',
        color='ell_bins',
        custom_data=['CHIP','POS','row_index'],
        color_discrete_sequence = plotly.colors.cyclical.Edge,
        )
    # figspread.update_layout(xaxis=dict(scaleanchor='y', scaleratio=1))
    figspread.update_traces(marker=dict(size=3))
    

    figspread.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # font_family="Courier New",
        font_color="white",
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white",
        legend=dict(
            title = 'Bin #',
            orientation='v',
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.85,
            bgcolor = 'rgba(184, 171, 178, 0.28)',
            bordercolor = 'white'),
        )
    
    figspread.update_layout(margin=dict(l=20, r=20, t=20, b=20),)
    figspread.update_xaxes(showgrid = False,showline=False, ) #linewidth=10, linecolor='purple',
    figspread.update_yaxes(showgrid = False, showline=False, ) #linewidth=10, linecolor='red'
    
    for trace in range(len(figspread["data"])):
        fig.add_trace(figspread["data"][trace], col = 1, row = 2,)
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # font_family="Courier New",
        font_color="white",
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white",
        legend=dict(
            title = 'Bin #',
            orientation='v',
            yanchor="top",
            y=1,
            xanchor="right",
            x=1,
            bgcolor = 'rgba(184, 171, 178, 0.28)',
            bordercolor = 'white'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    fig.add_hline(
            # x0=0,
            y=df_reset.loc[(c,p,ri),'u_scm'],#.iloc[0],
            # x1=7,
            # y1 = stats[0],
            # line=dict(color='rgba(100,0,100,0.9)',dash = 'dash',line_width = 1),
            line_color='rgba(57, 203, 251, 0.8)',
            line_dash = 'dash',
            line_width = 3,
            # xref='x',
            # yref='y',
            row = 2,
            col = 1,
        )
    fig.add_vline(
            # x0=0,
            x=df_reset.loc[(c,p,ri),'q_scm'],#.iloc[0],
            # x1=7,
            # y1 = stats[0],
            # line=dict(color='Black',dash = 'dash',line_width = 1),
            line_color=' rgba(213, 86, 135, 0.8)',
            line_dash = 'dash',
            line_width = 3,
            name = 'vline',
            # xref='x',
            # yref='y',
            row = 2,
            col = 1,
        )
    
    
    fig.update_xaxes(zerolinecolor='lightslategray',zerolinewidth=1, row=1, col = 1)
    fig.update_yaxes(zerolinecolor='lightslategray',zerolinewidth=1, row=1, col = 1)

    fig.update_xaxes(title = 'q_scm', range=[df_reset.loc[(c,p,ri),'q_scm']-0.3,df_reset.loc[(c,p,ri),'q_scm']+0.3],zerolinecolor='lightslategray',zerolinewidth=1, row=2, col = 1)
    fig.update_yaxes(title = 'u_scm',range=[df_reset.loc[(c,p,ri),'u_scm']-0.3,df_reset.loc[(c,p,ri),'u_scm']+0.3],zerolinecolor='lightslategray',zerolinewidth=1, row=2, col = 1)

    fig.update_xaxes(zerolinecolor='lightslategray',zerolinewidth=1,showticklabels = False, row=2, col = 2)
    fig.update_yaxes(zerolinecolor='lightslategray',zerolinewidth=1, row=2, col = 2)
    # fig.update_layout.yaxis.range = [df_reset.loc[(c,p,ri),'u_scm']-0.3,df_reset.loc[(c,p,ri),'u_scm']+0.3]
    # fig.update_layout.xaxis.zerolinecolor = 'lightslategray'
    # fig.update_layout.yaxis.zerolinecolor = 'lightslategray'

    fig.update_xaxes(showgrid = False,showline=False, row=1,col=1) #linewidth=10, linecolor='purple',
    fig.update_yaxes(showgrid = False, showline=False, row=1,col=1) #linewidth=10, linecolor='red'

    fig.update_xaxes(showgrid = False,showline=False, row=2,col=1) #linewidth=10, linecolor='purple',
    fig.update_yaxes(showgrid = False, showline=False, row=2,col=1)

    fig.update_xaxes(showgrid = False,showline=False, row=2,col=2) #linewidth=10, linecolor='purple',
    fig.update_yaxes(showgrid = False, showline=False, row=2,col=2)
    return fig
def make_individ_stats_fig(qu,c,p,ri,sig,qudf_dict,df_reset):
    qudf = qudf_dict[qu]
    # print(qudf['Offset0':'Offset7'])
    scm = df_reset.loc[(c,p,ri),qu+'_scm']#.iloc[0]
    scerr = df_reset.loc[(c,p,ri),qu+'_scerr']#.iloc[0]
    # print(scm,scerr)
    fig = px.scatter(qudf,
        x=offsetlist,
        y = qu + '1',
        error_y = 'err' + qu + '1',
        # height = 270,
        # labels=dict(y="<"+str(qu)+">", color="Ellipticity bins"),
    )
    # fig1.show()
    fig2 = px.scatter(qudf,
        x=offsetlist,
        y = qu+'2',
        error_y = 'err' + qu + '2',
        color_discrete_sequence=['red'],
        # height = 270,
        # custom_data=[qu,c,p,ri],
        # labels=dict(x="Offsets", y="<"+str(qu)+">", color="Ellipticity bins"),
    )
    # if qu == 'u':
    fig.update_layout(xaxis_title=None, yaxis_title = None) #yaxis_title = "<"+str(qu)+">"
    # else:
    #     fig.update_layout(xaxis_title=None, yaxis_title = "<"+str(qu)+">")
    # fig.update()

    fig.add_trace(fig2['data'][0])
    fig.add_hline(
            # x0=0,
            y=0,
            # x1=7,
            # y1 = stats[0],
            # line=dict(color='Black',dash = 'dash',line_width = 1),
            # color='Black',
            line_dash = 'solid',
            line_width = 2,
            # xref='x',
            # yref='y',
        )
    fig.add_hline(
            # x0=0,
            y=scm,
            # x1=7,
            # y1 = stats[0],
            # line=dict(color='darkgreen',dash = 'solid',),
            line_color='rgba(153, 74, 246, 0.8)',
            line_dash = 'dash',
            line_width = 0.7,
            # xref='x',
            # yref='y',
        )
    fig.add_annotation(
        x=6.5,
        y=scm,
        text = qu+"_scm = {}".format(np.around(scm,4)),
        # textposition = 'top center',
        showarrow=True,
        bgcolor = 'green',
        font = {'color':'white'},
        opacity = 0.5,
        xanchor="right",
        )
    fig.add_hrect(
            # x0=0,
            y0 = scm-(sig*scerr),
            # x1=7,
            y1 = scm+(sig*scerr),
            # line=dict(color=None,dash = 'dash'),
            fillcolor="turquoise",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # font_family="Courier New",
        font_color="white",
        # autosize=True,
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white"
    ),
    fig.update_yaxes(range = [scm-(5*scerr),scm+(5*scerr)])
    fig.update_yaxes(showgrid = False)
    fig.update_xaxes(showgrid = False)
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)') #,plot_bgcolor='rgba(0,0,0,0)',
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)
    return fig
def make_trendplot(df_reset,dfstats):
    # fig = make_subplots(rows=1, cols=2, shared_yaxes='rows',subplot_titles=("rotated_q vs. ellipticity","rotated_u vs. ellipticity"), horizontal_spacing=0.07)
    figq = px.scatter(
        df_reset,
        x='ellipticity',
        y ='rotated_q',
        color='ell_bins',
        # facet_row='ell_bins',
        # opacity = get_rls(df_reset),
        # trendline = 'lowess',
        # trendline_options = dict(frac=1),
        # height=1500,
    )
    # fig1.show()

    figu = px.scatter(
        df_reset,
        x='ellipticity',
        y ='rotated_u',
        color='ell_bins',
        # title=""
        # opacity = get_rls(df_reset),
        # trendline = 'lowess',
        # trendline_options = dict(frac=1),
        # height=50,
        # showlegend = False
    )

    fig_qyerr = px.scatter(dfstats, x="x_center", y="q_wa",error_y="weighted_err_q",color_discrete_sequence = ['black'])
    fig_uyerr = px.scatter(dfstats, x="x_center", y="u_wa",error_y="weighted_err_u",color_discrete_sequence = ['black'])

    figq.add_trace(fig_qyerr["data"][0], col = 1, row = 1,)

    for trace in range(len(figq["data"])):
        figq["data"][trace]['showlegend'] = False

    figu.add_trace(fig_uyerr["data"][0], col = 1, row = 1,)



    figq.update_yaxes(range=[-0.3, 0.3], dtick=0.05,col = 1,row = 1)
    # figq.update_yaxes(range=[-0.3, 0.3], dtick=0.05,col = 2,row = 1)
    figq.update_xaxes(range=[1,0.3], dtick=0.1,col = 1,row = 1)
    # figq.update_xaxes(range=[1,0.3], dtick=0.1,col = 2,row = 1)
    figq.add_hline(y = 0,col = 1, row = 1)
    # figq.add_hline(y = 0,col = 2, row = 1) #,plot_bgcolor='rgba(0,0,0,0)',
    figq.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # font_family="Courier New",
        font_color="white",
        # autosize=True,
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white",
        margin=dict(l=20, r=20, t=20, b=20),
    ),
    figq.update_yaxes(showgrid = False)
    figq.update_xaxes(showgrid = False)

    figu.update_yaxes(range=[-0.3, 0.3], dtick=0.05,col = 1,row = 1)
    # figu.update_yaxes(range=[-0.3, 0.3], dtick=0.05,col = 2,row = 1)
    figu.update_xaxes(range=[1,0.3], dtick=0.1,col = 1,row = 1)
    # figu.update_xaxes(range=[1,0.3], dtick=0.1,col = 2,row = 1)
    figu.add_hline(y = 0,col = 1, row = 1)
    # figu.add_hline(y = 0,col = 2, row = 1)
    figu.update_layout(paper_bgcolor='rgba(0,0,0,0)') #,plot_bgcolor='rgba(0,0,0,0)',
    figu.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # font_family="Courier New",
        font_color="white",
        # autosize=True,
        # title_font_family="Times New Roman",
        title_font_color="white",
        legend_title_font_color="white",
        margin=dict(l=20, r=20, t=20, b=20),
    ),
    figu.update_yaxes(showgrid = False)
    figu.update_xaxes(showgrid = False)
    return figq, figu

###################################### get init datasets #####################################################
# tm2 = time.time()
init_df_stats = calc_scstats_and_avgphil(init_t,init_sigexp)
# tm1 = time.time()
# print("stats took {} seconds".format(tm1-tm2))

# t0 = time.time()
init_df_rotated = rotate_df(init_df_stats,init_tkinput)
# t1 = time.time()
init_df_filtered, init_df_binned_stats = clean_and_bin(init_df_rotated,init_npix,init_errorthresh,init_bins)
# t2 = time.time()
# print("rotation took {} seconds".format(t1-t0))
# print("cleaning and binning took {} seconds".format(t2-t1))

init_chip,init_pos,init_rowindex = init_df_filtered.iloc[0].name
init_object_statedict = {
    'chip':init_chip,
    'pos':init_pos,
    'row_index':init_rowindex,
}
# init_df_statedict = {
#     't':init_t,
#     'sigexp':init_sigexp,
#     'npix':init_npix,
#     'errorthresh':init_errorthresh,
#     'bins':init_bins,
#     'tk':init_tkinput,
# }

init_rotated_col_names = init_df_rotated.columns
init_rotated_row_names = init_df_rotated.index
init_filtered_col_names = init_df_filtered.columns
init_filtered_row_names = init_df_filtered.index
# t5 = time.time()
init_qudict = get_qudf(init_chip, init_pos, init_rowindex, init_df_filtered)
# t6 = time.time()



# print("qudicts took {} seconds".format(t6-t5))

# # make init plots
init_fig_spread = make_stats_scatter(init_chip, init_pos, init_rowindex, init_df_filtered['across_offsets'])
init_figq_rotated, init_figu_rotated = make_trendplot(init_df_filtered['across_offsets'],init_df_binned_stats)
init_fig_q = make_individ_stats_fig('q', init_chip, init_pos, init_rowindex, init_sigexp, init_qudict, init_df_filtered['across_offsets'])
init_fig_u = make_individ_stats_fig('u', init_chip, init_pos, init_rowindex, init_sigexp, init_qudict, init_df_filtered['across_offsets'])
# dummy = dict(name='ambar', lastname = 'qadeer')
# print(json.dumps(dummy))

#################################################
###################################################
dash.register_page(__name__, path='/', name = 'Home')

def serve_layout():
    return dbc.Container(fluid = True,
                children = [
                    dbc.Row([
                        dbc.Col([

                            # title
                            dbc.Row([
                                dbc.Col(
                                    ["Object Statistics"],
                                    style = {'height':'10vh'},
                                    class_name = "d-flex w-100 text-uppercase fs-2 fw-bold align-items-center justify-content-start"
                                ),
                            ],class_name= "d-flex w-100 justify-content-center align-items-center"),

                            dbc.Row([
                                dbc.Col([

                                    # Filters
                                    dbc.Row([
                                        dbc.Col(
                                            ["Data Filters"],
                                            style = {'height':'auto'},
                                            class_name = "d-flex fs-5 fw-bold pt-1 pb-1 text-uppercase align-items-center justify-content-start"
                                        ),
                                    ],class_name= "d-flex w-100 justify-content-center align-items-center"),

                                    dbc.Row([
                                        ######################################
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.Row(["Select source:",html.Hr(className='mt-2')],style = {'height':'6vh',}, class_name = 'd-flex justify-content-start p-1 ps-2'),
                                                dbc.Row([
                                                    dbc.InputGroup(
                                                        [
                                                            
                                                            dbc.InputGroupText(["Apperture scale"], class_name='text-wrap text-start'),
                                                            dbc.Select(
                                                                id = 't_input',
                                                                options=[
                                                                    {'label':vals.replace('p','.'), 'value':vals} for vals in tlist
                                                                ],
                                                                value = init_t,
                                                            ),
                                                        ], class_name='m-0 ps-2 pe-2 pt-1 pb-1',
                                                    )
                                                ],class_name = 'd-flex justify-content-center'),
                                                dbc.Row([
                                                    dbc.InputGroup(
                                                        [
                                                            
                                                            dbc.InputGroupText(["Sigma"],),
                                                            dbc.Select(
                                                                id = 'sig_input',
                                                                options=[1,2,3],
                                                                value = init_sigexp
                                                            ),
                                                        ], class_name='m-0 ps-2 pe-2 pt-1 pb-1',
                                                    ),
                                                ],class_name = 'd-flex justify-content-center'),
                                            ],style = {'width':'100%',}, class_name = 'd-flex p-1 m-1'),
                                        ], class_name = 'd-flex h-100 justify-content-center align-items-center align-content-around m-0 p-0'),
                                        ###############################
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.Row(["Filter objects:",html.Hr(className='mt-2')],style = {'height':'6vh',}, class_name = 'd-flex justify-content-start p-1 ps-2'),
                                                dbc.Row([
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(["Min. size"],),
                                                            dbc.Input(id='npix_input', placeholder = 'minimum 20', type="number",value = init_npix),
                                                            dbc.InputGroupText("pixels"),
                                                        ], className="m-0 ps-2 pe-2 pt-1 pb-1",
                                                    ),
                                                ],class_name = 'd-flex justify-content-center'),
                                                dbc.Row([
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(["Max. stdev."], style = {'width':'110px',},),
                                                            dbc.Input(id='errorthresh_input', placeholder = 'minimum 20', type="number",required = True, value = init_errorthresh),
                                                        ], className="m-0 ps-2 pe-2 pt-1 pb-1",
                                                    ),
                                                ],class_name = 'd-flex justify-content-center'),
                                            ],style = {'width':'100%',}, class_name = 'd-flex p-1 m-1'),
                                        ], class_name = 'd-flex h-100 justify-content-center align-items-center align-content-around m-0 p-0'),
                                        ###########################
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.Row(["Setup analysis:",html.Hr(className='mt-2')],style = {'height':'6vh',}, class_name = 'd-flex justify-content-start p-1 ps-2'),
                                                dbc.Row([
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(["Bin size"],),
                                                            dbc.Input(id='bins_input', placeholder = 'minimum 20', type="number",required = True, value = init_bins),
                                                            # dbc.InputGroupText('$\Sigma$'),
                                                        ], className="m-0 ps-2 pe-2 pt-1 pb-1",
                                                    ),
                                                ],class_name = 'd-flex justify-content-center'),
                                                dbc.Row([
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(["Rotation Angle"], class_name='text-wrap text-start'),
                                                            dbc.Select(
                                                                    id = 'tk_input',
                                                                    options = ['theta','avg_phil'],
                                                                    value = init_tkinput,
                                                                ),
                                                            # dbc.InputGroupText('$\Sigma$'),
                                                        ], className="m-0 ps-2 pe-2 pt-1 pb-1",
                                                    ),
                                                ], class_name = 'd-flex justify-content-center'),
                                            ],style = {'width':'100%',}, class_name = 'd-flex p-1 m-1'),
                                        ], class_name = 'd-flex h-100 justify-content-center align-items-center align-content-around m-0 p-0'),
                                    ], class_name = "d-flex flex-column flex-lg-row w-100 p-0 m-0 align-items-center justify-content-between"),
                                    
                                    ######################################################
                                    dbc.Row([
                                        dbc.Col([dbc.Button('Refresh Graphs', color = 'secondary', id='refresh_button',n_clicks=0,)],style = {'width':'auto'},class_name = 'd-flex justify-content-start'),
                                    ], style = {'height':'auto', 'width':'auto'}, class_name= "d-flex w-100 justify-content-start p-0 m-0"),

                                ], style = {'height':'auto'},class_name = 'd-flex flex-wrap justify-content-center align-items-center p-0 m-0'),
                            ], style = {'height':'auto'},class_name = 'd-flex w-100 pt-2 m-0 pb-2 justify-content-center border border-3 rounded-2 border-secondary align-items-center'),
                            #######################################################
                            
                            dbc.Row([
                                dbc.Col([
                                        dbc.Row([
                                                dbc.Col(
                                                    ["Object spread in q-u space"],
                                                    style = {'height':'8vh'},
                                                    class_name = "d-flex fs-5 fw-bold text-uppercase align-items-center justify-content-start"
                                                ),
                                            ],class_name= "d-flex w-100 justify-content-center align-items-center"
                                        ),

                                        ##########################################
                                        dbc.Row([
                                                dbc.Col(
                                                    ["Object ID : "],
                                                    style = {'height':'8vh'},
                                                    width = 'auto',
                                                    class_name = "d-flex fs-5 fw-bold text-uppercase align-items-center justify-content-start"
                                                ),

                                                dbc.Col([
                                                        dbc.Select(
                                                            id = 'chip_input',
                                                            options=chiplist,
                                                            value = init_chip,                                                      
                                                        ),
                                                    ],
                                                    class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                ),

                                                dbc.Col([
                                                        dbc.Select(
                                                            id = 'pos_input',
                                                            options=poslist,
                                                            value = init_pos,                                                        
                                                        ),
                                                    ],
                                                    class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                ),

                                                dbc.Col([
                                                        dbc.Input(
                                                            id = 'row_index_input',
                                                            type = 'number',
                                                            value = init_rowindex,                                                        
                                                        ),
                                                    ],
                                                    class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                ),

                                                dbc.Col([
                                                        dbc.Input(
                                                            id = 'iloc_input',
                                                            type = 'number',
                                                            placeholder = 'current index : 0',                                                        
                                                        ),
                                                    ],
                                                    class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                ),

                                                dbc.Col([
                                                    dbc.Button('Submit', id='submit_button',n_clicks=0, class_name='w-100')],
                                                    class_name = 'd-flex justify-content-center m-0 p-1'
                                                ),

                                            ],
                                            class_name= "d-flex w-100 justify-content-center border-top border-bottom border-2 align-items-center "
                                        ),

                                        #####################################

                                        dbc.Row(
                                            [
                                            ########################
                                            dbc.Col(
                                                [
                                                    html.Div(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dcc.Loading(
                                                                        dcc.Graph(
                                                                            figure = init_fig_spread,
                                                                            id='spread',
                                                                            style = {'height':'100%', 'width':'100%'}
                                                                        ),
                                                                        color='rgba(109, 49, 128, 0.8)',
                                                                        parent_style = {'height':'100%', 'width':'100%',},
                                                                        type="default",
                                                                    ),
                                                                ],
                                                                class_name = 'd-flex h-100 w-100'
                                                            ),
                                                        ],
                                                        className = 'd-flex h-100 p-0 m-0 border border-1 border-primary text-uppercase',
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Card(
                                                                        [
                                                                            dcc.Loading(
                                                                                dcc.Graph(
                                                                                    figure = init_fig_u,
                                                                                    id='polarimetry_u',
                                                                                    style = {'height':'100%', 'width':'100%'}
                                                                                ),
                                                                                color='rgba(109, 49, 128, 0.8)',
                                                                                parent_style = {'height':'100%', 'width':'100%',},
                                                                                type="default",
                                                                            ),
                                                                        ],
                                                                        class_name = 'd-flex h-100 w-100'
                                                                    ),
                                                                ],
                                                                style = {'height':'49%'},
                                                                className = 'd-flex p-0 m-0 border border-1 border-success text-uppercase',
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dbc.Card(
                                                                        [
                                                                            dcc.Loading(
                                                                                dcc.Graph(
                                                                                    figure = init_fig_q,
                                                                                    id='polarimetry_q',
                                                                                    style = {'height':'100%', 'width':'100%'}
                                                                                ),
                                                                                color='rgba(109, 49, 128, 0.8)',
                                                                                parent_style = {'height':'100%', 'width':'100%',},
                                                                                type="default",
                                                                            ),
                                                                        ],
                                                                        class_name = 'd-flex h-100 w-100'
                                                                    ),
                                                                ],
                                                                style = {'height':'49%'},
                                                                className = 'd-flex p-0 m-0 border border-1 border-success text-uppercase',
                                                            ),
                                                        ],
                                                        style = {'height':'100%',},
                                                        className = 'd-flex flex-column p-0 m-0',
                                                    ),
                                                ],
                                                style = {'height':'70vh','width':'100%'},
                                                class_name = "d-flex flex-column flex-lg-row p-0 m-0 border border-1 border-secondary justify-content-center",
                                            ),
                                            ########################
                                            # dbc.Col(
                                            #     [
                                            #         html.Div(
                                            #             [
                                            #                 dbc.Card(
                                            #                     [
                                            #                         dcc.Loading(
                                            #                             dcc.Graph(
                                            #                                 figure = init_fig_u,
                                            #                                 id='polarimetry_u',
                                            #                                 style = {'height':'100%', 'width':'100%'}
                                            #                             ),
                                            #                             color='rgba(109, 49, 128, 0.8)',
                                            #                             parent_style = {'height':'100%', 'width':'100%',},
                                            #                             type="default",
                                            #                         ),
                                            #                     ],
                                            #                     class_name = 'd-flex h-100 w-100'
                                            #                 ),
                                            #             ],
                                            #             style = {'height':'49%'},
                                            #             className = 'd-flex p-0 m-0 border border-1 border-success text-uppercase',
                                            #         ),
                                            #         html.Div(
                                            #             [
                                            #                 dbc.Card(
                                            #                     [
                                            #                         dcc.Loading(
                                            #                             dcc.Graph(
                                            #                                 figure = init_fig_q,
                                            #                                 id='polarimetry_q',
                                            #                                 style = {'height':'100%', 'width':'100%'}
                                            #                             ),
                                            #                             color='rgba(109, 49, 128, 0.8)',
                                            #                             parent_style = {'height':'100%', 'width':'100%',},
                                            #                             type="default",
                                            #                         ),
                                            #                     ],
                                            #                     class_name = 'd-flex h-100 w-100'
                                            #                 ),
                                            #             ],
                                            #             style = {'height':'49%'},
                                            #             className = 'd-flex p-0 m-0 border border-1 border-success text-uppercase',
                                            #         ),
                                            #         # dbc.Row(
                                            #         #     dbc.Col(
                                            #                 # dbc.Card([
                                            #                         # dcc.Loading(
                                            #                         #     children = [dcc.Graph(
                                            #                         #         figure = init_fig_u,
                                            #                         #         id='polarimetry_u',
                                            #                         #         style = {'height':'100%', 'width':'100%'}
                                            #                         #         ),
                                            #                         #     ],
                                            #                         #     color='rgba(109, 49, 128, 0.8)',
                                            #                         #     parent_style = {'height':'100%', 'width':'100%',},
                                            #                         #     type="default",
                                            #                         # ),
                                            #                     # ],
                                            #                     # style = {'height':'100%', 'width':'100%'},
                                            #                 #     class_name = 'd-flex align-items-center m-0 p-0'
                                            #                 # ),                                                    
                                            #         #         class_name = 'd-flex m-1 border border-1 align-items-center justify-content-center'
                                            #         #     ),
                                            #         # class_name = 'd-flex border border-1 align-items-center m-0 p-0',
                                            #         # ),
                                            #         # dbc.Row(
                                            #         #     dbc.Col(
                                            #                 # dbc.Card([
                                            #                         # dcc.Loading(
                                            #                         #     children = [dcc.Graph(
                                            #                         #         figure = init_fig_q,
                                            #                         #         id='polarimetry_q',
                                            #                         #         style = {'height':'100%', 'width':'100%'}
                                            #                         #         ),
                                            #                         #     ],
                                            #                         #     color='rgba(109, 49, 128, 0.8)',
                                            #                         #     parent_style = {'height':'100%', 'width':'100%',},
                                            #                         #     type="default",
                                            #                         # ),
                                            #                     # ],
                                            #                     # style = {'height':'100%', 'width':'100%'},
                                            #                 #     class_name = 'd-flex align-items-center m-0 p-0'
                                            #                 # ),                                                    
                                            #         #         class_name = 'd-flex m-1 border border-1 align-items-center justify-content-center'
                                            #         #     ),
                                            #         # class_name = 'd-flex border border-1 align-items-center m-0 p-0',
                                            #         # ),



                                            #     ],
                                            #     style = {'height':'70vh','width':'50%'},
                                            #     class_name = "d-flex flex-column border border-1 border-info justify-content-between align-content-between p-0 m-0",
                                            # ),
                                            ########################
                                            ],
                                            style = {'height':'70vh',},
                                            class_name = 'd-flex justify-content-evenly align-content-evenly p-0 m-1',
                                        ),
                                    ],
                                    style = {'height':'auto'},
                                    class_name = 'd-flex flex-column justify-content-center align-items-center p-0 m-1'
                                ),
                                ],
                                style = {'height':'auto',},
                                class_name = 'd-flex w-100 pt-0 m-0 pb-0 mt-2 justify-content-center border border-3 border-primary rounded-2 align-items-between'
                            ),
                            ###################################

                            # rotated plots
                            dbc.Row([
                                dbc.Col(
                                    ["Polarimetry aligned with the object vs Ellipticity"],
                                    style = {'height':'auto'},
                                    class_name = "d-flex fs-5 fw-bold text-uppercase pt-2 pb-2 align-items-center justify-content-start"
                                ),
                            ],class_name= "d-flex w-100 justify-content-center align-items-center "),

                            dbc.Row([
                                ########################
                                dbc.Col([
                                    # dbc.Row([
                                        # dbc.Col([
                                            dbc.Card([
                                                dcc.Loading(
                                                    children = [dcc.Graph(figure = init_figq_rotated, id='rotated_qvsell',style = {'height':'30vh', 'width':'45vw'}),],
                                                    color="black",
                                                    type="default",
                                                ),
                                            ], class_name = 'd-flex-inline border-start border-info border-5 m-0 p-0'),
                                        # ],class_name = 'd-flex h-100 align-items-center justify-content-center m-0 p-0')
                                    # ],style = {'height':'45vh'},className = 'd-flex align-items-center justify-content-center border border-info m-0 p-0'),

                                    # dbc.Row([
                                        # dbc.Col([
                                            dbc.Card([
                                                dcc.Loading(
                                                    children = [dcc.Graph(figure = init_figu_rotated, id='rotated_uvsell',style = {'height':'30vh', 'width':'45vw'}),],
                                                    color="black",
                                                    type="default",
                                                ),
                                            ], class_name = 'd-flex-inline border-start border-primary border-5 m-0 p-0'),
                                        # ],class_name = 'd-flex h-100 align-items-center justify-content-center m-0 p-0')
                                    # ],style = {'height':'45vh'},className = 'd-flex align-items-center justify-content-center border border-danger m-0 p-0'),
                                
                                ], class_name = "d-flex h-100 w-100 flex-wrap justify-content-around align-items-center border border-primary p-0 m-0"),
                                ########################
                            ],style = {'height':'35vh',}, class_name = 'd-flex h-100 w-100 border border-success p-0 m-0'),

                        ],class_name = 'd-flex flex-wrap h-100 w-100 justify-content-center p-0 m-0'),
                    ], style={'position':'absolute','top':-60,'width':'95vw', 'padding':'0px', 'margin':'0px'}, className = "shadow-lg d-flex justify-content-center"),                
                ], style={'width':'100vw', 'padding':'0px', 'margin':'0px'}, className = "position-absolute d-flex justify-content-center",
            )

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

# app.layout = serve_layout

layout = serve_layout

@callback(
    # # update figures
    Output('polarimetry_q','figure'),
    Output('polarimetry_u','figure'),
    Output('spread','figure'),
    Output('rotated_qvsell','figure'),
    Output('rotated_uvsell','figure'),

    # # update stores
    Output('global_df_rotated','data'),
    Output('global_rotated_rows','data'),
    Output('global_rotated_cols','data'),
    Output('global_df_filtered','data'),
    Output('global_filtered_rows','data'),
    Output('global_filtered_cols','data'),
    Output('global_qudict','data'),
    Output('global_df_binned','data'),
    Output('global_df_statedict','data'),
    Output('global_object_statedict','data'),

    # update dropdowns/inputs
    Output('chip_input','value'),
    Output('pos_input','value'),
    Output('row_index_input','value'),
    Output('iloc_input','placeholder'),
    Output('iloc_input','value'),
    Output('spread','clickData'),

    #######################################
    
    # get click data
    Input('spread','clickData'),
    Input('refresh_button', 'n_clicks'),
    Input('submit_button', 'n_clicks'),

    #######################################

    # read df states
    State('t_input','value'),
    State('sig_input','value'),
    State('npix_input','value'),
    State('errorthresh_input','value'),
    State('bins_input','value'),
    State('tk_input','value'),

    # read object states
    State('chip_input','value'),
    State('pos_input','value'),
    State('row_index_input','value'),
    State('iloc_input','value'),

    # read stores
    State('global_df_rotated','data'),
    State('global_rotated_rows','data'),
    State('global_rotated_cols','data'),
    State('global_df_filtered','data'),
    State('global_filtered_rows','data'),
    State('global_filtered_cols','data'),
    State('global_qudict','data'),
    State('global_df_binned','data'),
    State('global_df_statedict','data'),
    State('global_object_statedict','data'),

    # callback setup
    # prevent_initial_call = 'True',
    # suppress_callback_exceptions=True,#data_clickcount, 
)   
def update_figs(spread_clickdata,refresh_clicks,submit_clicks,\
                t_input, sig_input, npix_input, errorthresh_input, bins_input, tk_input,\
                chip_input, pos_input, row_index_input, iloc_input,\
                global_df_rotated, global_rotated_rows, global_rotated_cols, global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_binned, global_df_statedict, global_object_statedict):

    
    print("triggered by element with id : {}".format(dash.callback_context.triggered_id))
    print("triggered: {}".format(format(dash.callback_context.triggered)))

    # firstrun: get df, expand, and plot
    if dash.callback_context.triggered_id is None:
        print("all initial setup is done in the local page file serverside")
        raise PreventUpdate
    
    # if global stores are empty
    if (refresh_clicks == 0) and any([kw is None for kw in [global_df_rotated, global_rotated_rows, global_rotated_cols, global_df_filtered, global_filtered_rows, global_filtered_cols,global_qudict, global_df_binned, global_df_statedict, global_object_statedict]]):
        print("populating datasets")
        # print("Nones : {}".format([kw is None for kw in [global_df_rotated, global_rotated_rows, global_rotated_cols, global_df_filtered, global_filtered_rows, global_filtered_cols,\
                                                        #  global_qudict, global_df_binned, global_df_statedict, global_object_statedict]]))
        # run calcs
        df_stats = calc_scstats_and_avgphil(t_input,sig_input)
        df_rotated = rotate_df(df_stats,tk_input)
        # get dfs for plots
        df_filtered, df_binned_stats = clean_and_bin(df_rotated,npix_input,errorthresh_input,bins_input)
        chip_plot,pos_plot,rowindex_plot = df_filtered.iloc[0].name
        iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))
        
        # spread_clickdata_plot = spread_clickdata
        qudict = get_qudf(chip_plot, pos_plot, rowindex_plot, df_filtered)

        # prep data for store
        global_rotated_rows = rotated_row_names = df_rotated.index
        global_rotated_cols = rotated_col_names = df_rotated.columns
        global_filtered_rows = filtered_row_names = df_filtered.index
        global_filtered_cols = filtered_col_names = df_filtered.columns
        
        # print("before (r,c): \n",rotated_row_names,rotated_col_names)
        global_object_statedict = object_statedict = {
            'chip':chip_plot,
            'pos':pos_plot,
            'row_index':rowindex_plot,
        }
        global_df_statedict = df_statedict = {
            't':t_input,
            'sigexp':sig_input,
            'npix':npix_input,
            'errorthresh':errorthresh_input,
            'bins':bins_input,
            'tk':tk_input,
        }
        global_df_rotated = dict_df_rotated_reset = jsonify(df_rotated)
        global_df_filtered = dict_df_filtered_reset = jsonify(df_filtered)
        global_qudict = dict_qudict = {qu:df.to_dict() for qu,df in qudict.items()}
        global_df_binned = dict_dfbinned = df_binned_stats.to_dict()

    # if refresh clicked
    if (dash.callback_context.triggered_id == 'refresh_button') and (refresh_clicks > 0):
        print('refresh clicked',refresh_clicks)

        # run calcs
        df_stats = calc_scstats_and_avgphil(t_input,sig_input)
        df_rotated = rotate_df(df_stats,tk_input)

        # get dfs for plots
        df_filtered, df_binned_stats = clean_and_bin(df_rotated,npix_input,errorthresh_input,bins_input)
        chip_plot,pos_plot,rowindex_plot = df_filtered.iloc[0].name
        # spread_clickdata_plot = spread_clickdata
        qudict = get_qudf(chip_plot, pos_plot, rowindex_plot, df_filtered)

        # prep data for store
        rotated_col_names = df_rotated.columns
        rotated_row_names = df_rotated.index
        filtered_row_names = df_filtered.index
        filtered_col_names = df_filtered.columns
        iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))
        object_statedict = {
            'chip':chip_plot,
            'pos':pos_plot,
            'row_index':rowindex_plot,
        }
        df_statedict = {
            't':t_input,
            'sigexp':sig_input,
            'npix':npix_input,
            'errorthresh':errorthresh_input,
            'bins':bins_input,
            'tk':tk_input,
        }
        dict_df_rotated_reset = jsonify(df_rotated)
        dict_df_filtered_reset = jsonify(df_filtered)
        dict_qudict = {qu:df.to_dict() for qu,df in qudict.items()}
        dict_dfbinned = df_binned_stats.to_dict()

    # if submit clicked
    elif (dash.callback_context.triggered_id == 'submit_button') and (submit_clicks > 0):
        print('submit clicked',submit_clicks)
        df_filtered, df_binned_stats = read_from_store_wrapper(global_df_filtered, rows = jsontomind(global_filtered_rows, rows = True), cols = jsontomind(global_filtered_cols), levels = 1),\
            read_from_store_wrapper(global_df_binned, levels = 1)
        # print(df_filtered)
        
        # check which object to display
        if iloc_input == None:
            # if iloc is none get data from row_index and setup iloc pklaceholder
            chip_plot,pos_plot,rowindex_plot = chip_input,pos_input,int(row_index_input)
            # print(df_filtered.loc[('CHIP2','POS2',133)])
            # print(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))
            iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))

        else:
            # if iloc is given use that to set row_index
            chip_plot,pos_plot,rowindex_plot = df_filtered.iloc[iloc_input].name
            iloc_placeholder = "current iloc : {}".format(iloc_input)

        # get object data for plot
        qudict = get_qudf(chip_plot, pos_plot, rowindex_plot, df_filtered)

        # prep data for store
        rotated_col_names = jsontomind(global_rotated_cols)
        rotated_row_names = jsontomind(global_rotated_rows, rows = True)
        filtered_row_names = df_filtered.index
        filtered_col_names = df_filtered.columns
        object_statedict = {
            'chip':chip_plot,
            'pos':pos_plot,
            'row_index':rowindex_plot,
        }
        df_statedict = {
            't':t_input,
            'sigexp':sig_input,
            'npix':npix_input,
            'errorthresh':errorthresh_input,
            'bins':bins_input,
            'tk':tk_input,
        }
        dict_df_rotated_reset = global_df_rotated
        dict_df_filtered_reset = global_df_filtered
        dict_qudict = {qu:df.to_dict() for qu,df in qudict.items()}
        dict_dfbinned = df_binned_stats.to_dict()

    # if spread clicked
    elif (dash.callback_context.triggered_id == 'spread') and (spread_clickdata is not None):
        print('clickdata triggered', spread_clickdata)
        filtered_row_names = jsontomind(global_filtered_rows, rows = True)
        filtered_col_names = jsontomind(global_filtered_cols)
        # print("after: \n",pd.MultiIndex.from_arrays(np.array(global_filtered_rows)),pd.MultiIndex.from_arrays(np.array(global_filtered_cols)))
        df_filtered, df_binned_stats = read_from_store_wrapper(global_df_filtered, rows = filtered_row_names, cols = filtered_col_names, levels = 1),\
            read_from_store_wrapper(global_df_binned, levels = 1)
        # print(df_filtered)
        chip_plot,pos_plot,rowindex_plot = spread_clickdata['points'][0]['customdata']
        iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))
        qudict = get_qudf(chip_plot, pos_plot, rowindex_plot, df_filtered)

        # prep data for store
        rotated_col_names = jsontomind(global_rotated_cols)
        rotated_row_names = jsontomind(global_rotated_rows, rows = True)
        object_statedict = {
            'chip':chip_plot,
            'pos':pos_plot,
            'row_index':rowindex_plot,
        }
        df_statedict = {
            't':t_input,
            'sigexp':sig_input,
            'npix':npix_input,
            'errorthresh':errorthresh_input,
            'bins':bins_input,
            'tk':tk_input,
        }
        dict_df_rotated_reset = global_df_rotated
        dict_df_filtered_reset = global_df_filtered
        dict_qudict = {qu:df.to_dict() for qu,df in qudict.items()}
        dict_dfbinned = df_binned_stats.to_dict()

    else:
        print("probably a recursive primary trigger because of external store inputs / unknown reason")
        raise PreventUpdate

    # chip_output, pos_output, row_index_output, spread_clickdata_output = chip_plot, pos_plot, rowindex_plot, spread_clickdata_plot

    # # make plots
    fig_spread = make_stats_scatter(chip_plot, pos_plot, rowindex_plot, df_filtered['across_offsets'])
    fig_rotatedq, fig_rotatedu = make_trendplot(df_filtered['across_offsets'],df_binned_stats)
    fig_q = make_individ_stats_fig('q', chip_plot, pos_plot, rowindex_plot, sig_input, qudict, df_filtered['across_offsets'])
    fig_u = make_individ_stats_fig('u', chip_plot, pos_plot, rowindex_plot, sig_input, qudict, df_filtered['across_offsets'])

    return (fig_q, fig_u, fig_spread, fig_rotatedq, fig_rotatedu,\
        dict_df_rotated_reset, mindtojson(rotated_row_names), mindtojson(rotated_col_names), dict_df_filtered_reset, mindtojson(filtered_row_names), mindtojson(filtered_col_names), dict_qudict, dict_dfbinned, df_statedict, object_statedict,\
        chip_plot, pos_plot, rowindex_plot, iloc_placeholder, None, spread_clickdata)


# if __name__ == "__main__":
#     app.run(debug=True, port = 8005)