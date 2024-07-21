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

chiplist = ["CHIP1","CHIP2"]
poslist = ["POS1","POS2"]
offsetlist = ["Offset0","Offset1","Offset2","Offset3","Offset4","Offset5","Offset6","Offset7"]
tlist = ['2','2p25','2p5','2p75','3']

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
        height = 310,
        # labels=dict(y="<"+str(qu)+">", color="Ellipticity bins"),
    )
    # fig1.show()
    fig2 = px.scatter(qudf,
        x=offsetlist,
        y = qu+'2',
        error_y = 'err' + qu + '2',
        color_discrete_sequence=['red'],
        height = 310,
        # custom_data=[qu,c,p,ri],
        # labels=dict(x="Offsets", y="<"+str(qu)+">", color="Ellipticity bins"),
    )
    if qu == 'u':
        fig.update_layout(xaxis_title="Offsets", yaxis_title = "<"+str(qu)+">")
    else:
        fig.update_layout(xaxis_title=None, yaxis_title = "<"+str(qu)+">")
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
            line_width = 1,
            # xref='x',
            # yref='y',
        )
    fig.add_hline(
            # x0=0,
            y=scm,
            # x1=7,
            # y1 = stats[0],
            line=dict(color='darkgreen',dash = 'solid',),
            # color='Black',
            # line_dash = 'dashed',
            line_width = 0.5,
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
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    fig.update_yaxes(range = [scm-(5*scerr),scm+(5*scerr)])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)') #,plot_bgcolor='rgba(0,0,0,0)',
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)
    return fig

dash.register_page(__name__, name = "Source Images")

def serve_layout(data = 'test'):
    return dbc.Container(fluid = True,
        children = [
            dbc.Container(children = [
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink(page['name'], href=page['path'])) for page in dash.page_registry.values()
                    ],
                    brand="AanalitiQ",
                    brand_href="https://www.linkedin.com/in/ambar-qadeer",
                    color="primary",
                    dark=True,
                ),

                # title
                dbc.Row([
                    dbc.Col([
                        html.H1("Individual Object Statistics : {}".format(data),
                            style = {'textAlign': 'center'}),
                    ], width = 12, class_name = "bg-light border p-2 border-2"),
                    
                ],class_name= "m-2 border-0"),
                
                # object plots
                dbc.Row([
                        dbc.Row([
                            dcc.Graph(id='polarimetry_q_dd',)
                        ],className = "bg-light border-start border-5 border-info d-md-inline-block p-1 m-1"), #style = {'display': 'inline-block'} 'height': '25vh'
                        dbc.Row([
                            dcc.Graph(id='polarimetry_u_dd',)
                        ],className = "bg-light border-start border-5 border-danger d-md-inline-block p-1 m-1"), #style = {'display': 'inline-block'} 'height': '25vh'
                ],class_name = "gx-2 p-1 m-1 justify-content-evenly"),

                # object buttons
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='chip_input_dd', options = chiplist, placeholder = "enter object chip",),
                    ], width = 3),
                    dbc.Col([
                        dcc.Dropdown(id='pos_input_dd', options = poslist, placeholder = "enter object pos",),
                    ], width = 3),
                    dbc.Col([
                        dcc.Input(id='row_index_input_dd', placeholder = "enter object row index", debounce = True, required = True),
                    ], width = 3),
                    dbc.Col([
                        dcc.Input(id='iloc_input_dd', type = 'number', placeholder = "current iloc : 0", value = None, debounce = True, required = True),
                    ], width = 3),
                    dbc.Col([
                        html.Button('Submit', id='submit_button_dd', n_clicks=0),
                    ], width = 3),
                ]),
        ],style = {'position':'absolute','top':-70,'width':'90vw', 'padding':'0px', 'margin':'0px','textAlign': 'center'},
        className = "bg-dark bg-opacity-25 shadow-lg justify-content-center"),
    ], style={'width':'100vw', 'padding':'0px', 'margin':'0px'}, className = "position-absolute bg-dark bg-opacity d-flex justify-content-center",
)

layout = serve_layout

@callback(
        
    ######################### OUTPUTS ########################
    # update figures
    Output('polarimetry_q_dd','figure'),
    Output('polarimetry_u_dd','figure'),

    # # read df states
    # Output('t_input','value'),
    # Output('sig_input','value'),
    # Output('npix_input','value'),
    # Output('errorthresh_input','value'),
    # Output('bins_input','value'),
    # Output('tk_input','value'),

    # update dropdowns/inputs
    Output('chip_input_dd','value'),
    Output('pos_input_dd','value'),
    Output('row_index_input_dd','value'),
    Output('iloc_input_dd','value'),
    Output('iloc_input_dd','placeholder'),

        # # update stores
    # Output('global_df_filtered','data', allow_duplicate=True),
    # Output('global_filtered_rows','data', allow_duplicate=True),
    # Output('global_filtered_cols','data', allow_duplicate=True),
    # Output('global_qudict','data', allow_duplicate=True),
    # Output('global_df_statedict','data', allow_duplicate=True),
    # Output('global_object_statedict','data', allow_duplicate=True),

    ######################### INPUTS ########################
    # get click data
    Input('submit_button_dd', 'n_clicks'),

    # # read df states
    # State('t_input','value'),
    # State('sig_input','value'),
    # State('npix_input','value'),
    # State('errorthresh_input','value'),
    # State('bins_input','value'),
    # State('tk_input','value'),

    # read object states
    # State('chip_input_dd','value'),
    # State('pos_input_dd','value'),
    # State('row_index_input_dd','value'),
    # State('iloc_input_dd','value'),

    # read stores
    # State('global_df_rotated','data'),
    # State('global_rotated_rows','data'),
    # State('global_rotated_cols','data'),
    State('global_df_filtered','data'),
    State('global_filtered_rows','data'),
    State('global_filtered_cols','data'),
    State('global_qudict','data'),
    State('global_df_statedict','data'),
    State('global_object_statedict','data'),
    ############################## SETUP ############################
    # callback setup
    prevent_initial_call = True,
    # suppress_callback_exceptions=True,
)
def deepdive(submit_clicks,\
            global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict,):
    # conditionals to get all data needed for plots
    print("triggered by element with id : {}".format(dash.callback_context.triggered_id))
    print("triggered: {}".format(format(dash.callback_context.triggered)))

    # firstrun: get df, expand, and plot
    # if dash.callback_context.triggered_id is None or submit_clicks == 0:

    # if new filtered df required: get basedf and apply filters
    if (dash.callback_context.triggered_id == 'submit_button_dd') and (submit_clicks > 0):
        print('submit clicked', submit_clicks)
        # print("initial setup")
        chip_plot,pos_plot,rowindex_plot = global_object_statedict['chip'],global_object_statedict['pos'],global_object_statedict['row_index']
        sig_plot, df_filtered, qudict = global_df_statedict['sigexp'], read_from_store_wrapper(global_df_filtered, rows = jsontomind(global_filtered_rows, rows = True), cols = jsontomind(global_filtered_cols), levels = 1), read_from_store_wrapper(global_qudict, levels = 2)
        iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))
    else:
        print("probably a recursive primary trigger because of external store inputs / unknown reason, using this as primary call")
        if any([kw is None for kw in [global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict]]):
            print([kw is None for kw in [global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict]])
            print('select object and get data from Home page')
            raise PreventUpdate

    fig_q = make_individ_stats_fig('q', chip_plot, pos_plot, rowindex_plot, sig_plot, qudict, df_filtered['across_offsets'])
    fig_u = make_individ_stats_fig('u', chip_plot, pos_plot, rowindex_plot, sig_plot, qudict, df_filtered['across_offsets'])
    return fig_q, fig_u,\
        chip_plot, pos_plot, rowindex_plot, iloc_placeholder, None
        


# # Run the App
# if __name__ == '__main__':
#     app.run_server(debug = True, port = 8002)