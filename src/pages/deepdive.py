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
# from pages import home
warnings.filterwarnings("ignore", message = ".*FITS standard.*")
warnings.filterwarnings("ignore", message = ".*RADECSYS.*")
warnings.filterwarnings("ignore", message = ".*invalid value encountered in sqrt.*")
warnings.filterwarnings("ignore", message = ".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message = ".*scalar divide.*")
warnings.filterwarnings("ignore", message = ".*PerformanceWarning.*")
warnings.filterwarnings("ignore", message = ".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", message = ".*Input data contains invalid values.*")

# print(home.layout.chip_input)

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

image_path = 'assets/c1p2ob100cy1.png'

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

dash.register_page(__name__, name = "Source Images")

def serve_layout():
    return dbc.Container(
        fluid = True,
        children = 
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [   
                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(dbc.ModalTitle("Data missing")),
                                        dbc.ModalBody("The data for the graphs below is linked to the delection on the object statistics page, please \
                                                      navigate back to the object statistics page to populate the graphs below for specific objects."),
                                        dbc.ModalFooter(
                                            dbc.Button("Go to object statistics!", href="/")
                                        ),
                                    ],
                                    id="modal",
                                    is_open=False,
                                ),
                                # title
                                dbc.Row([
                                    dbc.Col(
                                        ["Object Deepdive"],
                                        style = {'height':'10vh'},
                                        class_name = "d-flex w-100 text-uppercase fs-2 fw-bold align-items-center justify-content-start"
                                    ),
                                ],class_name= "d-flex w-100 justify-content-center align-items-center"),
                                
                                dbc.Toast(
                                    [html.P("This page has limited online functionality due to size of the dataset and server instance constraints.\
                                             While we are working to bring more functionality to the page, The image below is a place holder for \
                                            features that will be added and is not connected to the graphs", className="mb-0")],
                                    id="simple-toast",
                                    header="Under Development",
                                    icon="primary",
                                    dismissable=True,
                                    is_open=True,
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                # object buttons
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            ["Object ID : "],
                                                            style = {'height':'8vh'},
                                                            width = 'auto',
                                                            class_name = "d-flex fs-5 fw-bold text-uppercase align-items-center justify-content-start"
                                                        ),

                                                        dbc.Col([
                                                                dbc.Select(
                                                                    id = 'chip_input_dd',
                                                                    options=chiplist,
                                                                    # value = init_chip,                                                      
                                                                ),
                                                            ],
                                                            class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                        ),

                                                        dbc.Col([
                                                                dbc.Select(
                                                                    id = 'pos_input_dd',
                                                                    options=poslist,
                                                                    # value = init_pos,                                                        
                                                                ),
                                                            ],
                                                            class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                        ),

                                                        dbc.Col([
                                                                dbc.Input(
                                                                    id = 'row_index_input_dd',
                                                                    type = 'number',
                                                                    # value = init_rowindex,                                                        
                                                                ),
                                                            ],
                                                            class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                        ),

                                                        dbc.Col([
                                                                dbc.Input(
                                                                    id = 'iloc_input_dd',
                                                                    type = 'number',
                                                                    placeholder = 'current index : 0',                                                        
                                                                ),
                                                            ],
                                                            class_name='d-flex m-0 ps-1 pe-1 pt-1 pb-1',
                                                        ),

                                                        dbc.Col([
                                                            dbc.Button('Submit', id='submit_button_dd',n_clicks=0, class_name='w-100')],
                                                            class_name = 'd-flex justify-content-center m-0 p-1'
                                                        ),
                                                    ],
                                                    class_name= "d-flex flex-column flex-lg-row w-100 justify-content-center border-top border-bottom border-2 align-items-center "
                                                ),

                                                # object plots
                                                dbc.Row(
                                                    [
                                                        ########################
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        dbc.Card(
                                                                            [
                                                                                html.H4("<u> by offset", className="card-title mt-2 ms-1"),
                                                                                dcc.Loading(
                                                                                    dcc.Graph(
                                                                                        id='polarimetry_u_dd',
                                                                                        style = {'height':'100%', 'width':'100%'}
                                                                                    ),
                                                                                    color='rgba(109, 49, 128, 0.8)',
                                                                                    parent_style = {'height':'100%', 'width':'100%',},
                                                                                    type="default",
                                                                                ),
                                                                            ],
                                                                            class_name = 'd-flex border border-2 border-info h-100 w-100'
                                                                        ),
                                                                    ],
                                                                    # style = {'height':'49%'},
                                                                    className = 'd-flex p-0 m-0 h-100 w-100',
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dbc.Card(
                                                                            [
                                                                                html.H4("<q> by offset", className="card-title mt-2 ms-1"),
                                                                                dcc.Loading(
                                                                                    dcc.Graph(
                                                                                        id='polarimetry_q_dd',
                                                                                        style = {'height':'100%', 'width':'100%'}
                                                                                    ),
                                                                                    color='rgba(109, 49, 128, 0.8)',
                                                                                    parent_style = {'height':'100%', 'width':'100%',},
                                                                                    type="default",
                                                                                ),
                                                                            ],
                                                                            class_name = 'd-flex h-100 w-100 border border-2 border-tertiary'
                                                                        ),
                                                                    ],
                                                                    # style = {'height':'49%'},
                                                                    className = 'd-flex p-0 m-0 h-100 w-100',
                                                                ),

                                                            ],
                                                            style = {'height':'auto','width':'100%', 'gap':'1%'},
                                                            class_name = "d-flex flex-column flex-lg-row p-0 mt-1 mb-1 justify-content-center",
                                                        ),
                                                    ],
                                                    style = {'height':'auto','width':'100%'},
                                                    class_name = 'd-flex justify-content-between align-content-between p-0 m-0',
                                                ),
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Div(
                                                            [
                                                                dbc.Button(
                                                                    "Open images",
                                                                    id="collapse-button",
                                                                    className="mb-3",
                                                                    color="primary",
                                                                    n_clicks=0,
                                                                ),
                                                                dbc.Collapse(
                                                                    html.Img(src=image_path, className = 'img-fluid'),
                                                                    id="collapse",
                                                                    is_open=False,
                                                                ),
                                                            ]
                                                        ),
                                                    ],class_name = 'd-flex w-100 h-100 justify-content-center'),
                                                ],class_name = 'w-100 h-100'),
                                            ],
                                            class_name = 'd-flex flex-wrap h-100 w-100 justify-content-center p-0 m-0'
                                        ),
                                    ],
                                    style = {'height':'auto'},
                                    class_name = 'd-flex justify-content-center align-items-center p-0 m-1'
                                ),
                            ],
                            style = {'height':'auto',},
                            class_name = 'd-flex w-100 pt-0 m-0 pb-0 mt-2 flex-column justify-content-center border border-3 border-secondary rounded-2 align-items-between'
                        ),
                    ],
                    style={'width':'95vw', 'padding':'0px', 'margin':'0px'},
                    className = "shadow-lg d-flex justify-content-center"
                ),                
            ], 
            style={'width':'100vw', 'padding':'0px', 'margin':'0px'},
            className = "position-absolute d-flex justify-content-center",
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
    # Output('modal','is_open'),

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
    Input('modal','is_open'),

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
    Input('global_object_statedict','data'),
    State('global_df_filtered','data'),
    State('global_filtered_rows','data'),
    State('global_filtered_cols','data'),
    State('global_qudict','data'),
    State('global_df_statedict','data'),
    ############################## SETUP ############################
    # callback setup
    prevent_initial_call = True,
    # suppress_callback_exceptions=True,
    allow_duplicate=True,
)
def deepdive(submit_clicks, modal_isopen, \
            global_object_statedict, global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict,):
    # conditionals to get all data needed for plots
    # print("triggered by element with id : {}".format(dash.callback_context.triggered_id))
    # print("triggered: {}".format(format(dash.callback_context.triggered)))

    # firstrun: get df, expand, and plot
    # if dash.callback_context.triggered_id is None or submit_clicks == 0:
    if modal_isopen:
        raise PreventUpdate
    else:
    # if new filtered df required: get basedf and apply filters
        if (dash.callback_context.triggered_id == 'submit_button_dd') and (submit_clicks > 0):
            print('submit clicked', submit_clicks)
            # print("initial setup")
            chip_plot,pos_plot,rowindex_plot = global_object_statedict['chip'],global_object_statedict['pos'],global_object_statedict['row_index']
            sig_plot, df_filtered, qudict = global_df_statedict['sigexp'], read_from_store_wrapper(global_df_filtered, rows = jsontomind(global_filtered_rows, rows = True), cols = jsontomind(global_filtered_cols), levels = 1), read_from_store_wrapper(global_qudict, levels = 2)
            iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))
        # else:
        #     print("probably a recursive primary trigger because of external store inputs / unknown reason, using this as primary call")
        #     if any([kw is None for kw in [global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict]]):
        #         print([kw is None for kw in [global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict]])
        #         # print('select object and get data from Home page')
        #         raise PreventUpdate
        else:
            chip_plot,pos_plot,rowindex_plot = global_object_statedict['chip'],global_object_statedict['pos'],global_object_statedict['row_index']
            sig_plot, df_filtered, qudict = global_df_statedict['sigexp'], read_from_store_wrapper(global_df_filtered, rows = jsontomind(global_filtered_rows, rows = True), cols = jsontomind(global_filtered_cols), levels = 1), read_from_store_wrapper(global_qudict, levels = 2)
            iloc_placeholder = "current iloc : {}".format(df_filtered.index.get_loc((chip_plot,pos_plot,int(rowindex_plot))))

        fig_q = make_individ_stats_fig('q', chip_plot, pos_plot, rowindex_plot, sig_plot, qudict, df_filtered['across_offsets'])
        fig_u = make_individ_stats_fig('u', chip_plot, pos_plot, rowindex_plot, sig_plot, qudict, df_filtered['across_offsets'])
        return fig_q, fig_u,\
            chip_plot, pos_plot, rowindex_plot, None, iloc_placeholder
        
@callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("modal", "is_open"),
    Input("modal", "is_open"),
    State('global_object_statedict','data'),
    State('global_df_filtered','data'),
    State('global_filtered_rows','data'),
    State('global_filtered_cols','data'),
    State('global_qudict','data'),
    State('global_df_statedict','data'),
    # allow_duplicate=True,
)
def toggle_modal(is_open, 
                    global_object_statedict, global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict,):
    if any([kw is None for kw in [global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict]]):
        # print([kw is None for kw in [global_df_filtered, global_filtered_rows, global_filtered_cols, global_qudict, global_df_statedict, global_object_statedict]])
        # print('select object and get data from Home page')
        # raise PreventUpdate
        # print(is_open)
        return not is_open

    # print(is_open)
    return is_open

# # Run the App
# if __name__ == '__main__':
#     app.run_server(debug = True, port = 8002)