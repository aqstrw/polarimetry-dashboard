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

app = dash.Dash(__name__, use_pages = True,) # external_stylesheets=[dbc.themes.QUARTZ]

server = app.server

app.layout = html.Div(
    [
        
        dcc.Store(id = 'global_df_rotated',storage_type='memory'),
        dcc.Store(id = 'global_rotated_rows',storage_type='memory'),
        dcc.Store(id = 'global_rotated_cols',storage_type='memory'),
        dcc.Store(id = 'global_df_filtered',storage_type='memory'),
        dcc.Store(id = 'global_filtered_rows',storage_type='memory'),
        dcc.Store(id = 'global_filtered_cols',storage_type='memory'),
        dcc.Store(id = 'global_qudict',storage_type='memory'),
        dcc.Store(id = 'global_df_binned',storage_type='memory'),
        dcc.Store(id = 'global_df_statedict',storage_type='memory'),
        dcc.Store(id = 'global_object_statedict',storage_type='memory'),

        # main app framework
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink(page['name'], href=page['path'])) for page in dash.page_registry.values()
            ],
            brand="AanalitiQ",
            brand_href="https://www.linkedin.com/in/ambar-qadeer",
            color= 'rgba(130, 83, 18, 1)',
            # width = 'auto',
            dark=True,
            fluid = True,
            sticky = 'top',
            class_name = 'd-flex w-100'
        ),

        html.Div(
            [
                html.H1("Integrated Optical Polarimetry of Distant Galaxies",
                    style = {'height':'15vh','width':'100vw',},
                    className = 'd-flex w-100 h-100 mt-3 mb-2 text-center align-items-center justify-content-center'
                ),
                html.P("This dashboard showcases the results of polarimetric analysis of distant objects in a wide field survey set-up. \
                       The purpose of the dashboard is to allow convenient deep-dive and evaluation of each specific case based on it's \
                       deviation from sensible values along orthogonal linear polarisation planes (Q and U). This dashboard was developed to be \
                       used offline and has lost some functionality in making it available online due to size constraints and the \
                       intention is to showcase how this tool was used to make it easy to assess each objects case by case",
                    style = {'height':'auto','width':'60vw',},
                    className = 'd-flex  align-items-center justify-content-center'
                ),
            ],
            style = {'height':'auto','width':'100vw',},
            className="d-flex flex-column text-light align-items-center justify-content-center",
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
