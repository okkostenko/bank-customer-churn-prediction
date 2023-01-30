import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, ctx
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import tensorflow as tf
from interface.preprocessing import encode_categorical, recalculate_numerical, process_df
from interface.functions import predict_df, archive

app=dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model=tf.keras.models.load_model('./models/kostenko_model_1.h5')

# app div
app.layout=html.Div([
    # malual inputs
    html.Div(
        id='manual-input-div',
        children=[
            # personal info inputs
            html.Div(
                className='inputs',
                children=[
                    html.H5('Personal Info'),
                    html.Div([
                        dcc.Dropdown(
                            id='dropdown-geography',
                            options={0:'France', 1:'Germany', 2:'Spain'},
                            placeholder='Select a country'
                        ),
                        dcc.Dropdown(
                            id='dropdown-gender',
                            options={1:'Male', 0:'Feamle'},
                            placeholder='Select a gender'
                        ),
                        dcc.Input(
                            id='input-age',
                            type='number',
                            placeholder='Input the age',
                            min=18, max=120, step=1
                        )
                    ])
                ]
            ),  
            # finance info inputs
            html.Div(
                className='inputs',
                children=[
                    html.H5('Financial info'),
                    html.Div([
                        dcc.Input(
                            id='input-credit_score',
                            type='number',
                            placeholder='Input the Credit score',
                            min=350, max=850, step=1
                        ),
                        dcc.Input(
                            id='input-balance',
                            type='number',
                            placeholder='Input the Balance'
                        ),
                        dcc.Input(
                            id='input-salary',
                            type='number',
                            placeholder='Input the Salary'
                        )
                    ])
                ]
            ),  
            # Bank loyalty information
            html.Div(
                className='inputs',
                children=[
                    html.H5('Bank loyalty info'),
                    html.Div([
                        dcc.Input(
                            id='input-products',
                            type='number',
                            placeholder='Input the number of products',
                            min=1, max=4, step=1
                        ),
                        dcc.Input(
                            id='input-tenure',
                            type='number',
                            placeholder='Input the Tenure',
                            min=0, max=10, step=1
                        ),
                        dcc.Dropdown(
                            id='dropdown-has_card',
                            options={1:'Yes', 0:'No'},
                            placeholder='Has a card?'
                        ),
                        dcc.Dropdown(
                            id='dropdown-active_member',
                            options={1:'Yes', 0:'No'},
                            placeholder='Is active member?'
                        ),
                    ])
                ]
            ),
            html.Div(
                id='input-output',
                className='inputs prediction-non',
                children=[
                    html.Div(id='output-prediction', className='prediction-output-div'),
                    html.Button('Predict', id='predict-input', n_clicks=0)
                ]
            ),
        ] 
    ),
    html.H5('or', style={'width':'100%', 'textAlign': 'center'}),
    # upload file
    html.Div(
        id='upload-file-div',
        children=[# upload element
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            multiple=True
        ),
        html.Div(id='output-data-upload'),
        html.Div(id='output-prediction-upload'),
        html.Button('Predict', id='predict-file', n_clicks=0),
        dcc.Download(id='download-file')
        
    ])
])

def parse_contents(contents, filename, date):
    content_type, content_string=contents.split(',')
    decoded=base64.b64decode(content_string)

    try:
        if'csv' in filename:
            df=pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df=pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There wasan error preprocessing this file.'
        ])
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

    ])

def create_data_frame(contents, filename):
    content_type, content_string=contents.split(',')
    decoded=base64.b64decode(content_string)

    try:
        if'csv' in filename:
            df=pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df=pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There wasan error preprocessing this file.'
        ])
    return df

@app.callback(
    Output(component_id='output-prediction', component_property='children'),
    Input(component_id='predict-input', component_property='n_clicks'), 
    [State(component_id='dropdown-geography', component_property='value'),
    State(component_id='dropdown-gender', component_property='value'),
    State(component_id='input-age', component_property='value'),
    State(component_id='input-credit_score', component_property='value'),
    State(component_id='input-balance', component_property='value'),
    State(component_id='input-salary', component_property='value'),
    State(component_id='input-products', component_property='value'),
    State(component_id='input-tenure', component_property='value'),
    State(component_id='dropdown-has_card', component_property='value'),
    State(component_id='dropdown-active_member', component_property='value'),]
)

def update_output_input(n_clicks, geography, gender, age, credit_score, balance, salary, products, tenure, has_card, active_member):
    if 'predict-input'==ctx.triggered_id:
        numerical_features={'CreditScore': credit_score, 'Age': age, 'Tenure': tenure, 'Balance': balance, 'NumOfProducts': products, 'EstimatedSalary': salary}
        numerical_df=[]
        for feature in numerical_features:
            numerical_features[feature]=recalculate_numerical(float(numerical_features[feature]), feature)
            numerical_df.append(numerical_features[feature])
        numerical_df=numerical_df[:-1]+[int(has_card), int(active_member)]+[numerical_df[-1]]

        df=pd.DataFrame(np.array(numerical_df+ encode_categorical(int(geography), 'geography')+encode_categorical(int(gender), 'gender')).reshape(1, 13))

        prediction_result=model.predict(df)
        print(prediction_result)
        if prediction_result<0.5:
            msg='Not going to churn'
        elif prediction_result>=0.5:
            msg='Most likely to churn'
        else:
            msg='-----'
        
        return msg

@app.callback(
    Output(component_id='output-data-upload', component_property='children'),
    Input(component_id='upload-data', component_property='contents'),   
    State(component_id='upload-data', component_property='filename'),
    State(component_id='upload-data', component_property='last_modified')
)

def update_output_upload(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(
    [Output(component_id='output-prediction-upload', component_property='children'),
    Output(component_id='download-file', component_property='data')],
    [Input(component_id='predict-file', component_property='n_clicks'),
    Input(component_id='upload-data', component_property='contents')],
    State(component_id='upload-data', component_property='filename')
)

def update_prediction_upload(n_clicks, list_of_contents, list_of_names):
    if 'predict-file'==ctx.triggered_id:
        # predicted=[]
        if list_of_contents is not None:
            dframes=[process_df(create_data_frame(c, n)) for c, n in zip(list_of_contents, list_of_names)]
        predict_df(dframes)
        archive()
        msg='Prediction is ready!'
        return [msg, dcc.send_file('./files/prediction.zip')]
    



if __name__=='__main__':
    app.run_server(debug=False)

