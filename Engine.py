import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn import metrics
from sklearn.cluster import KMeans
from umap import UMAP


st.title('**Gas Lift Rate Prediction**')
st.markdown('''
This program is able to predict **Max Oil Rate & Injection Pressure** from Gas Lift well based on several input parameters.
* First, you need to either input your own dataset or use sample dataset
* Then, you need to select the model configuration
* Last, you need to input the value for each parameters
''')

st.sidebar.title('**Input Data Section**')
input = st.sidebar.selectbox('Choose Dataset: ',['Your Dataset','Sample Dataset'])

if input == 'Your Dataset':
    upload = st.sidebar.file_uploader("Drop your dataset (.csv): ")
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        df = pd.read_csv('Data.csv')

else:
    df = pd.read_csv('Data.csv')

st.sidebar.write("*Note: sample data is generated from two gas lift well actual data and simulated using prosper")

st.sidebar.subheader('Model Input Section')
model_type = st.sidebar.selectbox('Regression Model Method', ['Linear Regression','Gradient Boosting Regressor','Ridge Regressor'])
test_ratio = float(st.sidebar.slider('Test Data Ratio (%) ', min_value=0, max_value=100,value=40))

#Clustering
features = df.loc[:, :'Max Depth Injection']

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(features)

kmeans = KMeans(n_clusters= 2)
label = kmeans.fit_predict(proj_2d)
df.insert(19,"label", label, False)
filtered_label0 = proj_2d[label == 0]
filtered_label1 = proj_2d[label == 1]

chart = px.scatter(x=filtered_label0[:,0], y=filtered_label0[:,1], width=600, height=450)
chart.add_trace(go.Scatter(x=filtered_label0[:,0],y=filtered_label0[:,1],
                    mode='markers',
                    name='Cluster 0'))
chart.add_trace(go.Scatter(x=filtered_label1[:,0],y=filtered_label1[:,1],
                    mode='markers',
                    name='Cluster 1'))
chart.update_layout(title={'text':'UMAP with Kmeans clustering plot',
                            'xanchor' : 'left',
                            'yanchor' :'top',
                            'x' : 0})
st.plotly_chart(chart)

First_Data = df[df["label"] == 0]
Second_Data = df[df["label"] == 1]

input = df[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
        'GOR (scf/stb)', 'Temperature Res. (F)', 'ID tubing (in)',
        'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
        'Max CHP (Psig)', 'Gas Gravity', 'API', 'Depth of Reservoir', 'Max Depth Injection']]
input_predictions = []
st.sidebar.subheader('Prediction Input')
for i in input.columns:
    x = st.sidebar.number_input("Input {}".format(i), format="%f", min_value=0.)
    input_predictions.append(x)
title = {'Pressure (psi)':[input_predictions[0]], 'PI (psi/bbl)':[input_predictions[1]], 'Initial Watercut (%)':[input_predictions[2]],
        'GOR (scf/stb)':[input_predictions[3]], 'Temperature Res. (F)':[input_predictions[4]], 'ID tubing (in)':[input_predictions[5]],
        'FTP(Flowing Tubing Pressure) (psig)':[input_predictions[6]], 'Killing Fluid (psi/)':[input_predictions[7]],
        'Max CHP (Psig)':[input_predictions[8]], 'Gas Gravity':[input_predictions[9]], 'API':[input_predictions[10]], 
        'Depth of Reservoir':[input_predictions[11]], 'Max Depth Injection':[input_predictions[12]]}
inp = pd.DataFrame(title)
        
#FUNCTION
def GradBoost(x_data, Data, Factor_Y, input_predictions):

    scaler = StandardScaler()
    x_data_scaled = x_data.copy()
    x_data_scaled = pd.DataFrame(x_data_scaled, index=x_data.index, columns=x_data.columns)
   
    X = x_data_scaled
    Y = Data[Factor_Y]
    
    X_train, X_test, y_train ,y_test = train_test_split(X, Y, test_size=test_ratio/100, random_state=123)
    reg = GradientBoostingRegressor()
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    z = pd.DataFrame(columns = ['MAE','MSE','RMSE','R2'], 
                data=[[metrics.mean_absolute_error(y_test, predictions),
                metrics.mean_squared_error(y_test, predictions),
                np.sqrt(metrics.mean_squared_error(y_test, predictions)),
                metrics.r2_score(y_test,predictions)]])
    
    output = float(reg.predict([input_predictions]))

    st.info('**Predicted Output** : {} '.format(output))

    st.info('''**Model Evaluation Metrics**

    Mean Absolute Error (MAE): {}

    Mean Squared Error (MSE): {}

    Root Mean Squared Error (RMSE): {}

    R Squared (R2): {}
    '''.format(metrics.mean_absolute_error(y_test, predictions)
                ,metrics.mean_squared_error(y_test, predictions)
                ,np.sqrt(metrics.mean_squared_error(y_test, predictions))
                ,metrics.r2_score(y_test,predictions)))
                
    chart2 = px.scatter(x=y_test,y=predictions,width=600,height=450)
    chart2.add_trace(go.Scatter(x=y_test,y=y_test,
                        mode='lines',
                        name='Best Fit Line'))

    chart2.update_layout(title={'text':'Regression Plot',
                                'xanchor' : 'left',
                                'yanchor' :'top',
                                'x' : 0},
                       xaxis_title='Test Data',
                       yaxis_title='Predicted Data')
    st.plotly_chart(chart2)

def RidgeReg(x_data, Data, Factor_Y, input_predictions):

    scaler = StandardScaler()
    x_data_scaled = x_data.copy()
    x_data_scaled = pd.DataFrame(x_data_scaled, index=x_data.index, columns=x_data.columns)
   
    X = x_data_scaled
    Y = Data[Factor_Y]
    
    X_train, X_test, y_train ,y_test = train_test_split(X, Y, test_size=test_ratio/100, random_state=123)
    reg = Ridge(alpha = 1)
    reg.fit(X_train, y_train)
    
    predictions = reg.predict(X_test)
    
    z = pd.DataFrame(columns = ['MAE','MSE','RMSE','R2'], 
                data=[[metrics.mean_absolute_error(y_test, predictions),
                metrics.mean_squared_error(y_test, predictions),
                np.sqrt(metrics.mean_squared_error(y_test, predictions)),
                metrics.r2_score(y_test,predictions)]])

    output = float(reg.predict([input_predictions]))

    st.info('**Predicted Output** : {} '.format(output))

    st.info('''**Model Evaluation Metrics**

    Mean Absolute Error (MAE): {}

    Mean Squared Error (MSE): {}

    Root Mean Squared Error (RMSE): {}

    R Squared (R2): {}
    '''.format(metrics.mean_absolute_error(y_test, predictions)
                ,metrics.mean_squared_error(y_test, predictions)
                ,np.sqrt(metrics.mean_squared_error(y_test, predictions))
                ,metrics.r2_score(y_test,predictions)))
                
    chart2 = px.scatter(x=y_test,y=predictions,width=600,height=450)
    chart2.add_trace(go.Scatter(x=y_test,y=y_test,
                        mode='lines',
                        name='Best Fit Line'))

    chart2.update_layout(title={'text':'Regression Plot',
                                'xanchor' : 'left',
                                'yanchor' :'top',
                                'x' : 0},
                       xaxis_title='Test Data',
                       yaxis_title='Predicted Data')
    st.plotly_chart(chart2)

def LinearReg(Data, Factor_Y, input_predictions):
    x_data = Data[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
           'GOR (scf/stb)', 'Temperature Res. (F)', 'ID tubing (in)',
           'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
           'Max CHP (Psig)', 'Gas Gravity', 'API', 'Depth of Reservoir', 'Max Depth Injection']]
    x_data_scaled = x_data.copy()
    x_data_scaled = pd.DataFrame(x_data_scaled, index=x_data.index, columns=x_data.columns)
    
    X = x_data_scaled
    Y = Data[Factor_Y]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio/100, random_state=123)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    
    predictions = reg.predict(X_test)
    
    z = pd.DataFrame(columns = ['MAE','MSE','RMSE','R2'], 
                data=[[metrics.mean_absolute_error(y_test, predictions),
                metrics.mean_squared_error(y_test, predictions),
                np.sqrt(metrics.mean_squared_error(y_test, predictions)),
                metrics.r2_score(y_test,predictions)]])

    output = float(reg.predict([input_predictions]))

    st.info('**Predicted Output** : {} '.format(output))

    st.info('''**Model Evaluation Metrics**

    Mean Absolute Error (MAE): {}

    Mean Squared Error (MSE): {}

    Root Mean Squared Error (RMSE): {}

    R Squared (R2): {}
    '''.format(metrics.mean_absolute_error(y_test, predictions)
                ,metrics.mean_squared_error(y_test, predictions)
                ,np.sqrt(metrics.mean_squared_error(y_test, predictions))
                ,metrics.r2_score(y_test,predictions)))
                
    chart2 = px.scatter(x=y_test,y=predictions,width=600,height=450)
    chart2.add_trace(go.Scatter(x=y_test,y=y_test,
                        mode='lines',
                        name='Best Fit Line'))

    chart2.update_layout(title={'text':'Regression Plot',
                                'xanchor' : 'left',
                                'yanchor' :'top',
                                'x' : 0},
                       xaxis_title='Test Data',
                       yaxis_title='Predicted Data')
    st.plotly_chart(chart2)

#Regressor Method
if model_type == 'Linear Regression':
    st.title('**First Cluster**')
    st.markdown('''
    **Max Oil Rate**
    ''')   
    LinearReg(First_Data,'Max Oil Rate (STB/d)',input_predictions)
    st.markdown('''
    **Injection Pressure (psig)**
    ''')   
    LinearReg(First_Data,'Injection Pressure (psig)',input_predictions)
    st.title('**Second Cluster**')
    st.markdown('''
    **Max Oil Rate**
    ''')   
    LinearReg(Second_Data,'Max Oil Rate (STB/d)',input_predictions)
    st.markdown('''
    **Injection Pressure (psig)**
    ''')   
    LinearReg(Second_Data,'Injection Pressure (psig)',input_predictions)

elif model_type == 'Gradient Boosting Regressor':
    st.title('**First Cluster**')
    st.markdown('''
    **Max Oil Rate**
    ''')   
    x_data = First_Data[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
           'GOR (scf/stb)', 'Temperature Res. (F)', 'ID tubing (in)',
           'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
           'Max CHP (Psig)', 'Gas Gravity', 'API', 'Depth of Reservoir', 'Max Depth Injection']]
    inputs = inp[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
           'GOR (scf/stb)', 'Temperature Res. (F)', 'ID tubing (in)',
           'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
           'Max CHP (Psig)', 'Gas Gravity', 'API', 'Depth of Reservoir', 'Max Depth Injection']].values[0]
    GradBoost(x_data, First_Data, 'Max Oil Rate (STB/d)',inputs)
    st.markdown('''
    **Injection Pressure (psig)**
    ''')   
    x_data = First_Data[['Killing Fluid (psi/)', 'Max CHP (Psig)', 'API', 'Max Depth Injection']]
    inputs = inp[['Killing Fluid (psi/)', 'Max CHP (Psig)', 'API', 'Max Depth Injection']].values[0]
    GradBoost(x_data, First_Data, 'Injection Pressure (psig)',inputs)
    st.title('**Second Cluster**')
    st.markdown('''
    **Max Oil Rate**
    ''')  
    x_data = Second_Data[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
            'GOR (scf/stb)', 'Gas Gravity']]
    inputs = inp[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
            'GOR (scf/stb)', 'Gas Gravity']].values[0]
    GradBoost(x_data, Second_Data, 'Max Oil Rate (STB/d)',inputs)
    st.markdown('''
    **Injection Pressure (psig)**
    ''')   
    x_data = Second_Data[['Pressure (psi)',  'PI (psi/bbl)',  'Temperature Res. (F)',
            'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
            'Max CHP (Psig)', 'Gas Gravity','Depth of Reservoir']]
    inputs = inp[['Pressure (psi)',  'PI (psi/bbl)',  'Temperature Res. (F)',
            'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
            'Max CHP (Psig)', 'Gas Gravity','Depth of Reservoir']].values[0]
    GradBoost(x_data, Second_Data, 'Injection Pressure (psig)',inputs)
    
else:
    st.title('**First Cluster**')
    st.markdown('''
    **Max Oil Rate**
    ''')   
    x_data = First_Data[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
           'GOR (scf/stb)', 'Max CHP (Psig)', 'Gas Gravity']]
    inputs = inp[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
           'GOR (scf/stb)', 'Max CHP (Psig)', 'Gas Gravity']].values[0]
    RidgeReg(x_data, First_Data, 'Max Oil Rate (STB/d)',inputs)
    st.markdown('''
    **Injection Pressure (psig)**
    ''')   
    x_data = First_Data[['Killing Fluid (psi/)', 'Max CHP (Psig)', 'API', 'Max Depth Injection']]
    inputs = inp[['Killing Fluid (psi/)', 'Max CHP (Psig)', 'API', 'Max Depth Injection']].values[0]
    RidgeReg(x_data, First_Data, 'Injection Pressure (psig)',inputs)
    st.title('**Second Cluster**')
    st.markdown('''
    **Max Oil Rate**
    ''')  
    x_data = Second_Data[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
       'GOR (scf/stb)', 'Gas Gravity']]
    inputs = inp[['Pressure (psi)', 'PI (psi/bbl)', 'Initial Watercut (%)',
       'GOR (scf/stb)', 'Gas Gravity']].values[0]
    RidgeReg(x_data, Second_Data, 'Max Oil Rate (STB/d)',inputs)
    st.markdown('''
    **Injection Pressure (psig)**
    ''')   
    x_data = Second_Data[['Pressure (psi)',  'PI (psi/bbl)',  'Temperature Res. (F)',
       'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
       'Max CHP (Psig)', 'Gas Gravity','Depth of Reservoir']]
    inputs = inp[['Pressure (psi)',  'PI (psi/bbl)',  'Temperature Res. (F)',
       'FTP(Flowing Tubing Pressure) (psig)', 'Killing Fluid (psi/)',
       'Max CHP (Psig)', 'Gas Gravity','Depth of Reservoir']].values[0]
    RidgeReg(x_data, Second_Data, 'Injection Pressure (psig)',inputs)

check_data = st.checkbox('Display Input Dataset')

if check_data:
    st.write(df)
