import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import time

# Load data
test_df = pd.read_csv('test_with_anomalies.csv')
features = [col for col in test_df.columns if 'sensor' in col or 'efficiency' in col or 'vibration_rms' in col]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Predictive Maintenance Anomaly Detection Dashboard"),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dcc.Graph(id='sensor-plot'),
    dcc.Graph(id='anomaly-plot'),
    html.Div(id='alert', style={'color': 'red', 'fontSize': 24}),
    html.Div(id='diagnostics')
])

@app.callback(
    [Output('sensor-plot', 'figure'), Output('anomaly-plot', 'figure'), 
     Output('alert', 'children'), Output('diagnostics', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Simulate real-time: Cycle through data
    idx = n % len(test_df)
    current_data = test_df.iloc[:idx+1]
    
    # Sensor plot
    fig1 = go.Figure()
    for feat in ['sensor_1', 'sensor_2', 'efficiency']:
        fig1.add_trace(go.Scatter(x=current_data['cycle'], y=current_data[feat], mode='lines', name=feat))
    fig1.update_layout(title='Sensor Data Over Time')
    
    # Anomaly plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=current_data['cycle'], y=current_data['recon_error'], mode='lines', name='Reconstruction Error'))
    fig2.add_hline(y=test_df['recon_error'].mean() + 3 * test_df['recon_error'].std(), line_dash="dash", line_color="red")
    fig2.update_layout(title='Anomaly Score')
    
    # Alert
    latest_anomaly = current_data['ae_anomaly'].iloc[-1] if not current_data.empty else 0
    alert = "FAULT DETECTED!" if latest_anomaly == 1 else ""
    
    # Diagnostics
    if not current_data.empty:
        top_sensor = features[np.argmax(np.abs(current_data[features].iloc[-1] - current_data[features].mean()))]
        diagnostics = f"Top deviating feature: {top_sensor}"
    else:
        diagnostics = ""
    
    return fig1, fig2, alert, diagnostics

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
