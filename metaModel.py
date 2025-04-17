import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

def load_data():
    """
    Load all necessary data files for the evacuation visualization tool
    
    Returns:
        df: DataFrame containing road closure scenarios
        road_importances: Dictionary mapping road IDs to their importance scores
        scenario_rules: List of discovered scenario patterns
        road_network: DataFrame containing road network information
    """
    # Load scenario data
    try:
        df = pd.read_csv('evacuation_scenarios_sample.csv')
        print(f"Loaded {len(df)} scenario records with {len(df.columns)-1} road features")
    except FileNotFoundError:
        print("Warning: Evacuation scenarios file not found. Creating sample data...")
        # Create a small sample dataset if file doesn't exist
        num_roads = 50  # Using a small subset for demonstration
        columns = [f'road_{i}' for i in range(num_roads)]
        columns.append('evacuation_time')
        
        df = pd.DataFrame(columns=columns)
        for i in range(100):  # 100 sample scenarios
            # Generate random road closures (1 = closed, 0 = open)
            road_closures = np.zeros(num_roads)
            num_closed = np.random.randint(2, 10)
            closed_indices = np.random.choice(num_roads, num_closed, replace=False)
            road_closures[closed_indices] = 1
            
            # Generate evacuation time (simple model: base time + effect of closures)
            base_time = 60
            # More effect for lower-numbered roads (assumed to be more important)
            importance_weights = np.exp(-np.arange(num_roads)/10)
            evacuation_time = base_time + np.sum(road_closures * importance_weights * 20)
            evacuation_time += np.random.normal(0, 5)  # Add noise
            
            row = list(road_closures) + [evacuation_time]
            df.loc[i] = row
    
    # Load feature importance data
    try:
        importance_df = pd.read_csv('road_importance.csv')
        road_importances = dict(zip(importance_df['road_id'], importance_df['importance']))
        print(f"Loaded importance scores for {len(road_importances)} roads")
    except FileNotFoundError:
        print("Warning: Road importance file not found. Generating from scenario data...")
        # If feature importance file doesn't exist, calculate it from the scenario data
        # using a simple Random Forest model
        from sklearn.ensemble import RandomForestRegressor
        
        X = df.drop('evacuation_time', axis=1)
        y = df['evacuation_time']
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        road_importances = {f'road_{i}': imp for i, imp in enumerate(importances)}
    
    # Load scenario discovery results
    try:
        with open('scenario_discovery_results.json', 'r') as f:
            scenario_rules = json.load(f)
        print(f"Loaded {len(scenario_rules)} scenario patterns")
    except FileNotFoundError:
        print("Warning: Scenario discovery file not found. Creating sample scenarios...")
        # Create some sample scenario rules if file doesn't exist
        scenario_rules = [
            {
                'scenario_id': 1,
                'description': 'Major arterial routes blocked',
                'conditions': [
                    {'road_id': 'road_0', 'status': 'closed'},
                    {'road_id': 'road_1', 'status': 'closed'}
                ],
                'avg_evacuation_time': 95.0,
                'probability_high_impact': 0.85,
                'coverage': 0.25
            },
            {
                'scenario_id': 2,
                'description': 'Northern exit routes blocked',
                'conditions': [
                    {'road_id': 'road_5', 'status': 'closed'},
                    {'road_id': 'road_10', 'status': 'closed'}
                ],
                'avg_evacuation_time': 88.5,
                'probability_high_impact': 0.75,
                'coverage': 0.20
            }
        ]
    
    # Load road network information
    try:
        road_network = pd.read_csv('road_network.csv')
        print(f"Loaded network data for {len(road_network)} roads")
    except FileNotFoundError:
        print("Warning: Road network file not found. Creating sample network...")
        # Create a sample road network if file doesn't exist
        road_network = pd.DataFrame(columns=[
            'road_id', 'start_x', 'start_y', 'end_x', 'end_y', 
            'capacity', 'road_type', 'zone'
        ])
        
        # Get the road IDs from the scenario data
        road_ids = [col for col in df.columns if col.startswith('road_')]
        
        # Create a grid-like network
        grid_size = int(np.sqrt(len(road_ids)))
        for i, road_id in enumerate(road_ids):
            # Determine road position in grid
            row = i // grid_size
            col = i % grid_size
            
            # Determine if horizontal or vertical road
            is_horizontal = (i % 2 == 0)
            
            if is_horizontal:
                start_x = col * 10
                start_y = row * 10
                end_x = (col + 1) * 10
                end_y = row * 10
            else:
                start_x = col * 10
                start_y = row * 10
                end_x = col * 10
                end_y = (row + 1) * 10
            
            # Assign road type based on position (central roads more important)
            center = grid_size // 2
            dist_from_center = max(abs(row - center), abs(col - center))
            if dist_from_center <= grid_size // 5:
                road_type = 'arterial'
                capacity = np.random.randint(1500, 2000)
            elif dist_from_center <= grid_size // 3:
                road_type = 'collector'
                capacity = np.random.randint(1000, 1500)
            else:
                road_type = 'local'
                capacity = np.random.randint(500, 1000)
            
            # Assign zone based on position
            if row < grid_size // 2 and col < grid_size // 2:
                zone = 'northwest'
            elif row < grid_size // 2 and col >= grid_size // 2:
                zone = 'northeast'
            elif row >= grid_size // 2 and col < grid_size // 2:
                zone = 'southwest'
            else:
                zone = 'southeast'
            
            # Add to dataframe
            road_network.loc[i] = [
                road_id, start_x, start_y, end_x, end_y,
                capacity, road_type, zone
            ]
    
    return df, road_importances, scenario_rules, road_network

# Build meta-model
def build_meta_model(df):
    # Separate features and target
    X = df.drop('evacuation_time', axis=1)
    y = df['evacuation_time']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Create dashboard
def create_dashboard(model, road_importances, scenario_rules, road_network):
    app = dash.Dash(__name__)
    
    # Get top important roads
    sorted_roads = sorted(road_importances.items(), key=lambda x: x[1], reverse=True)
    top_roads = [road for road, _ in sorted_roads[:50]]
    
    app.layout = html.Div([
        html.H1("Evacuation Time Predictor"),
        
        # Road selection panel
        html.Div([
            html.H3("Select Roads to Close:"),
            dcc.Dropdown(
                id='road-selector',
                options=[{'label': f'{road} (Importance: {road_importances[road]:.4f})', 
                          'value': road} for road, _ in sorted_roads[:100]],
                multi=True,
                value=[]
            ),
        ], style={'width': '50%', 'padding': '10px'}),
        
        # Controls and outputs
        html.Div([
            html.Button('Predict Evacuation Time', id='predict-button', 
                      style={'background-color': '#4CAF50', 'color': 'white', 'padding': '10px'}),
            html.Div(id='prediction-output', style={'margin': '20px', 'font-size': '20px'}),
        ]),
        
        # Tabs for different visualizations
        dcc.Tabs([
            dcc.Tab(label='Road Importance', children=[
                dcc.Graph(id='importance-graph')
            ]),
            dcc.Tab(label='Road Network', children=[
                dcc.Graph(id='network-graph')
            ]),
            dcc.Tab(label='Scenario Insights', children=[
                html.Div(id='scenario-discovery-insights')
            ]),
        ]),
    ])
    
    @app.callback(
        [Output('prediction-output', 'children'),
        Output('importance-graph', 'figure'),
        Output('network-graph', 'figure'),
        Output('scenario-discovery-insights', 'children')],
        [Input('predict-button', 'n_clicks')],
        [State('road-selector', 'value')]
    )
    def update_output(n_clicks, selected_roads):
        if n_clicks is None or not selected_roads:
            # Default empty state
            return (
                "No roads selected", 
                px.bar(title="Select roads to see their importance"),
                px.scatter(title="Road Network"),
                html.Div("No scenarios selected")
            )
        
        # Create input for model prediction
        # Get the number of features from the model
        n_features = len(df.columns) - 1  # All columns except evacuation_time
        
        # Initialize input array with zeros
        X_input = np.zeros(n_features)
        
        # Set selected roads to 1 (closed)
        for road in selected_roads:
            # Extract the road number from the road ID (e.g., 'road_10' -> 10)
            try:
                road_idx = int(road.split('_')[1])
                # Only set to 1 if the index is within bounds
                if road_idx < n_features:
                    X_input[road_idx] = 1
            except (IndexError, ValueError):
                print(f"Warning: Could not process road ID {road}")
        
        # Predict evacuation time
        pred_time = model.predict([X_input])[0]
        
        # Rest of the function remains the same...
        
        # Create importance graph
        selected_importances = {road: road_importances.get(road, 0) for road in selected_roads}
        imp_df = pd.DataFrame({
            'road': list(selected_importances.keys()),
            'importance': list(selected_importances.values())
        }).sort_values('importance', ascending=False)
        
        imp_fig = px.bar(
            imp_df,
            x='road', 
            y='importance',
            title='Importance of Selected Roads for Evacuation Time'
        )
        
        # Create a simplified network visualization
        # First, check if we have valid coordinate data
        has_valid_coords = (
            'start_x' in road_network.columns and 
            'start_y' in road_network.columns and
            not road_network['start_x'].isna().all() and
            not road_network['start_y'].isna().all()
        )

        if has_valid_coords:
            # Create a basic scatter plot (not a map) for the road network
            net_fig = px.scatter(
            road_network,
            x='start_x',
            y='start_y',
            color='road_type',
            size='capacity',
            hover_data=['road_id', 'zone'],
            title='Road Network with Selected Closures Highlighted in Red',
            opacity=0.7
            )

            # Add lines to connect start and end points if they exist
            if 'end_x' in road_network.columns and 'end_y' in road_network.columns:
                # First add all open road lines
                for idx, row in road_network.iterrows():
                    if row['road_id'] not in selected_roads:
                        net_fig.add_trace(
                            go.Scatter(
                                x=[row['start_x'], row['end_x']],
                                y=[row['start_y'], row['end_y']],
                                mode='lines',
                                line=dict(color='gray', width=1),
                                showlegend=False
                            )
                        )
                # Then add closed road lines AFTER (so they appear on top)
                for idx, row in road_network[road_network['road_id'].isin(selected_roads)].iterrows():
                    net_fig.add_trace(
                        go.Scatter(
                            x=[row['start_x'], row['end_x']],
                            y=[row['start_y'], row['end_y']],
                            mode='lines',
                            line=dict(color='red', width=3),
                            name='Closed Roads',
                            showlegend=idx == 0  # Only show in legend once
                        )
                    )
            
                # Add closed road markers LAST (to ensure they're on top)
                selected_road_info = road_network[road_network['road_id'].isin(selected_roads)]
                if not selected_road_info.empty:
                    net_fig.add_trace(
                        go.Scatter(
                            x=selected_road_info['start_x'],
                            y=selected_road_info['start_y'],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='x',
                                line=dict(width=2, color='black')
                            ),
                            name='Closed Road Points',
                            text=selected_road_info['road_id'],
                            hoverinfo='text'
                        )
                    )
            
            # Update layout to improve visibility
            net_fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=False),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            # Fallback visualization if coordinate data is missing
            road_status = []
            for road_id in road_network['road_id']:
                status = 'Closed' if road_id in selected_roads else 'Open'
                road_type = road_network.loc[road_network['road_id']==road_id, 'road_type'].iloc[0]
                road_status.append({'road_id': road_id, 'status': status, 'road_type': road_type})
            
            status_df = pd.DataFrame(road_status)
            
            # Create a simple bar chart showing closed vs open roads
            net_fig = px.bar(
                status_df,
                x='road_type',
                color='status',
                title='Number of Roads by Type and Status',
                barmode='group',
                color_discrete_map={'Closed': 'red', 'Open': 'green'}
            )        
        # Create scenario insights
        matching_scenarios = []
        
        for scenario in scenario_rules:
            # Check if selected roads match any scenario
            conditions = scenario['conditions']
            road_ids = [cond['road_id'] for cond in conditions 
                       if cond['status'] == 'closed']
            
            # Simple matching - if all closed roads in the scenario are selected
            if all(road in selected_roads for road in road_ids):
                matching_scenarios.append(scenario)
        
        if matching_scenarios:
            scenario_elements = [html.H3("Matching Scenario Patterns:")]
            
            for scenario in matching_scenarios:
                scenario_card = html.Div([
                    html.H4(f"Scenario {scenario['scenario_id']}: {scenario['description']}"),
                    html.P(f"Average Evacuation Time: {scenario['avg_evacuation_time']:.1f} minutes"),
                    html.P(f"Probability of High Impact: {scenario['probability_high_impact']:.2f}"),
                    html.Ul([
                        html.Li(f"{cond['road_id']} - {cond['status']}")
                        for cond in scenario['conditions']
                    ])
                ], style={'background': '#ffffcc', 'padding': '15px', 
                          'border': '1px solid #e6e600', 'margin': '10px'})
                
                scenario_elements.append(scenario_card)
                
            scenario_output = html.Div(scenario_elements)
        else:
            scenario_output = html.Div([
                html.H3("No Known Scenario Patterns Matched"),
                html.P("The current combination of road closures doesn't match any known high-impact patterns.")
            ])
        
        return (
            f"Predicted Evacuation Time: {pred_time:.1f} minutes",
            imp_fig,
            net_fig,
            scenario_output
        )
    
    return app

# Main execution
if __name__ == '__main__':
    print("Loading data...")
    df, road_importances, scenario_rules, road_network = load_data()
    
    print("Building model...")
    model = build_meta_model(df)
    
    print("Creating dashboard...")
    app = create_dashboard(model, road_importances, scenario_rules, road_network)
    
    print("Running dashboard...")
    app.run(debug=True)