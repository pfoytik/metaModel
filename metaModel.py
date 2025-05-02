import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import joblib

def is_in_box(row, box_min, box_max):
    for feat, min_val in box_min.items():
        if row.get(int(feat), 0) < min_val:
            return False
    for feat, max_val in box_max.items():
        if row.get(int(feat), 0) > max_val:
            return False
    return True

def generate_box_features(X: pd.DataFrame, box_data: dict) -> pd.DataFrame:
    box_features = pd.DataFrame(index=X.index)

    for box_id, box_limits in box_data.items():
        box_min = box_limits.get('min', {})
        box_max = box_limits.get('max', {})

        # For each row, determine if it satisfies the box conditions
        box_features[f'box_{box_id}'] = X.apply(lambda row: is_in_box(row, box_min, box_max), axis=1).astype(int)

    return box_features

### evacScenarioFile, roadImportanceFile, scenarioDiscoveryFile, roadNetworkFile
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
        #df = pd.read_csv('evacuation_scenarios_sample.csv')
        df = pd.read_csv('ladris_df.csv')
        print(f"Loaded {len(df)} scenario records with {len(df.columns)-1} road features")
    except FileNotFoundError:
        print("Warning: Evacuation scenarios file not found. Creating sample data...")
        # Create a small sample dataset if file doesn't exist
        num_roads = 50  # Using a small subset for demonstration
        columns = [f'road_{i}' for i in range(num_roads)]
        columns.append('evacuation_time')
        
        df = pd.DataFrame(columns=columns)
        for i in range(df.columns):  # 100 sample scenarios
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
        #importance_df = pd.read_csv('road_importance.csv')
        importance_df = pd.read_csv('ladris_feature_importance.csv')
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

    ### Load ladris_conv2_scenarioDiscovery.json
    try:
        #with open('scenario_discovery_results.json', 'r') as f:
        with open('ladris_conv2_scenarioDiscovery.json', 'r') as f:
            scenario_rules = json.load(f)
        print(f"Loaded {len(scenario_rules)} scenario patterns")
    except FileNotFoundError:
        print("Warning: Scenario discovery file not found. Creating sample scenarios...")

    X_augmented = pd.concat([df, generate_box_features(df, scenario_rules)], axis=1)
    
    # Load scenario discovery results
    try:
        #with open('scenario_discovery_results.json', 'r') as f:
        with open('ladris_scenario_discovery.json', 'r') as f:
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
        #road_network = pd.read_csv('road_network.csv')
        road_network = pd.read_csv('ladris_latlong.csv')
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
    
    return X_augmented, road_importances, scenario_rules, road_network

# Custom model as fallback
class CustomModel:
    def __init__(self, road_importances):
        self.road_importances = road_importances
    
    def predict(self, X):
        X_array = X[0]  # Get the input array
        selected_indices = np.where(X_array > 0)[0]
        selected_roads = [idx for idx in selected_indices]
        
        # Base time + penalty for each closed road
        base_time = 3.0
        road_penalty = sum(self.road_importances.get(idx, 0.01) * 50 
                          for idx in selected_indices)
        
        return np.array([base_time + road_penalty])

# Build meta-model
def build_meta_model(df):
    # Separate features and target
    X = df.drop('evacuation_time', axis=1)
    y = df['evacuation_time']
    
    # Check if there's enough variation in the target
    if y.std() < 0.1:
        print("WARNING: Not enough variation in evacuation times. Adding synthetic data...")
        # Create some synthetic data points
        for i in range(20):
            # Generate random road closures
            row = np.zeros(len(X.columns))
            num_closed = np.random.randint(2, 10)
            closed_indices = np.random.choice(len(X.columns), num_closed, replace=False)
            row[closed_indices] = 1
            
            # Generate reasonable evacuation time (10-60 minutes)
            evacuation_time = 10 + np.random.rand() * 50
            
            # Add to training data
            X = pd.concat([X, pd.DataFrame([row], columns=X.columns)])
            y = pd.concat([y, pd.Series([evacuation_time])])

    # Train model
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42)
    model.fit(X, y)
    
    # Verify model works by testing different inputs
    test_input = np.zeros(X.shape[1])
    baseline = model.predict([test_input])[0]
    
    test_input[0] = 1  # Close first road
    pred1 = model.predict([test_input])[0]
    
    test_input[0] = 0
    test_input[10] = 1  # Close a different road
    pred2 = model.predict([test_input])[0]
    
    print(f"Model test - Baseline: {baseline}, Pred1: {pred1}, Pred2: {pred2}")
    if abs(baseline - pred1) < 0.01 and abs(baseline - pred2) < 0.01:
        print("WARNING: Model not responding to different inputs. Returning a custom model...")
        return CustomModel(road_importances)

    return model

# Create dashboard
def create_dashboard(model, road_importances, scenario_rules, road_network):
    app = dash.Dash(__name__)
    
    # Get top important roads
    sorted_roads = sorted(road_importances.items(), key=lambda x: x[1], reverse=True)
    top_roads = [road for road, _ in sorted_roads]
    
    app.layout = html.Div([
        html.H1("Evacuation Time Predictor"),
        
        # Road selection panel
        html.Div([
            html.H3("Select Roads to Close:"),
            dcc.Dropdown(
                id='road-selector',
                options=[{'label': f'{road} (Importance: {road_importances[road]:.4f})', 
                          'value': road} for road, _ in sorted_roads],
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
    
    ### make a prediction with model
    def make_prediction(model, input_data):
        """
        Make a prediction using the trained model.
        
        Parameters:
            model: Trained model
            input_data: DataFrame containing the input data for prediction
        
        Returns:
            prediction: Predicted evacuation time
        """
        # Ensure input data is in the correct format
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        
        print(input_data)
        # Make prediction
        prediction = model.predict(input_data)
        
        return prediction

    @app.callback(
        [Output('prediction-output', 'children'),
        Output('importance-graph', 'figure'),
        Output('network-graph', 'figure'),
        Output('scenario-discovery-insights', 'children')],
        [Input('predict-button', 'n_clicks')],
        [State('road-selector', 'value')]
    )
    def update_output(n_clicks, selected_roads):
        print(selected_roads)
#        if n_clicks is None or not selected_roads:
#            # Default empty state
#            return (
#                "No roads selected", 
#                px.bar(title="Select roads to see their importance"),
#                px.scatter(title="Road Network"),
#                html.Div("No scenarios selected")
#            )
        
        # Create input for model prediction
        # Get the number of features from the model
        n_features = len(df.columns) - 1  # All columns except evacuation_time
        
        # Initialize input array with zeros
        X_input = np.zeros(n_features)
        print("X_input", X_input)
        # Set selected roads to 1 (closed)
        for road in selected_roads:
            # Extract the road number from the road ID (e.g., 'road_10' -> 10)
            print(road)
            try:
                road_idx = road
                ### get the column index of road_idx from df.columns
                print(df.columns)
                print(road_idx)
                road_col_idx = df.columns.get_loc(str(road))
                X_input[road_col_idx] = 1  # Close this road
                print(X_input)
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
            
            ### convert road_ids to int
            road_ids = [int(road_id) for road_id in road_ids]
            # append scenario if any of selected roads is in road_ids
            if any(road in selected_roads for road in road_ids):
                matching_scenarios.append(scenario)
            # Simple matching - if all closed roads in the scenario are selected
            #if all(road in selected_roads for road in road_ids):
            #    matching_scenarios.append(scenario)
        
        if matching_scenarios:
            scenario_elements = [html.H3("Matching Scenario Patterns:")]
            
            for scenario in matching_scenarios:
                scenario_card = html.Div([
                    html.H4(f"Scenario {scenario['scenario_id']}: {scenario['description']}"),
                    html.P(f"Average Evacuation Time: {scenario['avg_evacuation_time']:.1f} hours"),
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
            f"Predicted Evacuation Time: {pred_time:.1f} hours",
            imp_fig,
            net_fig,
            scenario_output
        )
    
    return app

# Main execution
if __name__ == '__main__':
    print("Loading data...")
    ### evacScenarioFile, roadImportanceFile, scenarioDiscoveryFile, roadNetworkFile
    df, road_importances, scenario_rules, road_network = load_data()#"ladris_df.csv", )
    
    ### check if evacuation_model.pkl exists
    try:
        model = joblib.load('evacuation_model.pkl')
        print("Model loaded from evacuation_model.pkl")
    except FileNotFoundError:
        print("No existing model found. Building a new one...")
        # Build a new model
        print("Building model...")
        model = build_meta_model(df)    
        ### export model
        # Save the model to a file    
        joblib.dump(model, 'evacuation_model.pkl')
        print("Model saved as evacuation_model.pkl")

    print("Creating dashboard...")
    app = create_dashboard(model, road_importances, scenario_rules, road_network)
    
    print("Running dashboard...")
    app.run(debug=True)