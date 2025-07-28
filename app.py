import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import joblib
import plotly.graph_objects as go
import random

# Page configuration
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Team data with colors
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

team_colors = {
    'Sunrisers Hyderabad': '#FF8C00',
    'Mumbai Indians': '#004BA0',
    'Royal Challengers Bangalore': '#CC0000',
    'Kolkata Knight Riders': '#3A225D',
    'Kings XI Punjab': '#DD1F2D',
    'Chennai Super Kings': '#FDB813',
    'Rajasthan Royals': '#E91E63',
    'Delhi Capitals': '#17479E'
}

# Load the model
@st.cache_resource
def load_model():
    try:
        with open("pipe.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'pipe.pkl' not found.")
        return None
    except AttributeError as e:
        st.error(f"Failed to load model. Likely version mismatch.\n\nError: {e}")
        return None

pipe = load_model()

if pipe is not None:
    st.success("Model loaded successfully!")
else:
    st.stop()

# Header
st.markdown('<h1 class="main-header">🏏 IPL Win Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Cricket Analytics Dashboard</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🏏 Navigation")
page = st.sidebar.selectbox("Choose a section", ["Match Predictor", "Live Simulation", "Team Analytics", "Historical Data"])

if page == "Match Predictor":
    st.header("🎯 Match Prediction")
    
    # Team selection with enhanced UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏏 Batting Team")
        batting_team = st.selectbox('Select the batting team', sorted(teams), key='batting')
        if batting_team:
            st.markdown(f'<div class="team-card" style="background: linear-gradient(135deg, {team_colors.get(batting_team, "#667eea")} 0%, #764ba2 100%);">{batting_team}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("🥎 Bowling Team")
        bowling_team = st.selectbox('Select the bowling team', sorted(teams), key='bowling')
        if bowling_team:
            st.markdown(f'<div class="team-card" style="background: linear-gradient(135deg, {team_colors.get(bowling_team, "#667eea")} 0%, #764ba2 100%);">{bowling_team}</div>', unsafe_allow_html=True)
    
    # Validation
    if batting_team == bowling_team:
        st.error("⚠️ Batting and bowling teams cannot be the same!")
    
    # City selection
    st.subheader("🏟️ Match Venue")
    selected_city = st.selectbox('Select host city', sorted(cities))
    
    # Match details
    st.subheader("📊 Match Details")
    
    col3, col4 = st.columns(2)
    with col3:
        target = st.number_input('🎯 Target Score', min_value=1, max_value=300, value=180, step=1)
    with col4:
        st.metric("Target Set", f"{target} runs", "First Innings Complete")
    
    # Current match situation
    st.subheader("⚡ Current Match Situation")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        score = st.number_input('📈 Current Score', min_value=0, max_value=target, value=120, step=1)
    with col6:
        overs = st.number_input('⏱️ Overs Completed', min_value=0.1, max_value=20.0, value=15.0, step=0.1)
    with col7:
        wickets = st.number_input('❌ Wickets Lost', min_value=0, max_value=10, value=3, step=1)
    
    # Real-time calculations
    if overs > 0:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_remaining = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        # Display current metrics
        st.subheader("📊 Live Match Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Runs Needed", runs_left, f"{runs_left - target//2} from par")
        with metric_col2:
            st.metric("Balls Left", int(balls_left), f"{int(balls_left//6)} overs")
        with metric_col3:
            st.metric("Current RR", f"{crr:.2f}", f"{crr - 8:.2f} vs avg")
        with metric_col4:
            st.metric("Required RR", f"{rrr:.2f}", f"{rrr - crr:.2f} vs current")
        
        # Run rate comparison chart
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = rrr,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Required Run Rate"},
            delta = {'reference': crr},
            gauge = {
                'axis': {'range': [None, 15]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 6], 'color': "lightgray"},
                    {'range': [6, 9], 'color': "yellow"},
                    {'range': [9, 12], 'color': "orange"},
                    {'range': [12, 15], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': crr}}))
        
        fig_rr.update_layout(height=300)
        st.plotly_chart(fig_rr, use_container_width=True)
    
    # Prediction button
    if st.button('🔮 Predict Win Probability', key='predict_btn'):
        if pipe is None:
            st.error("Model not loaded. Cannot make predictions.")
        elif batting_team == bowling_team:
            st.error("Please select different teams for batting and bowling.")
        elif overs <= 0:
            st.error("Overs completed must be greater than 0.")
        else:
            with st.spinner('🤖 AI is analyzing the match...'):
                time.sleep(2)  # Simulate processing time
                
                # Prepare input data
                input_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': [selected_city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets': [wickets_remaining],
                    'total_runs_x': [target],
                    'crr': [crr],
                    'rrr': [rrr]
                })
                
                # Make prediction
                result = pipe.predict_proba(input_df)
                loss_prob = result[0][0]
                win_prob = result[0][1]
                
                # Display results with enhanced visualization
                st.success("🎉 Prediction Complete!")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, {team_colors.get(batting_team, '#4facfe')} 0%, #00f2fe 100%);">
                        <h2>{batting_team}</h2>
                        <h1>{round(win_prob*100)}%</h1>
                        <p>Win Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_result2:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, {team_colors.get(bowling_team, '#f093fb')} 0%, #f5576c 100%);">
                        <h2>{bowling_team}</h2>
                        <h1>{round(loss_prob*100)}%</h1>
                        <p>Win Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability visualization
                fig_prob = go.Figure(data=[
                    go.Bar(name=batting_team, x=[batting_team], y=[win_prob*100], 
                           marker_color=team_colors.get(batting_team, '#4facfe')),
                    go.Bar(name=bowling_team, x=[bowling_team], y=[loss_prob*100], 
                           marker_color=team_colors.get(bowling_team, '#f093fb'))
                ])
                
                fig_prob.update_layout(
                    title="Win Probability Comparison",
                    yaxis_title="Probability (%)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Match situation analysis
                st.subheader("🔍 Match Analysis")
                
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                
                with analysis_col1:
                    pressure_level = "High" if rrr > crr + 2 else "Medium" if rrr > crr else "Low"
                    st.metric("Pressure Level", pressure_level, f"RRR: {rrr:.2f}")
                
                with analysis_col2:
                    batting_strength = "Strong" if wickets_remaining >= 6 else "Medium" if wickets_remaining >= 3 else "Weak"
                    st.metric("Batting Depth", batting_strength, f"{wickets_remaining} wickets left")
                
                with analysis_col3:
                    time_factor = "Plenty" if balls_left >= 60 else "Moderate" if balls_left >= 30 else "Limited"
                    st.metric("Time Available", time_factor, f"{balls_left} balls left")

elif page == "Live Simulation":
    st.header("⚡ Live Match Simulation")
    
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = {
            'runs': 0,
            'wickets': 0,
            'balls': 0,
            'recent_balls': [],
            'is_running': False
        }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Simulation"):
            st.session_state.simulation_data['is_running'] = True
    
    with col2:
        if st.button("⏸️ Pause Simulation"):
            st.session_state.simulation_data['is_running'] = False
    
    with col3:
        if st.button("🔄 Reset Simulation"):
            st.session_state.simulation_data = {
                'runs': 0,
                'wickets': 0,
                'balls': 0,
                'recent_balls': [],
                'is_running': False
            }
    
    # Simulation display
    if st.session_state.simulation_data['is_running']:
        # Simulate a ball
        outcomes = ['0', '1', '2', '3', '4', '6', 'W', '0', '1', '2']
        outcome = random.choice(outcomes)
        
        if outcome == 'W':
            st.session_state.simulation_data['wickets'] += 1
        else:
            st.session_state.simulation_data['runs'] += int(outcome)
        
        st.session_state.simulation_data['balls'] += 1
        st.session_state.simulation_data['recent_balls'].insert(0, outcome)
        st.session_state.simulation_data['recent_balls'] = st.session_state.simulation_data['recent_balls'][:6]
        
        time.sleep(1)
        st.rerun()
    
    # Display current state
    sim_data = st.session_state.simulation_data
    overs_completed = sim_data['balls'] // 6
    balls_in_over = sim_data['balls'] % 6
    
    st.subheader(f"Score: {sim_data['runs']}/{sim_data['wickets']} ({overs_completed}.{balls_in_over} overs)")
    
    # Recent balls
    if sim_data['recent_balls']:
        st.subheader("Recent Balls:")
        cols = st.columns(len(sim_data['recent_balls']))
        for i, ball in enumerate(sim_data['recent_balls']):
            with cols[i]:
                color = "🔴" if ball == 'W' else "🟢" if ball in ['4', '6'] else "⚪"
                st.markdown(f"<div style='text-align: center; font-size: 2rem;'>{color}<br>{ball}</div>", unsafe_allow_html=True)

elif page == "Team Analytics":
    st.header("📊 Team Analytics Dashboard")
    
    # Mock team statistics
    team_stats = {
        'Team': teams,
        'Matches': [45, 42, 44, 43, 41, 40, 39, 38],
        'Wins': [28, 26, 24, 23, 21, 19, 18, 16],
        'Win Rate': [62.2, 61.9, 54.5, 53.5, 51.2, 47.5, 46.2, 42.1]
    }
    
    df_stats = pd.DataFrame(team_stats)
    
    # Team performance chart
    fig_performance = px.bar(df_stats, x='Team', y='Win Rate', 
                           color='Win Rate', color_continuous_scale='viridis',
                           title="Team Win Rate Comparison")
    fig_performance.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Detailed stats table
    st.subheader("Detailed Team Statistics")
    st.dataframe(df_stats, use_container_width=True)
    
    # Team comparison
    st.subheader("Team Comparison")
    selected_teams = st.multiselect("Select teams to compare", teams, default=teams[:3])
    
    if selected_teams:
        comparison_data = df_stats[df_stats['Team'].isin(selected_teams)]
        
        fig_comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Matches Played', 'Wins'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_comparison.add_trace(
            go.Bar(x=comparison_data['Team'], y=comparison_data['Matches'], name='Matches'),
            row=1, col=1
        )
        
        fig_comparison.add_trace(
            go.Bar(x=comparison_data['Team'], y=comparison_data['Wins'], name='Wins'),
            row=1, col=2
        )
        
        fig_comparison.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_comparison, use_container_width=True)

elif page == "Historical Data":
    st.header("📈 Historical Match Data")
    
    # Generate mock historical data
    @st.cache_data
    def generate_historical_data():
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='W')
        data = []
        for date in dates[:100]:  # Limit to 100 matches
            team1, team2 = random.sample(teams, 2)
            winner = random.choice([team1, team2])
            runs = random.randint(120, 220)
            data.append({
                'Date': date,
                'Team 1': team1,
                'Team 2': team2,
                'Winner': winner,
                'Runs': runs,
                'City': random.choice(cities[:10])  # Limit cities for better visualization
            })
        return pd.DataFrame(data)
    
    historical_df = generate_historical_data()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_filter = st.selectbox("Select Year", 
                                 options=['All'] + list(range(2020, 2025)),
                                 index=0)
    
    with col2:
        team_filter = st.selectbox("Select Team", 
                                 options=['All'] + teams,
                                 index=0)
    
    with col3:
        city_filter = st.selectbox("Select City", 
                                 options=['All'] + sorted(historical_df['City'].unique()),
                                 index=0)
    
    # Apply filters
    filtered_df = historical_df.copy()
    
    if year_filter != 'All':
        filtered_df = filtered_df[filtered_df['Date'].dt.year == year_filter]
    
    if team_filter != 'All':
        filtered_df = filtered_df[(filtered_df['Team 1'] == team_filter) | 
                                (filtered_df['Team 2'] == team_filter)]
    
    if city_filter != 'All':
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Display filtered data
    st.subheader(f"Showing {len(filtered_df)} matches")
    
    # Wins by team
    if team_filter == 'All':
        wins_by_team = filtered_df['Winner'].value_counts()
        fig_wins = px.pie(values=wins_by_team.values, names=wins_by_team.index,
                         title="Wins Distribution")
        st.plotly_chart(fig_wins, use_container_width=True)
    
    # Runs distribution
    fig_runs = px.histogram(filtered_df, x='Runs', nbins=20,
                           title="Runs Distribution in Matches")
    st.plotly_chart(fig_runs, use_container_width=True)
    
    # Match timeline
    if len(filtered_df) > 0:
        timeline_data = filtered_df.groupby(filtered_df['Date'].dt.date).size().reset_index()
        timeline_data.columns = ['Date', 'Matches']
        
        fig_timeline = px.line(timeline_data, x='Date', y='Matches',
                              title="Matches Over Time")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Data table
    st.subheader("Match Details")
    st.dataframe(filtered_df.sort_values('Date', ascending=False), use_container_width=True)

# Footer text with HTML formatting
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🏏 IPL Win Predictor - Powered by Advanced Machine Learning</p>
    <p>Made by Kishan Yadav ❤️  | Accuracy: 81.62% | Predictions: 50K+</p>
</div>
""", unsafe_allow_html=True)

# Add the proof image using use_container_width
st.image("proof.png", caption="Model Accuracy Proof", use_container_width=True)
