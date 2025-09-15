import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time
from scipy import stats
import os

# Configure page with custom theme
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and modern UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-out;
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        flex: 1;
        min-width: 200px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .kpi-card:hover::before {
        left: 100%;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0;
        font-family: 'Poppins', sans-serif;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Chart Containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .chart-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: #2c3e50;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Filter Section */
    .filter-header {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
        }
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Metric Delta Styling */
    .metric-delta-positive {
        color: #27ae60 !important;
        font-weight: 600;
    }
    
    .metric-delta-negative {
        color: #e74c3c !important;
        font-weight: 600;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .kpi-container {
            flex-direction: column;
        }
        
        .kpi-card {
            min-width: auto;
        }
    }
    
    /* Success/Info Messages */
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Error message styling */
    .error-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def get_file_path(filename):
    """Get the correct file path for CSV files"""
    # Check if file exists in current directory
    if os.path.exists(filename):
        return filename
    
    # Check common possible locations
    possible_paths = [
        filename,
        f"./{filename}",
        f"./data/{filename}",
        f"./marketing-dashboard/{filename}",
        f"../marketing-dashboard/{filename}",
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "data", filename),
        os.path.join(os.path.dirname(__file__), "marketing-dashboard", filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

@st.cache_data
def load_and_process_data():
    """Load and process all marketing and business data with loading animation"""
    try:
        # Show loading message
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            '<div style="text-align: center; padding: 2rem;"><div class="loading-spinner"></div><br>Loading your marketing data...</div>',
            unsafe_allow_html=True
        )
        
        # Simulate loading time for dramatic effect
        time.sleep(1)
        
        # Define required files
        required_files = ['Facebook.csv', 'Google.csv', 'TikTok.csv', 'Business.csv']
        file_paths = {}
        missing_files = []
        
        # Check for file existence
        for filename in required_files:
            file_path = get_file_path(filename)
            if file_path:
                file_paths[filename] = file_path
                st.success(f"✅ Found {filename} at: {file_path}")
            else:
                missing_files.append(filename)
        
        if missing_files:
            loading_placeholder.empty()
            st.error("🚨 Missing required CSV files:")
            for file in missing_files:
                st.error(f"❌ {file}")
            
            st.info("📁 Current working directory: " + os.getcwd())
            st.info("📂 Available files in current directory:")
            current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            for file in current_files:
                st.info(f"📄 {file}")
            
            return None, None, None
        
        # Load individual marketing channel data
        facebook_df = pd.read_csv(file_paths['Facebook.csv'])
        google_df = pd.read_csv(file_paths['Google.csv'])
        tiktok_df = pd.read_csv(file_paths['TikTok.csv'])
        
        # Rename 'impression' to 'impressions' and 'attributed revenue' to 'attributed_revenue'
        facebook_df = facebook_df.rename(columns={'impression': 'impressions', 'attributed revenue': 'attributed_revenue'})
        google_df = google_df.rename(columns={'impression': 'impressions', 'attributed revenue': 'attributed_revenue'})
        tiktok_df = tiktok_df.rename(columns={'impression': 'impressions', 'attributed revenue': 'attributed_revenue'})
        
        # Add channel column to each dataset
        facebook_df['channel'] = 'Facebook'
        google_df['channel'] = 'Google'
        tiktok_df['channel'] = 'TikTok'
        
        # Combine all marketing data
        marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)
        
        # Load business data
        business_df = pd.read_csv(file_paths['Business.csv'])
        
        # Convert date columns to datetime
        marketing_df['date'] = pd.to_datetime(marketing_df['date'])
        business_df['date'] = pd.to_datetime(business_df['date'])
        
        # Join marketing and business data on date
        combined_df = marketing_df.merge(business_df, on='date', how='left')
        
        # Ensure numeric columns and handle missing values
        numeric_cols = ['impressions', 'clicks', 'spend', 'attributed_revenue', 'total revenue', 'gross profit', 'COGS', 'new customers']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Create derived metrics
        combined_df['CTR'] = combined_df['clicks'] / combined_df['impressions']
        combined_df['ROAS'] = combined_df['attributed_revenue'] / combined_df['spend']
        combined_df['Profit_Margin'] = combined_df['gross profit'] / combined_df['total revenue']
        combined_df['CAC'] = combined_df['spend'] / combined_df['new customers']
        
        # Handle infinite and NaN values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        
        # Clear loading message
        loading_placeholder.empty()
        
        return combined_df, marketing_df, business_df
    
    except FileNotFoundError as e:
        st.markdown(f'<div class="error-message">🚨 CSV file not found: {str(e)}<br>Please ensure all required CSV files are uploaded to your Streamlit app.</div>', unsafe_allow_html=True)
        return None, None, None
    except Exception as e:
        st.markdown(f'<div class="error-message">⚠️ Error loading data: {str(e)}</div>', unsafe_allow_html=True)
        return None, None, None

def create_sample_data():
    """Create sample data if CSV files are not available"""
    st.info("🔧 Creating sample data for demonstration...")
    
    # Create sample date range
    date_range = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    
    # Create sample marketing data
    channels = ['Facebook', 'Google', 'TikTok']
    sample_data = []
    
    for date in date_range:
        for channel in channels:
            sample_data.append({
                'date': date,
                'channel': channel,
                'campaign': f'{channel}_Campaign_{np.random.randint(1, 5)}',
                'impressions': np.random.randint(1000, 10000),
                'clicks': np.random.randint(50, 500),
                'spend': np.random.uniform(100, 1000),
                'attributed_revenue': np.random.uniform(200, 1500),
                'orders': np.random.randint(1, 50),
                'total revenue': np.random.uniform(500, 2000),
                'gross profit': np.random.uniform(100, 800),
                'COGS': np.random.uniform(100, 500),
                'new customers': np.random.randint(1, 20)
            })
    
    combined_df = pd.DataFrame(sample_data)
    
    # Create derived metrics
    combined_df['CTR'] = combined_df['clicks'] / combined_df['impressions']
    combined_df['ROAS'] = combined_df['attributed_revenue'] / combined_df['spend']
    combined_df['Profit_Margin'] = combined_df['gross profit'] / combined_df['total revenue']
    combined_df['CAC'] = combined_df['spend'] / combined_df['new customers']
    
    return combined_df, combined_df, combined_df

def create_animated_kpi_cards(df):
    """Create animated KPI summary cards with modern design"""
    # Calculate KPIs
    total_spend = df['spend'].sum()
    total_attributed_revenue = df['attributed_revenue'].sum()
    total_profit = df['gross profit'].sum() if 'gross profit' in df.columns else 0
    avg_cac = df['CAC'].mean() if 'CAC' in df.columns else 0
    avg_roas = df['ROAS'].mean() if 'ROAS' in df.columns else 0
    
    # Create 5-column layout for KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    kpi_data = [
        ("💰", "Total Spend", f"${total_spend:,.0f}", "#ff6b6b", col1),
        ("📈", "Attributed Revenue", f"${total_attributed_revenue:,.0f}", "#4ecdc4", col2),
        ("💎", "Total Profit", f"${total_profit:,.0f}", "#45b7d1", col3),
        ("🎯", "Average CAC", f"${avg_cac:.2f}" if not pd.isna(avg_cac) else "N/A", "#96ceb4", col4),
        ("🚀", "Average ROAS", f"{avg_roas:.2f}x" if not pd.isna(avg_roas) else "N/A", "#ffeaa7", col5)
    ]
    
    for icon, label, value, color, column in kpi_data:
        with column:
            st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <h2 class="kpi-value">{value}</h2>
                    <p class="kpi-label">{label}</p>
                </div>
            """, unsafe_allow_html=True)

def create_modern_chart(fig, title, icon="📊"):
    """Wrap charts in modern containers with icons"""
    st.markdown(f"""
        <div class="chart-container">
            <h3 class="chart-title">{icon} {title}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhance chart styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, sans-serif", size=12, color="#2c3e50"),
        title_font=dict(size=16, color="#2c3e50"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, width='stretch')

def main():
    # Animated Header
    st.markdown("""
        <div class="main-header">
            <h1>🚀 Marketing Intelligence Dashboard</h1>
            <p>Transform your marketing data into actionable insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader section
    st.markdown("### 📁 Data Source")
    
    # Option to upload files
    uploaded_files = st.file_uploader(
        "Upload your CSV files (Facebook.csv, Google.csv, TikTok.csv, Business.csv)",
        accept_multiple_files=True,
        type=['csv'],
        help="Upload all 4 required CSV files to see your actual data"
    )
    
    # Process uploaded files
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        required_files = ['Facebook.csv', 'Google.csv', 'TikTok.csv', 'Business.csv']
        
        if all(req_file in file_names for req_file in required_files):
            st.success("✅ All required files uploaded successfully!")
            
            # Process uploaded files
            try:
                # Save uploaded files temporarily and process them
                facebook_df = pd.read_csv(next(f for f in uploaded_files if f.name == 'Facebook.csv'))
                google_df = pd.read_csv(next(f for f in uploaded_files if f.name == 'Google.csv'))
                tiktok_df = pd.read_csv(next(f for f in uploaded_files if f.name == 'TikTok.csv'))
                business_df = pd.read_csv(next(f for f in uploaded_files if f.name == 'Business.csv'))
                
                # Process data similar to the original function
                facebook_df = facebook_df.rename(columns={'impression': 'impressions', 'attributed revenue': 'attributed_revenue'})
                google_df = google_df.rename(columns={'impression': 'impressions', 'attributed revenue': 'attributed_revenue'})
                tiktok_df = tiktok_df.rename(columns={'impression': 'impressions', 'attributed revenue': 'attributed_revenue'})
                
                facebook_df['channel'] = 'Facebook'
                google_df['channel'] = 'Google'
                tiktok_df['channel'] = 'TikTok'
                
                marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)
                marketing_df['date'] = pd.to_datetime(marketing_df['date'])
                business_df['date'] = pd.to_datetime(business_df['date'])
                
                combined_df = marketing_df.merge(business_df, on='date', how='left')
                
                # Process numeric columns
                numeric_cols = ['impressions', 'clicks', 'spend', 'attributed_revenue', 'total revenue', 'gross profit', 'COGS', 'new customers']
                for col in numeric_cols:
                    if col in combined_df.columns:
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                
                # Create derived metrics
                combined_df['CTR'] = combined_df['clicks'] / combined_df['impressions']
                combined_df['ROAS'] = combined_df['attributed_revenue'] / combined_df['spend']
                combined_df['Profit_Margin'] = combined_df['gross profit'] / combined_df['total revenue']
                combined_df['CAC'] = combined_df['spend'] / combined_df['new customers']
                combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
                
            except Exception as e:
                st.error(f"Error processing uploaded files: {str(e)}")
                combined_df, marketing_df, business_df = create_sample_data()
        else:
            missing = [f for f in required_files if f not in file_names]
            st.warning(f"Missing files: {', '.join(missing)}. Using sample data for now.")
            combined_df, marketing_df, business_df = create_sample_data()
    else:
        # Try to load from filesystem first, then use sample data
        combined_df, marketing_df, business_df = load_and_process_data()
        
        if combined_df is None:
            st.info("📊 Using sample data for demonstration. Upload your CSV files above to see your actual data.")
            combined_df, marketing_df, business_df = create_sample_data()
    
    # Success message
    st.markdown("""
        <div class="success-message">
            ✅ Dashboard is ready for analysis!
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
            <div class="filter-header">
                🎛️ Dashboard Controls
            </div>
        """, unsafe_allow_html=True)
        
        # Channel filter with custom styling
        st.markdown("### 📺 Channel Selection")
        channels = ['All'] + list(combined_df['channel'].unique())
        selected_channel = st.selectbox("Choose Channel", channels, key="channel_filter")
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        min_date = combined_df['date'].min().date()
        max_date = combined_df['date'].max().date()
        
        date_range = st.date_input(
            "Select Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        
        # Advanced filters
        st.markdown("### ⚙️ Advanced Options")
        show_trends = st.checkbox("Show Trend Lines", value=True)
        animate_charts = st.checkbox("Animate Charts", value=True)
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        st.info(f"📋 Total Records: {len(combined_df):,}")
        st.info(f"🗓️ Date Range: {(max_date - min_date).days} days")
        st.info(f"📺 Channels: {len(combined_df['channel'].unique())}")
    
    # Apply filters
    filtered_df = combined_df.copy()
    
    # Filter by channel
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
    
    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Display KPI cards with animations
    st.markdown("## 📈 Key Performance Indicators")
    create_animated_kpi_cards(filtered_df)
    
    st.markdown("---")
    
    # Enhanced Charts section with better layout
    st.markdown("## 📊 Performance Analytics")
    
    # Row 1: Daily Spend vs Revenue & ROAS by Channel
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Daily Spend vs Revenue with enhanced styling
        daily_data = filtered_df.groupby('date').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['spend'],
            mode='lines+markers',
            name='💰 Spend',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=6, color='#ff6b6b', line=dict(width=2, color='white')),
            hovertemplate='<b>Spend</b><br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
        ))
        fig1.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['attributed_revenue'],
            mode='lines+markers',
            name='📈 Revenue',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=6, color='#4ecdc4', line=dict(width=2, color='white')),
            yaxis='y2',
            hovertemplate='<b>Revenue</b><br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
        ))
        
        fig1.update_layout(
            title="💰 Daily Spend vs Revenue Trends",
            xaxis_title='📅 Date',
            yaxis_title='💰 Spend ($)',
            yaxis2=dict(title='📈 Revenue ($)', overlaying='y', side='right'),
            height=400,
            hovermode='x unified'
        )
        
        if show_trends and len(daily_data) > 1:
            # Add trend lines
            x_numeric = pd.to_numeric(daily_data['date'])
            slope_spend, intercept_spend, r_spend, p_spend, std_err_spend = stats.linregress(x_numeric, daily_data['spend'])
            
            fig1.add_trace(go.Scatter(
                x=daily_data['date'],
                y=slope_spend * x_numeric + intercept_spend,
                mode='lines',
                name='Spend Trend',
                line=dict(color='#ff6b6b', dash='dash', width=2),
                opacity=0.7
            ))
        
        create_modern_chart(fig1, "Daily Spend vs Revenue Trends", "💰")
    
    with col2:
        # Enhanced ROAS by Channel
        channel_roas = filtered_df.groupby('channel').agg({
            'attributed_revenue': 'sum',
            'spend': 'sum'
        }).reset_index()
        channel_roas['ROAS'] = channel_roas['attributed_revenue'] / channel_roas['spend']
        
        # Create gradient colors
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        
        fig2 = px.bar(
            channel_roas,
            x='channel',
            y='ROAS',
            color='channel',
            color_discrete_sequence=colors,
            title='🎯 Return on Ad Spend by Channel'
        )
        
        fig2.update_traces(
            texttemplate='%{y:.2f}x',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ROAS: %{y:.2f}x<extra></extra>'
        )
        
        fig2.update_layout(height=400, showlegend=False)
        create_modern_chart(fig2, "ROAS by Channel", "🎯")
    
    # Row 2: Orders & Profit Trend & Spend Distribution
    col3, col4 = st.columns(2)
    
    with col3:
        # Enhanced Orders & Profit Trend
        if 'orders' in filtered_df.columns and 'gross profit' in filtered_df.columns:
            business_daily = filtered_df.groupby('date').agg({
                'orders': 'first',
                'gross profit': 'first'
            }).reset_index()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=business_daily['date'],
                y=business_daily['orders'],
                mode='lines+markers',
                name='📦 Orders',
                line=dict(color='#45b7d1', width=3),
                marker=dict(size=6, color='#45b7d1', line=dict(width=2, color='white')),
                fill='tozeroy',
                fillcolor='rgba(69, 183, 209, 0.1)'
            ))
            fig3.add_trace(go.Scatter(
                x=business_daily['date'],
                y=business_daily['gross profit'],
                mode='lines+markers',
                name='💎 Profit',
                line=dict(color='#96ceb4', width=3),
                marker=dict(size=6, color='#96ceb4', line=dict(width=2, color='white')),
                yaxis='y2'
            ))
            
            fig3.update_layout(
                title='📊 Orders & Profit Performance',
                xaxis_title='📅 Date',
                yaxis_title='📦 Orders',
                yaxis2=dict(title='💎 Profit ($)', overlaying='y', side='right'),
                height=400,
                hovermode='x unified'
            )
        else:
            # Create alternative chart if orders/profit data not available
            clicks_impressions = filtered_df.groupby('date').agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=clicks_impressions['date'],
                y=clicks_impressions['clicks'],
                mode='lines+markers',
                name='🖱️ Clicks',
                line=dict(color='#45b7d1', width=3),
                marker=dict(size=6, color='#45b7d1', line=dict(width=2, color='white'))
            ))
            fig3.add_trace(go.Scatter(
                x=clicks_impressions['date'],
                y=clicks_impressions['impressions'],
                mode='lines+markers',
                name='👀 Impressions',
                line=dict(color='#96ceb4', width=3),
                marker=dict(size=6, color='#96ceb4', line=dict(width=2, color='white')),
                yaxis='y2'
            ))
            
            fig3.update_layout(
                title='📊 Clicks & Impressions Performance',
                xaxis_title='📅 Date',
                yaxis_title='🖱️ Clicks',
                yaxis2=dict(title='👀 Impressions', overlaying='y', side='right'),
                height=400,
                hovermode='x unified'
            )
        
        create_modern_chart(fig3, "Performance Metrics", "📊")
    
    with col4:
        # Enhanced Spend Distribution
        channel_spend = filtered_df.groupby('channel')['spend'].sum().reset_index()
        
        fig4 = px.pie(
            channel_spend,
            values='spend',
            names='channel',
            title='💸 Marketing Spend Distribution',
            color_discrete_sequence=colors
        )
        
        fig4.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Spend: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05 if channel_spend.iloc[i]['spend'] == channel_spend['spend'].max() else 0 
                  for i in range(len(channel_spend))]
        )
        
        fig4.update_layout(height=400)
        create_modern_chart(fig4, "Marketing Spend Distribution", "💸")
    
    # Enhanced Data Analytics Section
    st.markdown("---")
    st.markdown("## 📋 Data Analytics & Insights")
    
    # Tabbed interface for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Statistics", "📈 Performance Metrics", "🔍 Raw Data", "💡 Insights"])
    
    with tab1:
        st.markdown("### 📊 Channel Performance Summary")
        summary_stats = filtered_df.groupby('channel').agg({
            'spend': ['sum', 'mean', 'std'],
            'impressions': ['sum', 'mean'],
            'clicks': ['sum', 'mean'],
            'attributed_revenue': ['sum', 'mean'],
            'CTR': ['mean', 'std'],
            'ROAS': ['mean', 'std']
        }).round(2)
        
        st.dataframe(summary_stats, width='stretch')
    
    with tab2:
        st.markdown("### 📈 Key Performance Metrics")
        
        # Create performance metrics cards
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        total_impressions = filtered_df['impressions'].sum()
        total_clicks = filtered_df['clicks'].sum()
        overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        overall_roas = filtered_df['attributed_revenue'].sum() / filtered_df['spend'].sum() if filtered_df['spend'].sum() > 0 else 0
        
        with perf_col1:
            st.metric("👀 Total Impressions", f"{total_impressions:,}")
        with perf_col2:
            st.metric("🖱️ Total Clicks", f"{total_clicks:,}")
        with perf_col3:
            st.metric("📊 Overall CTR", f"{overall_ctr:.2f}%")
        with perf_col4:
            st.metric("🎯 Overall ROAS", f"{overall_roas:.2f}x")
    
    with tab3:
        st.markdown("### 🔍 Detailed Raw Data")
        display_columns = ['date', 'channel', 'spend', 'impressions', 'clicks', 'attributed_revenue', 'CTR', 'ROAS']
        
        # Filter to only existing columns
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # Add campaign column if it exists
        if 'campaign' in filtered_df.columns:
            available_columns.insert(2, 'campaign')
        
        # Add business metrics if they exist
        business_columns = ['orders', 'total revenue', 'gross profit']
        for col in business_columns:
            if col in filtered_df.columns:
                available_columns.append(col)
        
        # Add search functionality
        search_term = st.text_input("🔍 Search campaigns:", placeholder="Enter campaign name...")
        
        display_df = filtered_df[available_columns].copy()
        if search_term and 'campaign' in display_df.columns:
            display_df = display_df[display_df['campaign'].str.contains(search_term, case=False, na=False)]
        
        st.dataframe(
            display_df.sort_values('date', ascending=False),
            width='stretch',
            height=400
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"marketing_data_{selected_channel}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.markdown("### 💡 Automated Insights")
        
        # Generate automatic insights
        if len(channel_roas) > 1:
            best_channel = channel_roas.loc[channel_roas['ROAS'].idxmax(), 'channel']
            best_roas = channel_roas.loc[channel_roas['ROAS'].idxmax(), 'ROAS']
            
            worst_channel = channel_roas.loc[channel_roas['ROAS'].idxmin(), 'channel']
            worst_roas = channel_roas.loc[channel_roas['ROAS'].idxmin(), 'ROAS']
            
            total_spend_sum = filtered_df['spend'].sum()
            total_revenue_sum = filtered_df['attributed_revenue'].sum()
            total_roi = ((total_revenue_sum - total_spend_sum) / total_spend_sum * 100) if total_spend_sum > 0 else 0
            
            insights = [
                f"🏆 **Best Performing Channel**: {best_channel} with {best_roas:.2f}x ROAS",
                f"⚠️ **Lowest Performing Channel**: {worst_channel} with {worst_roas:.2f}x ROAS",
                f"💰 **Overall ROI**: {total_roi:.1f}% return on marketing investment",
                f"📊 **Performance Gap**: {((best_roas/worst_roas-1)*100):.0f}% difference between best and worst channels"
            ]
            
            for insight in insights:
                st.markdown(insight)
                st.markdown("---")
        else:
            st.info("📊 Upload multiple channel data to see comparative insights.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
            🚀 Marketing Intelligence Dashboard | Built with Streamlit & Plotly
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()