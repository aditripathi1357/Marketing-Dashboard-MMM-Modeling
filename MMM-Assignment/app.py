import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Advanced Marketing Mix Model Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stage-header {
        background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>📊 Advanced Marketing Mix Model Dashboard</h1>
    <p>Two-Stage XGBoost MMM with Enhanced Analytics</p>
</div>
""", unsafe_allow_html=True)

class MMMPipelineXGB:
    def __init__(self):
        # XGBoost parameters optimized for marketing data
        self.stage1_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42
        }
        self.stage2_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.08,
            'max_depth': 4,
            'subsample': 0.85,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'seed': 42
        }
        self.stage1_num_boost_round = 200
        self.stage2_num_boost_round = 300
        self.scaler_stage1 = StandardScaler()
        self.scaler_stage2 = StandardScaler()
        self.stage1_model = None
        self.stage2_model = None
        self.feature_names_stage1 = None
        self.feature_names_stage2 = None

    def preprocess(self, df):
        """Enhanced data preprocessing with feature engineering"""
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(0)
        
        # Convert date column if exists
        date_cols = ['week', 'date', 'Week', 'Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col).reset_index(drop=True)
                break
        
        # Log transformation for spend variables (handles zeros)
        spend_cols = ['facebook_spend', 'tiktok_spend', 'snapchat_spend']
        for col in spend_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].fillna(0))
        
        # Create interaction features
        if all(f'{col}_log' in df.columns for col in spend_cols):
            df['social_total_log'] = df['facebook_spend_log'] + df['tiktok_spend_log'] + df['snapchat_spend_log']
            df['fb_tiktok_interaction'] = df['facebook_spend_log'] * df['tiktok_spend_log']
        
        # Adstock transformation (simple decay)
        for col in spend_cols:
            if col in df.columns:
                df[f'{col}_adstock'] = df[col].ewm(alpha=0.3).mean()
        
        return df

    def train_stage1(self, df):
        """Stage 1: Predict Google Spend from Social Channels using XGBoost"""
        # Features for Stage 1
        feature_cols = [f'{col}_log' for col in ['facebook_spend', 'tiktok_spend', 'snapchat_spend']]
        
        # Add interaction and adstock features if available
        additional_features = ['social_total_log', 'fb_tiktok_interaction']
        for feat in additional_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Filter available features
        self.feature_names_stage1 = [col for col in feature_cols if col in df.columns]
        
        X = df[self.feature_names_stage1].values
        y = df['google_spend'].values
        
        # Scale features
        X_scaled = self.scaler_stage1.fit_transform(X)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_scaled, label=y, feature_names=self.feature_names_stage1)
        
        # Train XGBoost model
        self.stage1_model = xgb.train(
            self.stage1_params,
            dtrain,
            num_boost_round=self.stage1_num_boost_round,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Predictions
        pred = self.stage1_model.predict(dtrain)
        
        # Calculate metrics
        metrics = self.compute_metrics(y, pred)
        
        return pred, metrics

    def train_stage2(self, df, google_pred):
        """Stage 2: Predict Revenue using XGBoost"""
        # Base features for Stage 2
        base_features = ['email_sends', 'sms_sends', 'avg_price', 'followers', 'promotions']
        available_features = [col for col in base_features if col in df.columns]
        
        # Create feature matrix
        X = df[available_features].values
        # Add predicted google spend
        X = np.column_stack([X, google_pred])
        available_features.append('google_spend_predicted')
        
        self.feature_names_stage2 = available_features
        y = df['revenue'].values
        
        # Scale features
        X_scaled = self.scaler_stage2.fit_transform(X)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_scaled, label=y, feature_names=self.feature_names_stage2)
        
        # Train XGBoost model
        self.stage2_model = xgb.train(
            self.stage2_params,
            dtrain,
            num_boost_round=self.stage2_num_boost_round,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=15,
            verbose_eval=False
        )
        
        # Predictions
        pred = self.stage2_model.predict(dtrain)
        
        # Calculate metrics
        metrics = self.compute_metrics(y, pred)
        
        return pred, metrics

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Comprehensive metrics calculation"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def time_series_cv(self, df, n_splits=5):
        """Enhanced time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (train_idx, val_idx) in enumerate(tscv.split(df)):
            status_text.text(f'Cross-validation fold {i+1}/{n_splits}...')
            progress_bar.progress((i+1) / n_splits)
            
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            
            # Stage 1
            google_pred_train, _ = self.train_stage1(train_df)
            
            # Predict google for validation
            X_val_stage1 = val_df[self.feature_names_stage1].values
            X_val_stage1_scaled = self.scaler_stage1.transform(X_val_stage1)
            dval_stage1 = xgb.DMatrix(X_val_stage1_scaled, feature_names=self.feature_names_stage1)
            google_pred_val = self.stage1_model.predict(dval_stage1)
            
            # Stage 2
            _, _ = self.train_stage2(train_df, google_pred_train)
            
            # Predict revenue for validation
            base_features = [col for col in ['email_sends', 'sms_sends', 'avg_price', 'followers', 'promotions'] 
                           if col in val_df.columns]
            X_val_stage2 = val_df[base_features].values
            X_val_stage2 = np.column_stack([X_val_stage2, google_pred_val])
            X_val_stage2_scaled = self.scaler_stage2.transform(X_val_stage2)
            dval_stage2 = xgb.DMatrix(X_val_stage2_scaled, feature_names=self.feature_names_stage2)
            revenue_pred_val = self.stage2_model.predict(dval_stage2)
            
            # Calculate metrics
            metrics = self.compute_metrics(val_df['revenue'].values, revenue_pred_val)
            cv_results.append(metrics)
        
        progress_bar.empty()
        status_text.empty()
        
        # Aggregate results
        avg_results = {}
        for metric in cv_results[0].keys():
            values = [fold[metric] for fold in cv_results]
            avg_results[f'{metric}_mean'] = np.mean(values)
            avg_results[f'{metric}_std'] = np.std(values)
        
        return avg_results

    def get_feature_importance(self, model, feature_names):
        """Get feature importance from XGBoost model"""
        if model is None:
            return pd.DataFrame()
        
        importance = model.get_score(importance_type='weight')
        
        # Create DataFrame
        importance_df = pd.DataFrame([
            {'Feature': k, 'Importance': v} 
            for k, v in importance.items()
        ])
        
        if len(importance_df) == 0:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': [0] * len(feature_names)
            })
        
        return importance_df.sort_values('Importance', ascending=True)

# Sidebar Configuration
st.sidebar.header("🔧 Model Configuration")

# XGBoost Parameters
with st.sidebar.expander("⚙️ XGBoost Parameters", expanded=False):
    learning_rate_s1 = st.slider("Stage 1 Learning Rate", 0.01, 0.3, 0.1, 0.01)
    learning_rate_s2 = st.slider("Stage 2 Learning Rate", 0.01, 0.3, 0.08, 0.01)
    max_depth_s1 = st.slider("Stage 1 Max Depth", 1, 10, 3)
    max_depth_s2 = st.slider("Stage 2 Max Depth", 1, 10, 4)

# Analysis Options
with st.sidebar.expander("📊 Analysis Options", expanded=True):
    run_cv = st.checkbox("Run Time Series Cross-Validation", value=True)
    cv_splits = st.slider("CV Splits", 3, 10, 5)
    show_feature_importance = st.checkbox("Show Feature Importance", value=True)
    show_residuals = st.checkbox("Show Residual Analysis", value=True)

# File Upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.file_uploader("Upload MMM Weekly.csv", type=['csv'])

if uploaded_file is not None:
    # Load and display data
    df = pd.read_csv(uploaded_file)
    
    # Data Overview Section
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Date Range", f"{df.shape[0]} weeks" if 'week' in df.columns else "N/A")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data Quality Check
    with st.expander("🔍 Data Quality Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue Statistics")
            st.dataframe(df['revenue'].describe())
        
        with col2:
            st.subheader("Spend Channels Available")
            spend_cols = [col for col in df.columns if 'spend' in col]
            st.write(spend_cols)
    
    # Initialize and prepare data
    mmm = MMMPipelineXGB()
    
    # Update parameters from sidebar
    mmm.stage1_params['learning_rate'] = learning_rate_s1
    mmm.stage1_params['max_depth'] = max_depth_s1
    mmm.stage2_params['learning_rate'] = learning_rate_s2
    mmm.stage2_params['max_depth'] = max_depth_s2
    
    df_processed = mmm.preprocess(df)
    
    # Time Series Visualization
    st.header("📊 Time Series Analysis")
    
    date_col = None
    for col in ['week', 'date', 'Week', 'Date']:
        if col in df_processed.columns:
            date_col = col
            break
    
    if date_col:
        # Create comprehensive time series plots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Revenue Trend', 'Google Spend Trend',
                'Social Media Spend', 'Marketing Channels',
                'Price & Promotions', 'Engagement Metrics'
            ],
            vertical_spacing=0.08
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(x=df_processed[date_col], y=df_processed['revenue'], 
                      name='Revenue', line=dict(color='#2E86AB', width=3)),
            row=1, col=1
        )
        
        # Google spend
        fig.add_trace(
            go.Scatter(x=df_processed[date_col], y=df_processed['google_spend'], 
                      name='Google', line=dict(color='#A23B72', width=2)),
            row=1, col=2
        )
        
        # Social channels
        social_channels = ['facebook_spend', 'tiktok_spend', 'snapchat_spend']
        colors = ['#F18F01', '#C73E1D', '#8E44AD']
        for i, (channel, color) in enumerate(zip(social_channels, colors)):
            if channel in df_processed.columns:
                fig.add_trace(
                    go.Scatter(x=df_processed[date_col], y=df_processed[channel], 
                              name=channel.replace('_spend', '').title(), 
                              line=dict(color=color)),
                    row=2, col=1
                )
        
        # Other marketing
        other_channels = ['email_sends', 'sms_sends']
        for channel in other_channels:
            if channel in df_processed.columns:
                fig.add_trace(
                    go.Scatter(x=df_processed[date_col], y=df_processed[channel], 
                              name=channel.replace('_', ' ').title()),
                    row=2, col=2
                )
        
        # Price and promotions
        if 'avg_price' in df_processed.columns:
            fig.add_trace(
                go.Scatter(x=df_processed[date_col], y=df_processed['avg_price'], 
                          name='Avg Price'),
                row=3, col=1
            )
        if 'promotions' in df_processed.columns:
            fig.add_trace(
                go.Scatter(x=df_processed[date_col], y=df_processed['promotions'], 
                          name='Promotions'),
                row=3, col=1
            )
        
        # Engagement
        if 'followers' in df_processed.columns:
            fig.add_trace(
                go.Scatter(x=df_processed[date_col], y=df_processed['followers'], 
                          name='Followers'),
                row=3, col=2
            )
        
        fig.update_layout(height=900, showlegend=True, title_text="Marketing Data Time Series")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Training Section
    st.markdown('<div class="stage-header"><h2>🤖 XGBoost Model Training</h2></div>', unsafe_allow_html=True)
    
    if st.button("🚀 Train Advanced MMM Model", type="primary"):
        with st.spinner("Training two-stage XGBoost model..."):
            
            # Stage 1 Training
            st.markdown('<div class="stage-header"><h3>Stage 1: Social Channels → Google Spend</h3></div>', unsafe_allow_html=True)
            
            google_pred, metrics_s1 = mmm.train_stage1(df_processed)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics_s1['r2']:.3f}")
            with col2:
                st.metric("RMSE", f"{metrics_s1['rmse']:.0f}")
            with col3:
                st.metric("MAE", f"{metrics_s1['mae']:.0f}")
            with col4:
                st.metric("MAPE", f"{metrics_s1['mape']:.1f}%")
            
            # Stage 2 Training
            st.markdown('<div class="stage-header"><h3>Stage 2: All Channels → Revenue</h3></div>', unsafe_allow_html=True)
            
            revenue_pred, metrics_s2 = mmm.train_stage2(df_processed, google_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics_s2['r2']:.3f}")
            with col2:
                st.metric("RMSE", f"{metrics_s2['rmse']:.0f}")
            with col3:
                st.metric("MAE", f"{metrics_s2['mae']:.0f}")
            with col4:
                st.metric("MAPE", f"{metrics_s2['mape']:.1f}%")
            
            # Cross-Validation
            if run_cv:
                st.markdown('<div class="stage-header"><h3>⏰ Time Series Cross-Validation</h3></div>', unsafe_allow_html=True)
                cv_results = mmm.time_series_cv(df_processed, n_splits=cv_splits)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("CV R²", f"{cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
                with col2:
                    st.metric("CV RMSE", f"{cv_results['rmse_mean']:.0f} ± {cv_results['rmse_std']:.0f}")
                with col3:
                    st.metric("CV MAE", f"{cv_results['mae_mean']:.0f} ± {cv_results['mae_std']:.0f}")
                with col4:
                    st.metric("CV MAPE", f"{cv_results['mape_mean']:.1f}% ± {cv_results['mape_std']:.1f}%")
            
            # Feature Importance
            if show_feature_importance:
                st.header("📊 Feature Importance Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    importance_s1 = mmm.get_feature_importance(mmm.stage1_model, mmm.feature_names_stage1)
                    if not importance_s1.empty:
                        fig_imp1 = px.bar(importance_s1, x='Importance', y='Feature', 
                                         orientation='h', title='Stage 1: Social → Google Spend')
                        fig_imp1.update_layout(height=400)
                        st.plotly_chart(fig_imp1, use_container_width=True)
                
                with col2:
                    importance_s2 = mmm.get_feature_importance(mmm.stage2_model, mmm.feature_names_stage2)
                    if not importance_s2.empty:
                        fig_imp2 = px.bar(importance_s2, x='Importance', y='Feature', 
                                         orientation='h', title='Stage 2: All Channels → Revenue')
                        fig_imp2.update_layout(height=400)
                        st.plotly_chart(fig_imp2, use_container_width=True)
            
            # Prediction Analysis
            st.header("🎯 Prediction Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stage 1 predictions
                fig_pred1 = px.scatter(
                    x=df_processed['google_spend'], y=google_pred,
                    labels={'x': 'Actual Google Spend', 'y': 'Predicted Google Spend'},
                    title=f'Stage 1: Google Spend Prediction (R² = {metrics_s1["r2"]:.3f})'
                )
                fig_pred1.add_shape(
                    type="line", line=dict(dash="dash", color="red"),
                    x0=df_processed['google_spend'].min(), 
                    y0=df_processed['google_spend'].min(),
                    x1=df_processed['google_spend'].max(), 
                    y1=df_processed['google_spend'].max()
                )
                st.plotly_chart(fig_pred1, use_container_width=True)
            
            with col2:
                # Stage 2 predictions
                fig_pred2 = px.scatter(
                    x=df_processed['revenue'], y=revenue_pred,
                    labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                    title=f'Stage 2: Revenue Prediction (R² = {metrics_s2["r2"]:.3f})'
                )
                fig_pred2.add_shape(
                    type="line", line=dict(dash="dash", color="red"),
                    x0=df_processed['revenue'].min(), 
                    y0=df_processed['revenue'].min(),
                    x1=df_processed['revenue'].max(), 
                    y1=df_processed['revenue'].max()
                )
                st.plotly_chart(fig_pred2, use_container_width=True)
            
            # Residual Analysis
            if show_residuals:
                st.header("📈 Residual Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    residuals1 = df_processed['google_spend'] - google_pred
                    fig_res1 = px.scatter(
                        x=google_pred, y=residuals1,
                        labels={'x': 'Predicted Google Spend', 'y': 'Residuals'},
                        title='Stage 1: Residuals vs Predicted'
                    )
                    fig_res1.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res1, use_container_width=True)
                
                with col2:
                    residuals2 = df_processed['revenue'] - revenue_pred
                    fig_res2 = px.scatter(
                        x=revenue_pred, y=residuals2,
                        labels={'x': 'Predicted Revenue', 'y': 'Residuals'},
                        title='Stage 2: Residuals vs Predicted'
                    )
                    fig_res2.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res2, use_container_width=True)
            
            # Time Series Prediction Plot
            if date_col:
                st.header("📅 Time Series Predictions")
                fig_ts = go.Figure()
                
                fig_ts.add_trace(go.Scatter(
                    x=df_processed[date_col], 
                    y=df_processed['revenue'],
                    mode='lines+markers',
                    name='Actual Revenue',
                    line=dict(color='blue', width=2)
                ))
                
                fig_ts.add_trace(go.Scatter(
                    x=df_processed[date_col], 
                    y=revenue_pred,
                    mode='lines+markers',
                    name='Predicted Revenue',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                fig_ts.update_layout(
                    title='Revenue: Actual vs Predicted Over Time',
                    xaxis_title='Date',
                    yaxis_title='Revenue',
                    height=500
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
            
            # Store results for scenario analysis
            st.session_state['mmm_trained'] = True
            st.session_state['mmm_pipeline'] = mmm
            st.session_state['df_processed'] = df_processed
            st.session_state['google_pred'] = google_pred
            st.session_state['revenue_pred'] = revenue_pred

    # Scenario Analysis
    if st.session_state.get('mmm_trained', False):
        st.header("🔮 Advanced Scenario Analysis")
        st.markdown("**What-if Analysis**: Simulate marketing spend changes and see the impact on revenue through the mediation effect")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fb_change = st.slider("Facebook Spend Change (%)", -50, 100, 0, 5)
        with col2:
            tiktok_change = st.slider("TikTok Spend Change (%)", -50, 100, 0, 5)
        with col3:
            snapchat_change = st.slider("Snapchat Spend Change (%)", -50, 100, 0, 5)
        
        # Additional scenario options
        with st.expander("🎛️ Advanced Scenario Options"):
            col1, col2 = st.columns(2)
            with col1:
                scenario_weeks = st.slider("Apply changes for how many weeks?", 1, len(df_processed), len(df_processed))
                start_week = st.slider("Start from week", 1, len(df_processed), 1)
            with col2:
                show_breakdown = st.checkbox("Show detailed breakdown", True)
                compare_scenarios = st.checkbox("Compare multiple scenarios", False)
        
        if st.button("🧮 Calculate Impact", type="primary"):
            mmm = st.session_state['mmm_pipeline']
            df_scenario = st.session_state['df_processed'].copy()
            
            # Apply changes to selected weeks
            end_week = min(start_week + scenario_weeks - 1, len(df_scenario))
            
            # Apply percentage changes
            df_scenario.loc[start_week-1:end_week-1, 'facebook_spend'] *= (1 + fb_change/100)
            df_scenario.loc[start_week-1:end_week-1, 'tiktok_spend'] *= (1 + tiktok_change/100)
            df_scenario.loc[start_week-1:end_week-1, 'snapchat_spend'] *= (1 + snapchat_change/100)
            
            # Recalculate log transforms and features
            for col in ['facebook_spend', 'tiktok_spend', 'snapchat_spend']:
                df_scenario[f'{col}_log'] = np.log1p(df_scenario[col])
            
            # Predict new outcomes
            X_scenario_s1 = df_scenario[mmm.feature_names_stage1].values
            X_scenario_s1_scaled = mmm.scaler_stage1.transform(X_scenario_s1)
            dscenario_s1 = xgb.DMatrix(X_scenario_s1_scaled, feature_names=mmm.feature_names_stage1)
            google_pred_new = mmm.stage1_model.predict(dscenario_s1)
            
            # Stage 2 prediction
            base_features = [col for col in mmm.feature_names_stage2 if col != 'google_spend_predicted']
            X_scenario_s2 = df_scenario[base_features].values
            X_scenario_s2 = np.column_stack([X_scenario_s2, google_pred_new])
            X_scenario_s2_scaled = mmm.scaler_stage2.transform(X_scenario_s2)
            dscenario_s2 = xgb.DMatrix(X_scenario_s2_scaled, feature_names=mmm.feature_names_stage2)
            revenue_pred_new = mmm.stage2_model.predict(dscenario_s2)
            
            # Calculate impacts
            original_revenue = st.session_state['df_processed']['revenue'].sum()
            original_google = st.session_state['df_processed']['google_spend'].sum()
            
            new_revenue_total = revenue_pred_new.sum()
            new_google_total = google_pred_new.sum()
            
            revenue_impact = new_revenue_total - original_revenue
            google_impact = new_google_total - original_google
            
            revenue_impact_pct = (revenue_impact / original_revenue) * 100
            google_impact_pct = (google_impact / original_google) * 100
            
            # Calculate incremental spend
            original_spend = (
                df_processed['facebook_spend'].sum() + 
                df_processed['tiktok_spend'].sum() + 
                df_processed['snapchat_spend'].sum()
            )
            
            new_spend = (
                df_scenario['facebook_spend'].sum() + 
                df_scenario['tiktok_spend'].sum() + 
                df_scenario['snapchat_spend'].sum()
            )
            
            incremental_spend = new_spend - original_spend
            
            # ROI Calculation
            roi = (revenue_impact / incremental_spend) if incremental_spend > 0 else 0
            
            # Display Results
            st.subheader("💰 Impact Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Revenue Impact", 
                    f"${revenue_impact:,.0f}", 
                    f"{revenue_impact_pct:+.1f}%"
                )
            with col2:
                st.metric(
                    "Google Spend Impact", 
                    f"${google_impact:,.0f}", 
                    f"{google_impact_pct:+.1f}%"
                )
            with col3:
                st.metric(
                    "Incremental Spend", 
                    f"${incremental_spend:,.0f}"
                )
            with col4:
                st.metric(
                    "ROI", 
                    f"{roi:.2f}x" if roi > 0 else "N/A"
                )
            
            # Detailed Breakdown
            if show_breakdown:
                st.subheader("📊 Detailed Breakdown")
                
                breakdown_data = {
                    'Channel': ['Facebook', 'TikTok', 'Snapchat', 'Total Social'],
                    'Original Spend': [
                        df_processed['facebook_spend'].sum(),
                        df_processed['tiktok_spend'].sum(),
                        df_processed['snapchat_spend'].sum(),
                        original_spend
                    ],
                    'New Spend': [
                        df_scenario['facebook_spend'].sum(),
                        df_scenario['tiktok_spend'].sum(),
                        df_scenario['snapchat_spend'].sum(),
                        new_spend
                    ],
                    'Change (%)': [fb_change, tiktok_change, snapchat_change, 
                                 ((new_spend - original_spend) / original_spend * 100)]
                }
                
                breakdown_df = pd.DataFrame(breakdown_data)
                breakdown_df['Spend Change ($)'] = breakdown_df['New Spend'] - breakdown_df['Original Spend']
                
                st.dataframe(breakdown_df.round(2))
            
            # Visualization of scenario impact
            st.subheader("📈 Scenario Visualization")
            
            if date_col and date_col in df_processed.columns:
                fig_scenario = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'Revenue: Original vs Scenario',
                        'Google Spend: Original vs Scenario',
                        'Social Spend Changes',
                        'Weekly Revenue Impact'
                    ]
                )
                
                # Revenue comparison
                fig_scenario.add_trace(
                    go.Scatter(x=df_processed[date_col], y=st.session_state['revenue_pred'], 
                              name='Original Revenue', line=dict(color='blue')),
                    row=1, col=1
                )
                fig_scenario.add_trace(
                    go.Scatter(x=df_scenario[date_col], y=revenue_pred_new, 
                              name='Scenario Revenue', line=dict(color='red', dash='dot')),
                    row=1, col=1
                )
                
                # Google spend comparison
                fig_scenario.add_trace(
                    go.Scatter(x=df_processed[date_col], y=st.session_state['google_pred'], 
                              name='Original Google', line=dict(color='green')),
                    row=1, col=2
                )
                fig_scenario.add_trace(
                    go.Scatter(x=df_scenario[date_col], y=google_pred_new, 
                              name='Scenario Google', line=dict(color='orange', dash='dot')),
                    row=1, col=2
                )
                
                # Social spend changes
                fig_scenario.add_trace(
                    go.Scatter(x=df_processed[date_col], y=df_processed['facebook_spend'], 
                              name='Original FB', line=dict(color='#1877F2')),
                    row=2, col=1
                )
                fig_scenario.add_trace(
                    go.Scatter(x=df_scenario[date_col], y=df_scenario['facebook_spend'], 
                              name='Scenario FB', line=dict(color='#1877F2', dash='dot')),
                    row=2, col=1
                )
                
                # Weekly impact
                weekly_impact = revenue_pred_new - st.session_state['revenue_pred']
                fig_scenario.add_trace(
                    go.Bar(x=df_processed[date_col], y=weekly_impact, 
                           name='Weekly Revenue Impact', marker_color='purple'),
                    row=2, col=2
                )
                
                fig_scenario.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig_scenario, use_container_width=True)
            
            # Multiple scenario comparison
            if compare_scenarios:
                st.subheader("🔄 Multiple Scenario Comparison")
                
                scenarios = [
                    {"name": "Conservative", "fb": fb_change * 0.5, "tt": tiktok_change * 0.5, "sc": snapchat_change * 0.5},
                    {"name": "Current", "fb": fb_change, "tt": tiktok_change, "sc": snapchat_change},
                    {"name": "Aggressive", "fb": fb_change * 1.5, "tt": tiktok_change * 1.5, "sc": snapchat_change * 1.5}
                ]
                
                scenario_results = []
                
                for scenario in scenarios:
                    df_temp = st.session_state['df_processed'].copy()
                    
                    # Apply changes
                    df_temp['facebook_spend'] *= (1 + scenario['fb']/100)
                    df_temp['tiktok_spend'] *= (1 + scenario['tt']/100)
                    df_temp['snapchat_spend'] *= (1 + scenario['sc']/100)
                    
                    # Recalculate features
                    for col in ['facebook_spend', 'tiktok_spend', 'snapchat_spend']:
                        df_temp[f'{col}_log'] = np.log1p(df_temp[col])
                    
                    # Predict
                    X_temp_s1 = df_temp[mmm.feature_names_stage1].values
                    X_temp_s1_scaled = mmm.scaler_stage1.transform(X_temp_s1)
                    dtemp_s1 = xgb.DMatrix(X_temp_s1_scaled, feature_names=mmm.feature_names_stage1)
                    google_temp = mmm.stage1_model.predict(dtemp_s1)
                    
                    base_features = [col for col in mmm.feature_names_stage2 if col != 'google_spend_predicted']
                    X_temp_s2 = df_temp[base_features].values
                    X_temp_s2 = np.column_stack([X_temp_s2, google_temp])
                    X_temp_s2_scaled = mmm.scaler_stage2.transform(X_temp_s2)
                    dtemp_s2 = xgb.DMatrix(X_temp_s2_scaled, feature_names=mmm.feature_names_stage2)
                    revenue_temp = mmm.stage2_model.predict(dtemp_s2)
                    
                    # Calculate metrics
                    temp_spend = (df_temp['facebook_spend'].sum() + 
                                df_temp['tiktok_spend'].sum() + 
                                df_temp['snapchat_spend'].sum())
                    temp_incremental = temp_spend - original_spend
                    temp_revenue_impact = revenue_temp.sum() - original_revenue
                    temp_roi = temp_revenue_impact / temp_incremental if temp_incremental > 0 else 0
                    
                    scenario_results.append({
                        'Scenario': scenario['name'],
                        'Revenue Impact ($)': temp_revenue_impact,
                        'Revenue Impact (%)': (temp_revenue_impact / original_revenue) * 100,
                        'Incremental Spend ($)': temp_incremental,
                        'ROI': temp_roi
                    })
                
                scenario_df = pd.DataFrame(scenario_results)
                st.dataframe(scenario_df.round(2))
                
                # Scenario comparison chart
                fig_comparison = px.bar(
                    scenario_df, 
                    x='Scenario', 
                    y='Revenue Impact ($)',
                    title='Revenue Impact Comparison Across Scenarios',
                    color='ROI',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

else:
    # Landing page when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>🚀 Welcome to Advanced Marketing Mix Modeling</h2>
        <p style="font-size: 1.2em; color: #666;">
            Upload your marketing data to get started with our enhanced XGBoost-powered MMM analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Expected data format
    st.info("👆 Please upload your 'MMM Weekly.csv' file to get started!")
    
    with st.expander("📋 Expected CSV Format & Requirements", expanded=True):
        st.markdown("""
        ### Required Columns:
        - **`week`** or **`date`**: Date column (weekly data points)
        - **`facebook_spend`**, **`tiktok_spend`**, **`snapchat_spend`**: Social media advertising spend
        - **`google_spend`**: Search advertising spend (mediator variable in our model)
        - **`email_sends`**, **`sms_sends`**: Direct marketing activities
        - **`avg_price`**: Average product price
        - **`followers`**: Social media followers count
        - **`promotions`**: Number of promotional campaigns
        - **`revenue`**: Target variable (what we're trying to predict)
        
        ### Data Quality Tips:
        - ✅ Ensure consistent weekly data (no missing weeks)
        - ✅ Handle missing values appropriately (0 for spend, forward-fill for prices)
        - ✅ Use consistent units (all spend in same currency)
        - ✅ Include at least 52 weeks of data for robust analysis
        
        ### Model Architecture:
        Our two-stage XGBoost model assumes:
        1. **Stage 1**: Social media spend influences Google spend (search intent)
        2. **Stage 2**: Google spend + other channels drive revenue
        """)
    
    # Sample data format
    with st.expander("📊 Sample Data Preview"):
        sample_data = pd.DataFrame({
            'week': pd.date_range('2023-01-01', periods=8, freq='W'),
            'facebook_spend': [5000, 4500, 6000, 5500, 4800, 5200, 5800, 6200],
            'tiktok_spend': [2000, 2200, 1800, 2100, 2300, 1900, 2400, 2000],
            'snapchat_spend': [1500, 1600, 1400, 1700, 1800, 1300, 1900, 1600],
            'google_spend': [8000, 7500, 9000, 8200, 7800, 8400, 9200, 8800],
            'email_sends': [50000, 48000, 52000, 49000, 51000, 47000, 53000, 50000],
            'sms_sends': [15000, 16000, 14000, 15500, 16500, 14500, 17000, 15800],
            'avg_price': [49.99, 49.99, 47.99, 49.99, 49.99, 52.99, 47.99, 49.99],
            'followers': [125000, 126500, 128000, 129200, 130800, 132100, 133500, 135000],
            'promotions': [2, 1, 3, 2, 1, 2, 3, 2],
            'revenue': [450000, 420000, 480000, 460000, 440000, 470000, 510000, 490000]
        })
        st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💡 <strong>Mediation Modeling Approach</strong>: Social media drives search intent → Google spend → Revenue</p>
    <p>🔧 Powered by XGBoost with advanced feature engineering and time series validation</p>
</div>
""", unsafe_allow_html=True)