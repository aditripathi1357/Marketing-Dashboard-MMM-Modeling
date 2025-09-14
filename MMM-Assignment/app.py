import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Marketing Mix Model Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Marketing Mix Model Dashboard")
st.markdown("### Two-Stage MMM with Mediation Assumption")

# Sidebar
st.sidebar.header("🔧 Model Configuration")

class MMMPipeline:
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.scaler_stage1 = StandardScaler()
        self.scaler_stage2 = StandardScaler()
        self.feature_names_stage1 = None
        self.feature_names_stage2 = None
        
    def prepare_data(self, df):
        """Data preparation and feature engineering"""
        # Convert week to datetime if it's not already
        if 'week' in df.columns:
            df['week'] = pd.to_datetime(df['week'])
            df = df.sort_values('week').reset_index(drop=True)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Create lagged features (optional)
        lag_columns = ['facebook_spend', 'tiktok_spend', 'snapchat_spend', 'google_spend']
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1).fillna(0)
        
        # Log transform for skewed spend variables (add 1 to handle zeros)
        spend_cols = [col for col in df.columns if 'spend' in col]
        for col in spend_cols:
            df[f'{col}_log'] = np.log1p(df[col])
        
        return df
    
    def train_stage1(self, df, model_type='ridge'):
        """Stage 1: Predict google_spend from social channels"""
        # Features for Stage 1: Social channels -> Google spend
        feature_cols = ['facebook_spend', 'tiktok_spend', 'snapchat_spend']
        
        # Use log transformed features
        X_cols = [f'{col}_log' for col in feature_cols if f'{col}_log' in df.columns]
        if not X_cols:  # Fallback to original columns
            X_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[X_cols].values
        y = df['google_spend'].values
        
        # Scale features
        X_scaled = self.scaler_stage1.fit_transform(X)
        
        # Choose model
        if model_type == 'ridge':
            self.stage1_model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.stage1_model = Lasso(alpha=0.1)
        else:
            self.stage1_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train
        self.stage1_model.fit(X_scaled, y)
        self.feature_names_stage1 = X_cols
        
        # Predict google spend
        google_pred = self.stage1_model.predict(X_scaled)
        
        return google_pred, r2_score(y, google_pred), np.sqrt(mean_squared_error(y, google_pred))
    
    def train_stage2(self, df, google_pred, model_type='ridge'):
        """Stage 2: Predict revenue from predicted google_spend + other features"""
        # Features for Stage 2: Predicted Google + other marketing channels + controls
        base_features = ['email_sends', 'sms_sends', 'avg_price', 'followers', 'promotions']
        X_cols = [col for col in base_features if col in df.columns]
        
        # Create feature matrix
        X = df[X_cols].values
        # Add predicted google spend as a feature
        X = np.column_stack([X, google_pred.reshape(-1, 1)])
        X_cols.append('google_spend_predicted')
        
        y = df['revenue'].values
        
        # Scale features
        X_scaled = self.scaler_stage2.fit_transform(X)
        
        # Choose model
        if model_type == 'ridge':
            self.stage2_model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.stage2_model = Lasso(alpha=0.1)
        else:
            self.stage2_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train
        self.stage2_model.fit(X_scaled, y)
        self.feature_names_stage2 = X_cols
        
        # Predict revenue
        revenue_pred = self.stage2_model.predict(X_scaled)
        
        return revenue_pred, r2_score(y, revenue_pred), np.sqrt(mean_squared_error(y, revenue_pred))
    
    def time_series_cv(self, df, n_splits=5):
        """Time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            # Stage 1
            google_pred_train, _, _ = self.train_stage1(train_df)
            
            # Predict google for validation
            X_val_stage1 = val_df[[f'{col}_log' for col in ['facebook_spend', 'tiktok_spend', 'snapchat_spend'] 
                                 if f'{col}_log' in val_df.columns]].values
            X_val_stage1_scaled = self.scaler_stage1.transform(X_val_stage1)
            google_pred_val = self.stage1_model.predict(X_val_stage1_scaled)
            
            # Stage 2
            _, _, _ = self.train_stage2(train_df, google_pred_train)
            
            # Predict revenue for validation
            base_features = ['email_sends', 'sms_sends', 'avg_price', 'followers', 'promotions']
            X_val_stage2 = val_df[[col for col in base_features if col in val_df.columns]].values
            X_val_stage2 = np.column_stack([X_val_stage2, google_pred_val])
            X_val_stage2_scaled = self.scaler_stage2.transform(X_val_stage2)
            revenue_pred_val = self.stage2_model.predict(X_val_stage2_scaled)
            
            score = r2_score(val_df['revenue'], revenue_pred_val)
            cv_scores.append(score)
        
        return np.mean(cv_scores), np.std(cv_scores)

# File upload
uploaded_file = st.file_uploader("📁 Upload MMM Weekly.csv", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Initialize pipeline
    mmm = MMMPipeline()
    
    # Data overview
    st.header("📈 Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
    
    with col2:
        st.subheader("Revenue Statistics")
        st.write(df['revenue'].describe())
        
        st.subheader("Spend Channels")
        spend_cols = [col for col in df.columns if 'spend' in col]
        st.write(spend_cols)
    
    # Prepare data
    df_processed = mmm.prepare_data(df)
    
    # Time series plots
    st.header("📊 Time Series Trends")
    
    if 'week' in df_processed.columns:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Revenue Over Time', 'Google Spend', 'Social Spend Channels', 'Other Channels']
        )
        
        # Revenue
        fig.add_trace(
            go.Scatter(x=df_processed['week'], y=df_processed['revenue'], name='Revenue'),
            row=1, col=1
        )
        
        # Google spend
        fig.add_trace(
            go.Scatter(x=df_processed['week'], y=df_processed['google_spend'], name='Google', line=dict(color='red')),
            row=1, col=2
        )
        
        # Social channels
        social_channels = ['facebook_spend', 'tiktok_spend', 'snapchat_spend']
        for channel in social_channels:
            if channel in df_processed.columns:
                fig.add_trace(
                    go.Scatter(x=df_processed['week'], y=df_processed[channel], name=channel.replace('_spend', '').title()),
                    row=2, col=1
                )
        
        # Other channels
        other_channels = ['email_sends', 'sms_sends']
        for channel in other_channels:
            if channel in df_processed.columns:
                fig.add_trace(
                    go.Scatter(x=df_processed['week'], y=df_processed[channel], name=channel.replace('_', ' ').title()),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model training section
    st.header("🤖 Model Training")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Choose Model Type", ['ridge', 'lasso', 'random_forest'])
    with col2:
        run_cv = st.checkbox("Run Time Series Cross-Validation")
    
    if st.button("🚀 Train MMM Model"):
        with st.spinner("Training two-stage model..."):
            
            # Stage 1: Social -> Google
            st.subheader("Stage 1: Social Channels → Google Spend")
            google_pred, r2_stage1, rmse_stage1 = mmm.train_stage1(df_processed, model_type)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stage 1 R²", f"{r2_stage1:.3f}")
            with col2:
                st.metric("Stage 1 RMSE", f"{rmse_stage1:.0f}")
            
            # Stage 2: Google + Others -> Revenue
            st.subheader("Stage 2: Google Spend + Other Channels → Revenue")
            revenue_pred, r2_stage2, rmse_stage2 = mmm.train_stage2(df_processed, google_pred, model_type)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stage 2 R²", f"{r2_stage2:.3f}")
            with col2:
                st.metric("Stage 2 RMSE", f"{rmse_stage2:.0f}")
            
            # Cross-validation
            if run_cv:
                st.subheader("⏰ Time Series Cross-Validation")
                cv_mean, cv_std = mmm.time_series_cv(df_processed)
                st.metric("CV R² Score", f"{cv_mean:.3f} ± {cv_std:.3f}")
            
            # Model coefficients/importance
            st.subheader("📊 Model Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Stage 1 Feature Importance**")
                if hasattr(mmm.stage1_model, 'coef_'):
                    coef_df = pd.DataFrame({
                        'Feature': mmm.feature_names_stage1,
                        'Coefficient': mmm.stage1_model.coef_
                    })
                    fig_coef1 = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                     title='Stage 1: Social → Google')
                    st.plotly_chart(fig_coef1, use_container_width=True)
            
            with col2:
                st.write("**Stage 2 Feature Importance**")
                if hasattr(mmm.stage2_model, 'coef_'):
                    coef_df2 = pd.DataFrame({
                        'Feature': mmm.feature_names_stage2,
                        'Coefficient': mmm.stage2_model.coef_
                    })
                    fig_coef2 = px.bar(coef_df2, x='Coefficient', y='Feature', orientation='h',
                                     title='Stage 2: All Channels → Revenue')
                    st.plotly_chart(fig_coef2, use_container_width=True)
            
            # Predicted vs Actual
            st.subheader("🎯 Predicted vs Actual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pred1 = px.scatter(x=df_processed['google_spend'], y=google_pred,
                                     labels={'x': 'Actual Google Spend', 'y': 'Predicted Google Spend'},
                                     title='Stage 1: Google Spend Prediction')
                fig_pred1.add_shape(type="line", line=dict(dash="dash"), 
                                   x0=df_processed['google_spend'].min(), 
                                   y0=df_processed['google_spend'].min(),
                                   x1=df_processed['google_spend'].max(), 
                                   y1=df_processed['google_spend'].max())
                st.plotly_chart(fig_pred1, use_container_width=True)
            
            with col2:
                fig_pred2 = px.scatter(x=df_processed['revenue'], y=revenue_pred,
                                     labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                                     title='Stage 2: Revenue Prediction')
                fig_pred2.add_shape(type="line", line=dict(dash="dash"),
                                   x0=df_processed['revenue'].min(), 
                                   y0=df_processed['revenue'].min(),
                                   x1=df_processed['revenue'].max(), 
                                   y1=df_processed['revenue'].max())
                st.plotly_chart(fig_pred2, use_container_width=True)
            
            # Residual plots
            st.subheader("📈 Residual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                residuals1 = df_processed['google_spend'] - google_pred
                fig_res1 = px.scatter(x=google_pred, y=residuals1,
                                    labels={'x': 'Predicted Google Spend', 'y': 'Residuals'},
                                    title='Stage 1 Residuals')
                fig_res1.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res1, use_container_width=True)
            
            with col2:
                residuals2 = df_processed['revenue'] - revenue_pred
                fig_res2 = px.scatter(x=revenue_pred, y=residuals2,
                                    labels={'x': 'Predicted Revenue', 'y': 'Residuals'},
                                    title='Stage 2 Residuals')
                fig_res2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res2, use_container_width=True)
            
            # Store results in session state for scenario analysis
            st.session_state['mmm_trained'] = True
            st.session_state['mmm_pipeline'] = mmm
            st.session_state['df_processed'] = df_processed

    # Scenario Analysis
    if st.session_state.get('mmm_trained', False):
        st.header("🔮 Scenario Analysis")
        st.markdown("**What-if Analysis**: See how changes in social spending affect revenue through Google spend")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fb_change = st.slider("Facebook Spend Change (%)", -50, 100, 0)
        with col2:
            tiktok_change = st.slider("TikTok Spend Change (%)", -50, 100, 0)
        with col3:
            snapchat_change = st.slider("Snapchat Spend Change (%)", -50, 100, 0)
        
        if st.button("🧮 Calculate Impact"):
            mmm = st.session_state['mmm_pipeline']
            df_scenario = st.session_state['df_processed'].copy()
            
            # Apply changes
            df_scenario['facebook_spend'] *= (1 + fb_change/100)
            df_scenario['tiktok_spend'] *= (1 + tiktok_change/100)
            df_scenario['snapchat_spend'] *= (1 + snapchat_change/100)
            
            # Recalculate log transforms
            df_scenario['facebook_spend_log'] = np.log1p(df_scenario['facebook_spend'])
            df_scenario['tiktok_spend_log'] = np.log1p(df_scenario['tiktok_spend'])
            df_scenario['snapchat_spend_log'] = np.log1p(df_scenario['snapchat_spend'])
            
            # Predict new Google spend
            X_scenario_stage1 = df_scenario[mmm.feature_names_stage1].values
            X_scenario_stage1_scaled = mmm.scaler_stage1.transform(X_scenario_stage1)
            google_pred_new = mmm.stage1_model.predict(X_scenario_stage1_scaled)
            
            # Predict new revenue
            base_features = ['email_sends', 'sms_sends', 'avg_price', 'followers', 'promotions']
            X_scenario_stage2 = df_scenario[[col for col in base_features if col in df_scenario.columns]].values
            X_scenario_stage2 = np.column_stack([X_scenario_stage2, google_pred_new])
            X_scenario_stage2_scaled = mmm.scaler_stage2.transform(X_scenario_stage2)
            revenue_pred_new = mmm.stage2_model.predict(X_scenario_stage2_scaled)
            
            # Calculate impact
            original_revenue = st.session_state['df_processed']['revenue'].sum()
            new_revenue = revenue_pred_new.sum()
            revenue_impact = new_revenue - original_revenue
            revenue_impact_pct = (revenue_impact / original_revenue) * 100
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Revenue", f"${original_revenue:,.0f}")
            with col2:
                st.metric("New Revenue", f"${new_revenue:,.0f}")
            with col3:
                st.metric("Revenue Impact", f"${revenue_impact:,.0f}", f"{revenue_impact_pct:+.1f}%")

else:
    st.info("👆 Please upload your 'MMM Weekly.csv' file to get started!")
    
    st.markdown("""
    ### Expected CSV Format:
    - `week`: Date column (weekly data)
    - `facebook_spend`, `tiktok_spend`, `snapchat_spend`: Social media spend
    - `google_spend`: Search spend (mediator variable)
    - `email_sends`, `sms_sends`: Direct marketing
    - `avg_price`, `followers`, `promotions`: Control variables  
    - `revenue`: Target variable
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Mediation Assumption**: Social media drives search intent → Google spend → Revenue")