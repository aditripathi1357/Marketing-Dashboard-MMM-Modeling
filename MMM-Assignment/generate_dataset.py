import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_mmm_dataset():
    """
    Generate realistic Marketing Mix Model dataset with mediation structure:
    Social Media → Google Spend → Revenue
    """
    
    # Generate 2 years of weekly data (104 weeks)
    start_date = datetime(2022, 1, 3)  # Start on a Monday
    weeks = pd.date_range(start=start_date, periods=104, freq='W-MON')
    
    n_weeks = len(weeks)
    
    # Base parameters
    base_revenue = 50000
    
    # 1. Generate exogenous variables (not affected by other marketing)
    data = {
        'week': weeks,
        'avg_price': np.random.normal(29.99, 3, n_weeks),  # Product price
        'followers': np.random.poisson(15000, n_weeks) + np.arange(n_weeks) * 50,  # Growing followers
        'promotions': np.random.binomial(1, 0.25, n_weeks),  # 25% weeks have promotions
    }
    
    # Add seasonality factors
    week_of_year = [w.isocalendar()[1] for w in weeks]
    seasonality = np.sin(2 * np.pi * np.array(week_of_year) / 52) * 0.2 + 1
    holiday_boost = np.array([1.5 if w in [47, 48, 51, 52] else 1.0 for w in week_of_year])  # Black Friday, Christmas
    
    # 2. Generate social media spends (independent of each other)
    facebook_base = np.random.exponential(5000, n_weeks)
    tiktok_base = np.random.exponential(3000, n_weeks) 
    snapchat_base = np.random.exponential(2000, n_weeks)
    
    # Add some correlation between social platforms (realistic)
    social_trend = np.random.normal(1, 0.15, n_weeks)
    
    data['facebook_spend'] = facebook_base * social_trend * seasonality
    data['tiktok_spend'] = tiktok_base * social_trend * seasonality * 0.8  # Slightly different trend
    data['snapchat_spend'] = snapchat_base * social_trend * seasonality * 0.9
    
    # 3. Generate Google spend based on social media (MEDIATION EFFECT)
    # Social media creates search demand → Google spend
    social_effect = (
        0.15 * data['facebook_spend'] + 
        0.12 * data['tiktok_spend'] + 
        0.10 * data['snapchat_spend']
    )
    
    google_base = 8000 + 0.3 * social_effect + np.random.normal(0, 1000, n_weeks)
    data['google_spend'] = np.maximum(google_base, 0)  # Can't be negative
    
    # 4. Generate email and SMS (independent channels)
    data['email_sends'] = np.random.poisson(25000, n_weeks) * (1 + data['promotions'] * 0.5)  # More emails during promos
    data['sms_sends'] = np.random.poisson(5000, n_weeks) * (1 + data['promotions'] * 0.8)    # Even more SMS during promos
    
    # 5. Generate Revenue (MAIN OUTCOME)
    # Revenue depends on Google spend (mediated social effect) + direct channels + controls
    
    # Direct channel effects
    google_effect = 0.8 * data['google_spend']
    email_effect = 0.15 * data['email_sends'] 
    sms_effect = 0.25 * data['sms_sends']
    
    # Control variable effects
    price_effect = -800 * (data['avg_price'] - 30)  # Higher price reduces revenue
    follower_effect = 0.5 * (data['followers'] - 15000)  # More followers = more revenue
    promo_effect = 8000 * data['promotions']  # Promotions boost revenue
    
    # Combine all effects
    revenue_deterministic = (
        base_revenue +
        google_effect +
        email_effect + 
        sms_effect +
        price_effect +
        follower_effect +
        promo_effect
    ) * seasonality * holiday_boost
    
    # Add noise
    revenue_noise = np.random.normal(0, 3000, n_weeks)
    data['revenue'] = revenue_deterministic + revenue_noise
    
    # Ensure no negative values for spends and revenue
    for col in ['facebook_spend', 'tiktok_spend', 'snapchat_spend', 'google_spend', 'revenue']:
        data[col] = np.maximum(data[col], 0)
    
    # Round to realistic values
    data['facebook_spend'] = np.round(data['facebook_spend'], 0)
    data['tiktok_spend'] = np.round(data['tiktok_spend'], 0)
    data['snapchat_spend'] = np.round(data['snapchat_spend'], 0)
    data['google_spend'] = np.round(data['google_spend'], 0)
    data['revenue'] = np.round(data['revenue'], 0)
    data['avg_price'] = np.round(data['avg_price'], 2)
    data['followers'] = np.round(data['followers'], 0).astype(int)
    data['email_sends'] = data['email_sends'].astype(int)
    data['sms_sends'] = data['sms_sends'].astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match assignment specs
    df = df[['week', 'facebook_spend', 'tiktok_spend', 'snapchat_spend', 'google_spend', 
             'email_sends', 'sms_sends', 'avg_price', 'followers', 'promotions', 'revenue']]
    
    return df

def add_realistic_noise_and_gaps(df):
    """Add some realistic data issues"""
    df = df.copy()
    
    # Add a few missing values (realistic)
    missing_indices = np.random.choice(df.index, size=3, replace=False)
    df.loc[missing_indices, 'email_sends'] = np.nan
    
    # Add some zero-spend weeks (realistic for smaller channels)
    zero_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[zero_indices, 'snapchat_spend'] = 0
    
    # Add some outliers (campaign spikes)
    spike_indices = np.random.choice(df.index, size=2, replace=False)
    df.loc[spike_indices, 'facebook_spend'] *= 3  # Big campaign weeks
    
    return df

if __name__ == "__main__":
    print("🔄 Generating MMM Weekly dataset...")
    
    # Generate base dataset
    df = generate_mmm_dataset()
    
    # Add realistic data issues
    df = add_realistic_noise_and_gaps(df)
    
    # Save to CSV
    df.to_csv('MMM Weekly.csv', index=False)
    
    print("✅ Dataset generated successfully!")
    print(f"📊 Shape: {df.shape}")
    print("\n📋 Sample data:")
    print(df.head())
    
    print("\n📈 Summary statistics:")
    print(df.describe().round(0))
    
    print("\n🔍 Data quality check:")
    print("Missing values:", df.isnull().sum().sum())
    print("Zero spend weeks:", (df[['facebook_spend', 'tiktok_spend', 'snapchat_spend']] == 0).sum().sum())
    
    print("\n💡 Mediation structure built in:")
    print("- Social media spend influences Google spend")
    print("- Google spend (+ other channels) drives Revenue")  
    print("- Seasonality and promotions included")
    print("- 2 years of weekly data (104 weeks)")
    
    # Quick correlation check
    print("\n🔗 Key correlations:")
    social_cols = ['facebook_spend', 'tiktok_spend', 'snapchat_spend']
    print(f"Social → Google: {df[social_cols + ['google_spend']].corr()['google_spend'][:-1].mean():.3f}")
    print(f"Google → Revenue: {df[['google_spend', 'revenue']].corr().iloc[0,1]:.3f}")
    
    print("\n🚀 Ready to use with your Streamlit app!")