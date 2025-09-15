# LIFESIGHTPROJECT - Marketing Analytics & Intelligence Suite

![Project Structure](https://img.shields.io/badge/Projects-2-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)

A comprehensive analytics suite featuring two powerful marketing intelligence solutions:
1. **Marketing Intelligence Dashboard** - Interactive BI dashboard for marketing performance analysis
2. **Advanced Marketing Mix Modeling (MMM)** - Two-stage XGBoost model with mediation analysis

---

## 📁 Project Structure

```
LIFESIGHTPROJECT/
├── marketing-dashboard/           # Assessment 1: BI Dashboard
│   ├── app.py                    # Main dashboard application
│   ├── business.csv              # Daily business performance data
│   ├── Facebook.csv              # Facebook campaign data
│   ├── Google.csv                # Google campaign data
│   ├── TikTok.csv               # TikTok campaign data
│   ├── README.md                 # Dashboard documentation
│   └── requirements.txt          # Dependencies
│
└── MMM-Assignment/               # Assessment 2: Marketing Mix Modeling
    ├── app.py                    # MMM dashboard application
    ├── generate_dataset.py       # Sample data generator
    ├── README.md                 # MMM documentation
    └── requirements.txt          # Dependencies
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
https://github.com/aditripathi1357/Marketing-Dashboard-MMM-Modeling.git
cd LIFESIGHTPROJECT
```

2. **Choose your project and install dependencies**

For Marketing Dashboard:
```bash
cd marketing-dashboard
pip install -r requirements.txt
streamlit run app.py
```

For MMM Assignment:
```bash
cd MMM-Assignment
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Assessment 1: Marketing Intelligence Dashboard

### Overview
An interactive BI dashboard analyzing 120 days of marketing and business activity across Facebook, Google, and TikTok channels. Transforms raw marketing data into actionable business insights.

![Marketing Dashboard](https://drive.google.com/file/d/1iLW2v_76AoGstJDug_5acMKGLxGxsn4y/view?usp=drive_link)
*Interactive dashboard showing KPIs, performance analytics, and trend analysis*

### Key Features

#### 📈 Performance Metrics
- **Total Marketing Spend**: $5.26M across all channels
- **Attributed Revenue**: $14.77M with ROI tracking
- **Total Profit**: $517.27M business performance
- **Average CAC**: $1.25 customer acquisition cost
- **ROAS**: 2.82x average return on ad spend

#### 🎯 Interactive Analytics
- Multi-channel performance comparison
- Real-time spend vs revenue analysis
- Daily trend visualization with statistical insights
- Campaign-level search and filtering
- Automated insights generation

### Data Requirements

#### Marketing Data Files
- **Facebook.csv, Google.csv, TikTok.csv**
```
- date: Campaign date (YYYY-MM-DD)
- tactic: Marketing tactic used
- state: Geographic state
- campaign: Campaign identifier
- impression: Number of impressions
- clicks: Number of clicks
- spend: Amount spent ($)
- attributed_revenue: Revenue attributed ($)
```

#### Business Data File
- **Business.csv**
```
- date: Business date (YYYY-MM-DD)
- # of orders: Total daily orders
- # of new orders: New orders count
- new customers: New customer acquisitions
- total revenue: Total daily revenue ($)
- gross profit: Daily gross profit ($)
- COGS: Cost of goods sold ($)
```

### Usage Instructions

1. **Launch Dashboard**
```bash
cd marketing-dashboard
streamlit run app.py
# Access at: http://localhost:8501
```

2. **Dashboard Navigation**
   - **Channel Selection**: Filter by Facebook, Google, TikTok, or All
   - **Advanced Options**: Toggle trend lines and chart animations
   - **Performance Analytics**: View spend vs revenue trends
   - **Data Analytics**: Access raw data and insights

3. **Key Insights Available**
   - Channel performance comparison with ROAS analysis
   - Daily marketing efficiency trends
   - Customer acquisition cost optimization
   - Profit margin and business outcome tracking

---

## 🧠 Assessment 2: Advanced Marketing Mix Modeling (MMM)

### Overview
A sophisticated two-stage XGBoost model implementing causal mediation analysis. Models the assumption that social media spend drives search intent, which influences Google spend, ultimately leading to revenue.

![MMM Dashboard](https://drive.google.com/file/d/1LX0vRAB0PKo0ACWoBCqIhDHKZit0iBnF/view?usp=drive_link)
*Two-stage XGBoost MMM with enhanced analytics and scenario planning*

### Model Architecture

#### Causal Framework
```
Social Media Spend → Search Intent → Google Spend → Revenue
    (Stage 1: XGBoost)              (Stage 2: XGBoost)
```

#### Stage 1: Social → Google Spend Prediction
- **R² Performance**: 0.932 (93.2% variance explained)
- **Features**: Log-transformed social spend, interactions, adstock effects
- **Purpose**: Capture mediation effect of social channels on search behavior

#### Stage 2: All Channels → Revenue Prediction  
- **R² Performance**: 0.925 (92.5% variance explained)
- **Features**: Predicted Google spend, direct channels, price, promotions
- **Purpose**: Model complete path to revenue generation

### Data Requirements

Upload CSV with required columns:
```
- week/date: Weekly time periods
- facebook_spend: Facebook advertising spend
- tiktok_spend: TikTok advertising spend  
- snapchat_spend: Snapchat advertising spend
- google_spend: Google advertising spend
- email_sends: Email marketing volume
- sms_sends: SMS marketing volume
- avg_price: Average product price
- followers: Social media followers
- promotions: Number of promotions
- revenue: Weekly revenue (target variable)
```

### Advanced Features

#### 🔧 Interactive Model Configuration
- **XGBoost Parameter Tuning**: Learning rate, depth, regularization
- **Cross-Validation Options**: Time series CV with configurable folds
- **Feature Engineering**: Automatic log transforms and interactions
- **Adstock Modeling**: Carryover effect implementation

#### 📊 Comprehensive Analytics
- **Model Performance**: R², RMSE, MAE metrics for both stages
- **Feature Importance**: Interactive visualization of channel contributions  
- **Residual Analysis**: Diagnostic plots for model validation
- **Prediction Analysis**: Actual vs predicted scatter plots

#### 🎯 Scenario Planning
- **What-if Analysis**: Slider-based spend adjustments (-50% to +100%)
- **ROI Calculation**: Automatic return on investment computation
- **Multi-scenario Comparison**: Conservative, current, aggressive planning
- **Impact Visualization**: Weekly revenue impact projections

### Usage Instructions

1. **Launch MMM Dashboard**
```bash
cd MMM-Assignment
streamlit run app.py
# Access at: http://localhost:8501
```

2. **Model Workflow**
   - **Data Upload**: Drag-and-drop CSV file with required format
   - **Configuration**: Adjust XGBoost parameters and analysis options
   - **Training**: Click "Train Advanced MMM Model"
   - **Analysis**: Review performance metrics and diagnostics
   - **Scenarios**: Use interactive sliders for what-if analysis

3. **Model Interpretation**
   - Stage 1 shows how social channels drive Google spend
   - Stage 2 reveals complete revenue attribution
   - Feature importance guides channel prioritization
   - Cross-validation results indicate model stability

---

## 📋 Technical Specifications

### Dependencies

#### Marketing Dashboard
```txt
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
numpy>=1.24.0
scipy>=1.10.0
```

#### MMM Assignment  
```txt
streamlit>=1.29.0
pandas>=2.1.0
numpy>=1.25.0
scikit-learn>=1.3.0
plotly>=5.15.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Performance Optimization
- **Data Caching**: Streamlit caching for improved load times
- **Memory Management**: Efficient pandas operations
- **Responsive Design**: Mobile-friendly dashboard layouts
- **Error Handling**: Robust data validation and graceful failures

---

## 🎯 Business Value & Applications

### Marketing Dashboard Applications
- **Executive Reporting**: High-level KPI tracking for leadership
- **Campaign Optimization**: Identify top-performing tactics and campaigns  
- **Budget Allocation**: Data-driven channel investment decisions
- **Performance Monitoring**: Real-time marketing effectiveness tracking

### MMM Model Applications
- **Channel Attribution**: Understand true impact of each marketing channel
- **Budget Optimization**: Optimize spend allocation across channels
- **Scenario Planning**: Test marketing strategies before implementation
- **Strategic Planning**: Long-term marketing investment guidance

### Key Business Insights

#### Dashboard Insights
- Google delivers highest ROAS at 3.06x
- Facebook shows strong performance at 2.61x ROAS  
- TikTok provides growth opportunities for optimization
- Customer acquisition cost optimized at $1.25

#### MMM Insights  
- Social media channels significantly drive search behavior
- Mediation effects explain 93.2% of Google spend variation
- Direct response channels (email/SMS) provide complementary revenue
- Price and promotion sensitivity clearly quantified

---

## 🔍 Model Limitations & Considerations

### Statistical Limitations
- **Cross-Validation**: Negative CV R² indicates potential overfitting
- **Generalization**: Time series patterns may not extrapolate well
- **Assumptions**: Mediation assumption may not capture all brand effects
- **External Factors**: Economic conditions and competition not modeled

### Recommended Best Practices
- **Regular Retraining**: Monthly model updates with new data
- **A/B Testing**: Validate model recommendations in market
- **Gradual Implementation**: Limit budget changes to 20% monthly
- **Performance Monitoring**: Track prediction accuracy on new data

---

## 🛠️ Development & Customization

### Adding New Channels
1. Include new CSV file with standard schema
2. Update data loading functions
3. Add channel to filtering options
4. Update visualization configurations

### Custom Metrics
1. Add metric calculations in data processing
2. Update KPI card displays  
3. Create corresponding visualizations
4. Integrate into automated insights

### Styling Modifications
1. Update CSS in Streamlit markdown sections
2. Modify Plotly chart color schemes
3. Adjust layout using column configurations
4. Update dashboard themes and branding

---

## 🚀 Deployment Options

### Local Development
```bash
# Marketing Dashboard
streamlit run marketing-dashboard/app.py

# MMM Assignment  
streamlit run MMM-Assignment/app.py
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration with free hosting
- **Heroku**: Container-based deployment with custom domains
- **AWS/GCP**: Scalable cloud hosting with enterprise features

### Production Considerations
- Environment variable management for sensitive data
- SSL certificates for secure data transmission
- User authentication for enterprise deployment
- Database integration for real-time data updates

---

## 📞 Support & Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Clean installation
pip uninstall -y streamlit xgboost plotly pandas scikit-learn
pip install -r requirements.txt --no-cache-dir

# For Apple M1 users
brew install libomp
```

#### Data Loading Issues
- Ensure CSV files match expected schema exactly
- Check date formats (YYYY-MM-DD recommended)
- Verify no excessive missing values in key columns
- Confirm numeric columns contain valid numbers

#### Performance Issues  
- Ensure minimum 4GB RAM available
- Clear Streamlit cache: `streamlit cache clear`
- Restart dashboard: Ctrl+C and re-run streamlit command
- Check for large file sizes causing memory issues

### Model Interpretation Guidance

#### Understanding Results
- R² values above 0.80 indicate strong predictive power
- RMSE should be evaluated relative to target variable means
- Cross-validation below training suggests overfitting
- Feature importance rankings guide channel prioritization

#### Business Translation
- Focus on directional insights rather than precise predictions
- Validate model recommendations against business intuition
- Regular performance monitoring essential for ongoing accuracy
- Consider external factors when interpreting results

---

## 🏆 Project Highlights

### Technical Achievements
- **Advanced Analytics**: Two-stage causal modeling with proper attribution
- **Production-Ready**: Professional dashboards with comprehensive features
- **Robust Validation**: Time series cross-validation with performance monitoring  
- **Extensible Design**: Modular architecture supporting future enhancements

### Business Impact
- **Strategic Insights**: Data-driven channel prioritization and optimization
- **Operational Tools**: Interactive scenario planning for marketing teams
- **Decision Support**: ROI-based recommendations with risk assessment
- **Scalable Framework**: Methodology applicable across marketing contexts

### Code Quality
- **Comprehensive Documentation**: Detailed setup and usage instructions
- **Reproducible Results**: Fixed seeds and versioned dependencies
- **Professional UI**: Modern dashboard design with intuitive navigation
- **Error Handling**: Robust validation and graceful failure modes

---

## 📄 License & Acknowledgments

**License**: MIT License - see LICENSE file for details

**Built With**:
- **Streamlit** - Interactive web application framework
- **XGBoost** - Gradient boosting machine learning library  
- **Plotly** - Interactive visualization library
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning utilities

**Assessment Context**: These projects were developed as part of marketing analytics assessments, demonstrating advanced capabilities in business intelligence, causal modeling, and interactive dashboard development.

---

**🚀 Ready for immediate deployment and business impact!**

For questions, issues, or contributions, please refer to the individual project README files or create an issue in this repository.