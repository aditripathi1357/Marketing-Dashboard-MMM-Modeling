# Marketing Intelligence Dashboard

A comprehensive interactive BI dashboard that transforms marketing data into actionable insights, helping business stakeholders understand how marketing activities connect with business outcomes.



## 📋 Project Overview

This project is an **Assessment for Marketing Intelligence Dashboard/Reporting** that analyzes 120 days of daily marketing and business activity across multiple channels (Facebook, Google, TikTok) and provides actionable insights through an interactive web dashboard.

### Context
The dashboard processes four key datasets:
- **Facebook.csv, Google.csv, TikTok.csv**: Campaign-level marketing data including impressions, clicks, spend, and attributed revenue
- **Business.csv**: Daily business performance data including orders, revenue, profit, and customer metrics

## 🚀 Features

### Key Performance Indicators
- **Total Marketing Spend**: $5.26M across all channels
- **Attributed Revenue**: $14.77M with clear ROI tracking
- **Total Profit**: $517.27M business performance
- **Customer Acquisition**: $1.25 average CAC
- **ROAS Performance**: 2.82x average return on ad spend



### Interactive Analytics
- **Channel Performance Comparison**: Real-time ROAS analysis across Facebook, Google, and TikTok
- **Daily Spend vs Revenue Trends**: Time-series visualization with trend lines
- **Orders & Profit Performance**: Business outcome tracking
- **Marketing Spend Distribution**: Channel allocation insights



### Advanced Features
- **Multi-level Filtering**: Channel selection and date range filtering
- **Trend Analysis**: Automated trend line generation with statistical insights
- **Data Export**: CSV download functionality for further analysis
- **Search & Filter**: Campaign-level search capabilities
- **Automated Insights**: AI-generated performance insights



## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```txt
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
numpy>=1.24.0
scipy>=1.10.0
```

### Installation Steps

1. **Clone the repository**
```bash
https://github.com/aditripathi1357/Marketing-Dashboard-MMM-Modeling.git
cd marketing-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare data files**
Ensure the following CSV files are in the project directory:
- `Facebook.csv`
- `Google.csv` 
- `TikTok.csv`
- `Business.csv`

4. **Run the application**
```bash
streamlit run app.py
```

### Expected Output
```
PS D:\LifesightProject\marketing-dashboard> streamlit run app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
  Network URL: http://192.168.29.104:8502
```

## 📊 Dashboard Sections

### 1. Key Performance Indicators
- Real-time KPI cards with animated hover effects
- Cross-channel performance summaries
- Essential metrics at a glance

### 2. Performance Analytics
- **Daily Spend vs Revenue**: Dual-axis time series with trend analysis
- **ROAS by Channel**: Comparative performance visualization
- **Orders & Profit Performance**: Business outcome tracking
- **Marketing Spend Distribution**: Channel allocation pie chart

### 3. Data Analytics & Insights
- **Summary Statistics**: Detailed channel performance metrics
- **Performance Metrics**: Comprehensive KPI breakdowns
- **Raw Data**: Searchable and filterable campaign data
- **Automated Insights**: AI-generated performance analysis



## 🎯 Key Insights Delivered

### Performance Insights
- **Best Performing Channel**: Google with 3.06x ROAS
- **Growth Opportunities**: Facebook showing 2.61x ROAS with optimization potential
- **Overall ROI**: 181% return on marketing investment
- **Efficiency Analysis**: 17% performance gap between channels

### Business Intelligence
- **Customer Acquisition Cost**: Optimized at $1.25 across channels
- **Revenue Attribution**: Clear tracking of marketing contribution
- **Profitability Analysis**: Margin tracking and optimization opportunities
- **Channel Optimization**: Data-driven budget allocation recommendations

## 🏗 Technical Architecture

### Data Processing
- **ETL Pipeline**: Automated data loading and transformation
- **Data Validation**: Robust error handling and data quality checks
- **Metric Derivation**: Calculated KPIs (CTR, ROAS, CAC, Profit Margin)
- **Statistical Analysis**: Trend analysis using scipy.stats

### Visualization Technology
- **Plotly Interactive Charts**: Advanced charting with hover interactions
- **Responsive Design**: Mobile-friendly dashboard layout
- **Real-time Filtering**: Dynamic data updates based on user selections
- **Modern UI/UX**: Glassmorphism design with smooth animations

### Performance Optimization
- **Data Caching**: Streamlit caching for improved load times
- **Efficient Joins**: Optimized pandas operations
- **Memory Management**: Proper data type handling and cleanup

## 📈 Product Thinking & Business Value

### Decision-Maker Focus
- **Executive Summary**: High-level KPIs for quick decision making
- **Operational Insights**: Campaign-level data for tactical adjustments
- **Strategic Planning**: Historical trends for budget allocation
- **Performance Monitoring**: Real-time tracking of marketing effectiveness

### Actionable Intelligence
- **Budget Optimization**: Data-driven channel allocation recommendations
- **Campaign Performance**: Identify top-performing campaigns and tactics
- **ROI Maximization**: Clear visibility into return on marketing investment
- **Growth Opportunities**: Highlight underperforming areas with potential

## 🎨 Design Principles

### User Experience
- **Intuitive Navigation**: Clear section organization and filtering
- **Progressive Disclosure**: Information hierarchy from summary to detail
- **Interactive Exploration**: Drill-down capabilities and dynamic filtering
- **Professional Aesthetics**: Modern design with consistent branding

### Data Visualization
- **Chart Selection**: Appropriate visualizations for different data types
- **Color Consistency**: Cohesive color scheme across all charts
- **Clarity Focus**: Minimal clutter with maximum information density
- **Accessibility**: High contrast and readable typography

## 🔧 Customization

### Adding New Channels
1. Add new CSV file with matching schema
2. Update the data loading function in `load_and_process_data()`
3. Include channel in the filtering dropdown

### Custom Metrics
1. Add metric calculation in the derived metrics section
2. Update KPI cards with new metrics
3. Create corresponding visualizations

### Styling Changes
1. Modify CSS in the `st.markdown()` sections
2. Update color schemes in Plotly chart configurations
3. Adjust layout using Streamlit columns

## 📝 Data Schema

### Marketing Data (Facebook.csv, Google.csv, TikTok.csv)
```
- date: Date of campaign activity
- tactic: Marketing tactic used
- state: Geographic state
- campaign: Campaign identifier
- impression: Number of impressions
- clicks: Number of clicks
- spend: Amount spent
- attributed revenue: Revenue attributed to campaign
```

### Business Data (Business.csv)
```
- date: Date of business activity
- # of orders: Total daily orders
- # of new orders: New orders count
- new customers: New customer acquisitions
- total revenue: Total daily revenue
- gross profit: Daily gross profit
- COGS: Cost of goods sold
```

## 🚀 Deployment Options

### Local Development
- Run with `streamlit run app.py`
- Access via `localhost:8502`

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container-based deployment
- **AWS/GCP**: Scalable cloud hosting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for rapid dashboard development
- Plotly for interactive visualizations
- Pandas for data manipulation and analysis
- Designed with modern UI/UX principles for optimal user experience

---

**Dashboard Live Demo**: [View Dashboard](http://localhost:8502)

**Project Status**: ✅ Production Ready

**Last Updated**: September 2025