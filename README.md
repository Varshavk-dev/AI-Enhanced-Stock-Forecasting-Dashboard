<img src="images/first dashboard.png" alt="Dashboard" width="500">


# AI - ENHANCED - STOCK - FORCASTING - DASHBOARD

This project is a **Stock Prediction Dashboard** that provides insights into future stock prices using **Machine Learning forecasting** and **AI-based Sentiment Analysis**. The system combines **technical data forecasting** with **sentiment analysis of financial reports** to generate actionable recommendations: **Buy, Sell, or Hold**.


##  Technology 

- **Backend:** Python, Flask  
- **Data Processing:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib  
- **Data Source:** yFinance (Yahoo Finance API)  
- **Development Tools:** Jupyter Notebook, VS Code

---

## consist of 2 sections

### 1. ML-Based Forecasting  
  Workflow
  - Get stock Close prices (yfinance).
  - Add features → lag(1), lag(7), rolling mean(7), day_of_week.
  - Clean data.
  - Train RandomForestRegressor.
  - Predict next 7 days.
  - Show RMSE + forecast chart + trend.

 ### 1. ML-Based Forecasting   


  

## ▶️ How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/stock-prediction-dashboard.git
   cd stock-prediction-dashboard


