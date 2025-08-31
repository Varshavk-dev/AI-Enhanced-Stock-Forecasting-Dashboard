# AI - ENHANCED - STOCK - FORCASTING - DASHBOARD
---

# Dashboard to get user input
<img src="image/first dashboard.png" alt="Dashboard" width="500">

# Output screen
<img src="image/output screen.png" alt="Dashboard" width="500">

This project is a **Stock Prediction Dashboard** that provides insights into future stock prices using **Machine Learning forecasting** and **AI-based Sentiment Analysis**. The system combines **technical data forecasting** with **sentiment analysis of financial reports** to generate actionable recommendations: **Buy, Sell, or Hold**.


##  Technology 

- **Backend:** Python, Flask  
- **Data Processing:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib  
- **Data Source:** yFinance (Yahoo Finance API)  
- **Development Tools:** Jupyter Notebook, VS Code
- **libraries:** PDFreader, TextBlob, OpenAI 

---

## consist of 2 sections

### 1.Stock price ML-Based Forecasting  
  Workflow
  - Get stock Close prices (yfinance).
  - Add features → lag(1), lag(7), rolling mean(7), day_of_week.
  - Clean data.
  - Train RandomForestRegressor.
  - Predict next 7 days.
  - Show RMSE + forecast chart + trend.

### 2. PDF Report Analysis of AI Prediction 
 Workflow
 - extract text from pdf using PDFreader
 - get sentiment score using TextBlob
 - summarize the text using OpenAI
 - Detect trend bias

---

### Setup Steps

- create .env file and provide the API_KEY
- install requirements.txt file
- run app.py file
- input : stock symbol and pdf ( sample pdf is there in uploads folder - StockPrediction\uploads)
- output : recomendation (Buy/Sell/Hold) , stockname , next 7 day stock price , Mean Square error , RMSE , Line Chart of stock price , Sentiment Score , Summary , Trend Bias

  

## How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/stock-prediction-dashboard.git
   cd stock-prediction-dashboard


