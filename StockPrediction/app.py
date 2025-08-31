from flask import Flask,request,render_template
import util
import asset
import os

app = Flask(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        stock = request.form['stock']
        symbol=stock.upper()+".NS"
        long_df = util.dataset(symbol)
        results_stock,img_base64 = util.run_stock_forecast(long_df, symbol , model_type='rf')
        if 'pdf_file' not in request.files:
            return "No file part"
        file = request.files['pdf_file']
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output = asset.analyze_pdf(filepath)
        if output['SentimentScore']=="Positive" and results_stock['trend']=="Expected Uptrend":
            recommendation = "Buy"
        elif (output['SentimentScore']=="Negative" and results_stock['trend']=="Expected Downtrend"):
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        return render_template("chart.html", chart_data=img_base64,results_stock=results_stock,output=output,recommendation=recommendation)
        
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)



""" 
Desclaimer : In future this project can be improved more by changing model etc...
"""