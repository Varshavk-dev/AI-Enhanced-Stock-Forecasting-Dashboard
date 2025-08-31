from PyPDF2 import PdfReader
from textblob import TextBlob
import insight



def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def get_sentiment_score(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"
    


def detect_trend_bias(text, sentiment):
    keywords = ["growth", "increase", "expansion", "positive outlook", "strong performance"]
    if sentiment == "Positive" and any(word in text.lower() for word in keywords):
        return "Likely Positive"
    elif sentiment == "Negative":
        return "Likely Negative"
    else:
        return "Unclear"
    

def analyze_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    sentiment = get_sentiment_score(text)
    summary = insight.summarize_text(text)  
    trend = detect_trend_bias(text, sentiment)

    result = {
        "SentimentScore": sentiment,
        "Summary": summary,
        "TrendBias": trend
    }

    return result