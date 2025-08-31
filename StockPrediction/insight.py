import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("API_KEY")

def summarize_text(text: str) -> str:
    """Summarize input text using gpt-4o-mini"""
    client = OpenAI(
        api_key=key
    )
    response = client.chat.completions.create(
          model="gpt-4o-mini",
          store=True,
          messages=[
            {"role": "user", "content": "write a haiku about ai"}
          ]
    )
    return response.choices[0].message.content