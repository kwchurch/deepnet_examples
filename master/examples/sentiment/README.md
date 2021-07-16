# Sentiment Analysis (for Chinese texts)

Super-simple example of pre-trained sentiment model from <a
href="https://www.paddlepaddle.org.cn/hubdetail?name=senta_bilstm&en_category=SentimentAnalysis">PaddleHub</a>.
See <a href="https://www.paddlepaddle.org.cn/hublist">PaddleHub
home</a> for many more pre-trained models.


Install:
   ```pip install -r requirements.txt```

Example of Usage:
   ```python sentiment.py < sample_input.txt```

Input: Chinese text such as <a href="sample_input.txt">this</a>.

Outputs sentiment label and probability for each input line.
