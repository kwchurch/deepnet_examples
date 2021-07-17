# Sentiment Analysis (for Chinese texts)

Super-simple example of pre-trained sentiment model from <a
href="https://www.paddlepaddle.org.cn/hubdetail?name=senta_bilstm&en_category=SentimentAnalysis">PaddleHub</a>.
See <a href="https://www.paddlepaddle.org.cn/hublist">PaddleHub
home</a> for many more pre-trained models.


```shell 
  pip install -r requirements.txt
  python paddle_sentiment.py < sample_Chinese_input.txt
  python huggingface_sentiment.py < sample_English_input.txt
```
Outputs sentiment label and probability for each input line.
