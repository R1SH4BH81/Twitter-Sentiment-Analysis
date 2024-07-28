from flask import Flask, render_template, request, redirect, url_for
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import os

app = Flask(__name__)

# Load the pre-trained model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    return ' '.join(tweet_words)

def analyze_sentiment(tweet):
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = {labels[i]: scores[i] for i in range(len(scores))}
    return sentiment

def process_csv(file_path):
    tweet_data = []
    sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip header if there is one

        for row in csvreader:
            tweet_content = row[0]
            processed_tweet = preprocess_tweet(tweet_content)
            sentiment = analyze_sentiment(processed_tweet)

            # Count each sentiment
            max_sentiment = max(sentiment, key=sentiment.get)
            sentiment_counts[max_sentiment] += 1

            tweet_data.append({'tweet': tweet_content, 'sentiment': sentiment})
    
    # Save the results to a new CSV file in the static directory
    results_file_path = os.path.join('static', 'sentiment_analysis_results.csv')
    with open(results_file_path, mode='w', newline='', encoding='utf-8') as results_file:
        csv_writer = csv.writer(results_file)
        csv_writer.writerow(['Tweet', 'Negative', 'Neutral', 'Positive'])
        
        for data in tweet_data:
            tweet = data['tweet']
            negative = data['sentiment']['Negative']
            neutral = data['sentiment']['Neutral']
            positive = data['sentiment']['Positive']
            csv_writer.writerow([tweet, negative, neutral, positive])
    
    return tweet_data, sentiment_counts, 'sentiment_analysis_results.csv'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'tweets.csv'
            file.save(file_path)
            return redirect(url_for('results'))
    return render_template('index.html')

@app.route('/results', methods=['GET'])
def results():
    tweet_data, sentiment_counts, results_file_name = process_csv('tweets.csv')
    return render_template('results.html', tweet_data=tweet_data, sentiment_counts=sentiment_counts, results_file_name=results_file_name)

if __name__ == '__main__':
    app.run(debug=True)
