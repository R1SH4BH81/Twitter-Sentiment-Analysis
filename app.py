import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

def preprocess_tweet(tweet):
    # Preprocess the tweet by replacing user mentions and URLs
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    return ' '.join(tweet_words)

def analyze_sentiment(tweet):
    # Perform sentiment analysis
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = {labels[i]: scores[i] for i in range(len(scores))}
    return sentiment

def process_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip header if there is one

        for row in csvreader:
            # Assuming tweet content is in the first column of the CSV
            tweet_content = row[0]
            processed_tweet = preprocess_tweet(tweet_content)
            sentiment = analyze_sentiment(processed_tweet)

            print(f"Tweet: {tweet_content}")
            for label, score in sentiment.items():
                print(f"{label}: {score:.3f}")
            print("\n")

# Example usage
file_path = 'tweets.csv'  # Path to your CSV file
process_csv(file_path)
