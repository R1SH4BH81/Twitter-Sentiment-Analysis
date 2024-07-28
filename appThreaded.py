import csv
import time
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Define the sentiment labels
labels = ['Negative', 'Neutral', 'Positive']

# Flag to control the status message
analyzing = False

def status_indicator():
    """
    A simple status indicator that runs in a separate thread.
    It prints "Analyzing" with a dot animation while the analysis is ongoing.
    """
    while analyzing:
        print("Analyzing", end="")
        for _ in range(3):
            if not analyzing:
                break
            print(".", end="", flush=True)
            time.sleep(0.5)
        print("\r", end="", flush=True)

def preprocess_tweet(tweet):
    """
    Preprocess the tweet by replacing user mentions and URLs.
    
    Args:
        tweet (str): The tweet text to preprocess.

    Returns:
        str: The preprocessed tweet text.
    """
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    return ' '.join(tweet_words)

def analyze_sentiment(tweet):
    """
    Perform sentiment analysis on the given tweet text.
    
    Args:
        tweet (str): The tweet text to analyze.

    Returns:
        dict: A dictionary with sentiment labels and their respective scores.
    """
    global analyzing
    analyzing = True
    # Start the status indicator in a separate thread
    status_thread = threading.Thread(target=status_indicator)
    status_thread.start()

    # Tokenize and perform sentiment analysis
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)

    # Compute scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Map scores to sentiment labels
    sentiment = {labels[i]: scores[i] for i in range(len(scores))}

    # Stop the status indicator
    analyzing = False
    status_thread.join()

    return sentiment

def process_csv(file_path):
    """
    Process a CSV file containing tweets and perform sentiment analysis on each tweet.
    
    Args:
        file_path (str): The path to the CSV file.
    """
    # Open the CSV file
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Read the header (skip if necessary)
        header = next(csvreader)
        
        # Iterate through each row in the CSV
        for row in csvreader:
            # Assuming tweet content is in the first column of the CSV
            tweet_content = row[0]
            
            # Preprocess and analyze the tweet
            processed_tweet = preprocess_tweet(tweet_content)
            sentiment = analyze_sentiment(processed_tweet)

            # Output results
            print(f"Tweet: {tweet_content}")
            for label, score in sentiment.items():
                print(f"{label}: {score:.3f}")
            print("\n")

# Example usage
file_path = 'tweets.csv'  # Path to your CSV file
process_csv(file_path)
