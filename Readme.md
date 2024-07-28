Sentiment Analysis Flask App
This Flask application analyzes the sentiment of tweets in a CSV file, categorizing them into Negative, Neutral, and Positive sentiments using a pre-trained model from Hugging Face's Transformers library.

Features
Upload a CSV file containing tweets.
Process and analyze tweets for sentiment.
Display sentiment results with counts and detailed scores.
Download the results as a CSV file for further analysis.


Demo


Screenshot of the app interface.

Installation
1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-flask.git
cd sentiment-analysis-flask
```
2. Create a Virtual Environment
Create a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
3. Install Dependencies
Install the required Python packages using requirements.txt:

```bash
pip install -r requirements.txt
```
4. Download Pre-trained Models
The first time you run the application, it will automatically download the pre-trained models used for sentiment analysis. Ensure you have a stable internet connection.

Usage
1. Run the Flask Application
Start the Flask application with:

```bash
flask run
```
By default, the app will be available at http://127.0.0.1:5000/.

2. Upload CSV File
Access the app in your browser.
Upload a CSV file containing tweets (one tweet per line).
3. Analyze Sentiment
The app processes the uploaded CSV file and analyzes each tweet's sentiment.
View the results on the /results page, including:
Sentiment breakdown for each tweet.
Overall sentiment counts.
Download the analysis results as a new CSV file.

Dependencies

Flask==2.3.2: Web framework for building web applications.
transformers==4.33.3: Hugging Face Transformers library for using pre-trained models.
scipy==1.10.1: Scientific computing library, used here for calculating softmax scores.
Ensure your requirements.txt includes:

```bash
Flask==2.3.2
transformers==4.33.3
scipy==1.10.1
gunicorn==20.1.0  # Optional for deployment
```
File Structure
Here's a brief overview of the file structure in your Flask project:

```bash
.
├── app.py                       # Main application file
├── requirements.txt             # Python dependencies
├── templates/
│   ├── index.html               # Main page for uploading CSV files
│   └── results.html             # Results page displaying sentiment analysis
├── static/
│   └── sentiment_analysis_results.csv  # Output CSV file
└── README.md                    # Project documentation
```



Contributing
Contributions are welcome! If you'd like to improve this project, please follow these steps:

Please ensure your code adheres to the project's coding standards and is well-documented.
