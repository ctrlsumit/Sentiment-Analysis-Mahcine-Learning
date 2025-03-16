from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords if not found
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Now get the stopwords
stopwords_set = set(stopwords.words('english'))

# Fix invalid escape sequences in regex patterns
emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    # Fix invalid escape sequence in regex
    text = re.sub(r'[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
