import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Step 0: Download required NLTK data (only needs to be run once)
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Input Sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

# Step 2: Tokenization
tokens = word_tokenize(sentence)
print("1. Original Tokens:", tokens)

# Step 3: Remove Stopwords
stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens if word.lower() not in stop_words]
print("2. Tokens Without Stopwords:", tokens_no_stopwords)

# Step 4: Apply Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stopwords]
print("3. Stemmed Words:", stemmed_tokens)
