import os
import re
import string
import joblib
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load stemmer, stopwords, and trained artifacts
stemmer = PorterStemmer()
stopwords_set = set(ENGLISH_STOP_WORDS)

print("Loading trained models and vectorizer...")
tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
nb_model = joblib.load(os.path.join(MODELS_DIR, 'multinomial_nb_model.joblib'))
svm_model = joblib.load(os.path.join(MODELS_DIR, 'linear_svm_model.joblib'))
print("Models successfully loaded.\n")

def clean_sms(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' httpaddr ', text)
    text = re.sub(r'\S+@\S+', ' emailaddr ', text)
    text = re.sub(r'[£$€₹]', ' moneysymb ', text)
    text = re.sub(r'\b\d{5,}\b', ' phonenum ', text)
    text = re.sub(r'\b\d+\b', ' numbr ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    cleaned = [stemmer.stem(w) for w in tokens if w not in stopwords_set and len(w) > 1]
    return " ".join(cleaned)

def predict_message(message):
    cleaned = clean_sms(message)
    vec = tfidf.transform([cleaned])
    
    # Naive Bayes prediction & confidence
    nb_pred = nb_model.predict(vec)[0]
    nb_probs = nb_model.predict_proba(vec)[0]
    
    # Linear SVM prediction & decision score
    svm_pred = svm_model.predict(vec)[0]
    svm_score = svm_model.decision_function(vec)[0]
    
    label_map = {0: 'HAM (Legitimate)', 1: 'SPAM (Unsolicited/Fraud)'}
    
    print("-" * 65)
    print(f"INPUT SMS : \"{message}\"")
    print(f"Cleaned   : \"{cleaned}\"")
    print("-" * 65)
    print(f"[Multinomial Naive Bayes]:")
    print(f"  Classification : {label_map[nb_pred]}")
    print(f"  Probability    : Ham: {nb_probs[0]*100:.2f}% | Spam: {nb_probs[1]*100:.2f}%")
    print(f"[Linear Support Vector Machine (Top Model)]:")
    print(f"  Classification : {label_map[svm_pred]}")
    print(f"  Decision Score : {svm_score:+.4f} (>0 denotes Spam)")
    print("-" * 65)

if __name__ == '__main__':
    print("=" * 65)
    print(" SMS SPAM MESSAGE DETECTION: LIVE TEST ENGINE ")
    print("=" * 65)
    
    sample_messages = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click http://claim-prize.com to redeem now!",
        "URGENT! Your mobile number has won £5000 prize bonus. Call 09066362206 to claim your cash award.",
        "Hey Ayush, are we still meeting in the computer lab at 4:30 PM for our ML project preparation?",
        "Can you please send me the class lecture notes from today morning? Thanks!",
        "FREE entry into our weekly £1000 draw! Text WIN to 80085 right now."
    ]
    
    for msg in sample_messages:
        predict_message(msg)
        
    print("\nYou can also test your own message by calling predict_message(\"your text here\").")
