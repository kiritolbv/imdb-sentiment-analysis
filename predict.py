import json
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Définir la même architecture que dans train.py
class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Charger la config
with open('model/config.json', 'r') as f:
    config = json.load(f)

# Charger le vectorizer
vectorizer = joblib.load('model/vectorizer.pkl')

# Charger le modèle
model = SentimentNN(config['input_dim'])
model.load_state_dict(torch.load('model/model.pt', map_location=torch.device('cpu')))
model.eval()

def predict_sentiment(review):
    """Prédit le sentiment d'une critique (positif/négatif)"""
    # Vectorisation
    X = vectorizer.transform([review]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Prédiction
    with torch.no_grad():
        prediction = model(X_tensor).item()
    
    sentiment = "POSITIF" if prediction > 0.5 else "NEGATIF"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence

# Test
if __name__ == "__main__":
    print("=" * 50)
    print("IMDb Sentiment Analysis - Prédiction")
    print("=" * 50)
    
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute.",
        "Terrible film, waste of time, boring and stupid.",
        "Pretty good, worth watching, the acting was solid."
    ]
    
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review)
        print(f"\nCritique: {review[:60]}...")
        print(f"Sentiment: {sentiment} (confiance: {confidence:.2%})")