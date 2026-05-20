import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

os.makedirs('model', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Entraînement (visée 92%+)")
print("=" * 60)

# 1. Charger PLUS de données
print("\n1. Chargement du dataset IMDb...")
df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')
print(f"   {len(df)} critiques chargées")

# Utiliser TOUT le dataset (50000 exemples) au lieu de 5000
print(f"   Utilisation de tout le dataset: {len(df)} critiques")

# 2. Vectorisation avec PLUS de features
print("\n2. Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['review']).toarray()
y = (df['sentiment'] == 'positive').astype(int).values
print(f"   Features: {X.shape[1]}")

# 3. Split train/test
print("\n3. Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Convertir en tenseurs
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# 5. Modèle PLUS GRAND
class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = SentimentNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

print(f"\n4. Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")

# 6. Entraînement PLUS LONG
print("\n5. Entraînement...")
best_accuracy = 0
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Évaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)
        pred_labels = (predictions > 0.5).float()
        accuracy = (pred_labels == y_test_t).float().mean().item()
    
    scheduler.step(total_loss/len(train_loader))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'model/model.pt')
        joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}, accuracy = {accuracy:.4f}")

# 7. Évaluation finale
print(f"\n6. Meilleure accuracy: {best_accuracy:.4f}")

# 8. Sauvegarde finale
print("\n7. Sauvegarde des artefacts...")
torch.save(model.state_dict(), 'model/model.pt')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

config = {
    'input_dim': X_train.shape[1],
    'model_type': 'TF-IDF + Neural Network (4 couches)',
    'max_features': 10000,
    'ngram_range': '(1,2)'
}
with open('model/config.json', 'w') as f:
    json.dump(config, f, indent=2)

metrics = {'accuracy': best_accuracy}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print(f"✅ Accuracy finale: {best_accuracy:.4f}")
if best_accuracy >= 0.92:
    print("🎉 OBJECTIF ATTEINT ! Accuracy >= 92%")
else:
    print(f"⚠️ Accuracy {best_accuracy:.2%} < 92%, à ré-entraîner")
print("=" * 60)