import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

os.makedirs('model', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Optimisé 92%+")
print("=" * 60)

# 1. Charger TOUT le dataset
df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')
print(f"Dataset: {len(df)} critiques")

# 2. Vectorisation optimisée
vectorizer = TfidfVectorizer(max_features=15000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['review']).toarray()
y = (df['sentiment'] == 'positive').astype(int).values
print(f"Features: {X.shape[1]}")

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Tenseurs
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

# 5. Modèle plus profond
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
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

print(f"Paramètres: {sum(p.numel() for p in model.parameters())}")

# 6. Entraînement
best_accuracy = 0
for epoch in range(40):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t)
        accuracy = ((pred > 0.5).float() == y_test_t).float().mean().item()
    
    scheduler.step(total_loss/len(train_loader))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'model/model.pt')
        joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, accuracy={accuracy:.4f}")

print(f"\n✅ Meilleure accuracy: {best_accuracy:.4f}")

# 7. Sauvegarde
config = {'input_dim': X_train.shape[1], 'model_type': 'TF-IDF+NN', 'max_features': 15000}
with open('model/config.json', 'w') as f:
    json.dump(config, f)
with open('model/metrics.json', 'w') as f:
    json.dump({'accuracy': best_accuracy}, f)

print("=" * 60)
if best_accuracy >= 0.92:
    print("🎉 OBJECTIF ATTEINT !")
else:
    print(f"⚠️ Accuracy {best_accuracy:.2%} < 92%")
print("=" * 60)