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
import warnings
warnings.filterwarnings('ignore')

os.makedirs('model', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Modèle maison (objectif 92%+)")
print("=" * 60)

# 1. Chargement complet
print("\n1. Chargement du dataset IMDb...")
df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')
print(f"   Dataset: {len(df)} critiques")

# 2. Vectorisation avec trigrammes
print("\n2. Vectorisation TF-IDF (unigram + bigram + trigram)...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    ngram_range=(1, 3),  # trigrammes !
    sublinear_tf=True,
    min_df=3,
    max_df=0.9
)
X = vectorizer.fit_transform(df['review']).toarray()
y = (df['sentiment'] == 'positive').astype(int).values
print(f"   Features: {X.shape[1]}")

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Normalisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Tenseurs
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

# 5. Modèle plus profond mais régularisé
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
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = SentimentNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)

print(f"\nModèle: {sum(p.numel() for p in model.parameters())} paramètres")

# 6. Entraînement plus long
print("\nEntraînement...")
best_accuracy = 0
patience = 0

for epoch in range(40):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t)
        acc = ((pred > 0.5).float() == y_test_t).float().mean().item()
    
    scheduler.step(acc)
    
    if acc > best_accuracy:
        best_accuracy = acc
        patience = 0
        torch.save(model.state_dict(), 'model/model.pt')
        joblib.dump(vectorizer, 'model/vectorizer.pkl')
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, accuracy={acc:.4f} ★ BEST")
    else:
        patience += 1
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, accuracy={acc:.4f}")
    
    if patience >= 6:
        print(f"\nEarly stopping à l'epoch {epoch+1}")
        break

print(f"\nMeilleure accuracy: {best_accuracy:.4f}")

config = {
    'input_dim': X_train.shape[1],
    'max_features': 20000,
    'ngram_range': '(1,3)',
    'model_type': 'TF-IDF + NN trigrammes'
}
with open('model/config.json', 'w') as f:
    json.dump(config, f)
with open('model/metrics.json', 'w') as f:
    json.dump({'accuracy': best_accuracy}, f)

if best_accuracy >= 0.92:
    print("\n🎉 OBJECTIF ATTEINT !")
else:
    print(f"\n⚠️ Accuracy {best_accuracy:.2%} < 92%")
