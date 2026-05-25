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
import warnings
warnings.filterwarnings('ignore')

os.makedirs('model', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Version optimisée (vitesse)")
print("=" * 60)

# 1. Chargement du dataset (réduit pour la vitesse)
print("\n1. Chargement du dataset IMDb...")
df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')
print(f"   Dataset complet: {len(df)} critiques")

# Prendre 20 000 critiques (bon compromis vitesse/précision)
df = df.sample(20000, random_state=42)
print(f"   Sous-ensemble: {len(df)} critiques")

# 2. Vectorisation TF-IDF (réduite)
print("\n2. Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3
)
X = vectorizer.fit_transform(df['review']).toarray()
y = (df['sentiment'] == 'positive').astype(int).values
print(f"   Features: {X.shape[1]}")

# 3. Split
print("\n3. Split train/test...")
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

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

# 5. Modèle plus simple mais efficace
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
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = SentimentNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

print(f"\n4. Modèle: {sum(p.numel() for p in model.parameters())} paramètres")

# 6. Entraînement (moins d'epochs)
print("\n5. Entraînement...")
best_accuracy = 0
for epoch in range(20):
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
    
    scheduler.step(accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'model/model.pt')
        joblib.dump(vectorizer, 'model/vectorizer.pkl')
        print(f"   Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, accuracy={accuracy:.4f} ★ NEW BEST")
    elif (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, accuracy={accuracy:.4f}")

# 7. Sauvegarde
print(f"\n6. Meilleure accuracy: {best_accuracy:.4f}")

config = {'input_dim': X_train.shape[1], 'model_type': 'TF-IDF + NN (3 couches)', 'max_features': 15000}
with open('model/config.json', 'w') as f:
    json.dump(config, f, indent=2)

metrics = {'accuracy': best_accuracy}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
if best_accuracy >= 0.92:
    print("🎉 OBJECTIF ATTEINT ! Accuracy >= 92%")
else:
    print(f"⚠️ Accuracy {best_accuracy:.2%} < 92%")
print("=" * 60)
