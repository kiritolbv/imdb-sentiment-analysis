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

# Créer le dossier model
os.makedirs('model', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Entraînement")
print("=" * 60)

# 1. Charger le dataset
print("\n1. Chargement du dataset IMDb...")
df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')
print(f"   {len(df)} critiques chargées")

# Prendre un sous-ensemble pour accélérer (5000 exemples)
df = df.sample(5000, random_state=42)
print(f"   Sous-ensemble: {len(df)} critiques")

# 2. Vectorisation TF-IDF
print("\n2. Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['review']).toarray()
y = (df['sentiment'] == 'positive').astype(int).values
print(f"   Features: {X.shape[1]}")

# 3. Split train/test
print("\n3. Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Convertir en tenseurs PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 5. Définir le modèle
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

model = SentimentNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n4. Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")

# 6. Entraînement
print("\n5. Entraînement...")
for epoch in range(10):
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
    
    print(f"   Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}, accuracy = {accuracy:.4f}")

# 7. Évaluation finale
model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    pred_labels = (predictions > 0.5).float()
    final_accuracy = (pred_labels == y_test_t).float().mean().item()

print(f"\n6. Accuracy finale: {final_accuracy:.4f}")

# 8. Sauvegarde des artefacts
print("\n7. Sauvegarde des artefacts...")

# Sauvegarde du modèle PyTorch (model.pt)
torch.save(model.state_dict(), 'model/model.pt')
print("   ✓ model.pt sauvegardé")

# Sauvegarde du vectorizer
joblib.dump(vectorizer, 'model/vectorizer.pkl')
print("   ✓ vectorizer.pkl sauvegardé")

# Sauvegarde de la config
config = {
    'input_dim': X_train.shape[1],
    'model_type': 'TF-IDF + Neural Network',
    'max_features': 5000
}
with open('model/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("   ✓ config.json sauvegardé")

# Sauvegarde des metrics
metrics = {'accuracy': final_accuracy}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("   ✓ metrics.json sauvegardé")

print("\n" + "=" * 60)
print(f"✅ Entraînement terminé! Accuracy: {final_accuracy:.4f}")
print("=" * 60)