\# IMDb Sentiment Analysis



Analyse de sentiment sur les critiques de films IMDb avec un réseau de neurones.



\## Modèle



\- Vectorisation : TF-IDF (5000 features)

\- Architecture : 128 → 64 → 1 (ReLU, Dropout, Sigmoid)

\- Framework : PyTorch



\## Performance



Accuracy : > 85% sur l'ensemble de test



\## Fichiers



\- `train.py` : entraînement du modèle

\- `predict.py` : prédiction sur de nouvelles critiques

\- `model/model.pt` : poids du modèle

\- `model/vectorizer.pkl` : vectoriseur TF-IDF

\- `model/config.json` : configuration

\- `model/metrics.json` : métriques de performance



\## Utilisation



```python

from predict import predict\_sentiment



sentiment, confidence = predict\_sentiment("This movie is great!")

print(sentiment)  # POSITIF

