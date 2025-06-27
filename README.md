# Latency-Aware HFT Arbitrage with Deep Reinforcement Learning

Projet de recherche explorant l'utilisation du Deep Reinforcement Learning pour concevoir une stratégie d'arbitrage statistique haute fréquence (HFT), en tenant compte de la latence d'exécution dans un environnement de marché simulé.

---

## Objectifs

- Simuler un environnement de marché réaliste avec latence, spread, et carnet d'ordres simplifié
- Entraîner un agent PPO via `Stable-Baselines3` pour détecter des opportunités d’arbitrage
- Comparer les performances à des stratégies naïves (Random, Buy & Hold)
- Visualiser la reward, l’évolution du P&L et les décisions de l’agent
- Préparer le terrain pour un passage à des données réelles et des agents plus complexes

---

## Structure du projet

```
hft-rl-latency-aware-arbitrage/
├── src/                    # Code source
│   ├── environment.py      # Gymnasium.Env simulé avec latence
│   ├── trainer.py          # Entraînement PPO
│   ├── evaluate.py         # Évaluation et comparaison
│   ├── baselines.py        # Stratégies naïves
│   └── models/             # (optionnel) agents custom
├── notebooks/              # Explorations et visualisations
│   ├── ppo_training.ipynb
│   └── evaluate_agent.ipynb
├── results/                # Checkpoints et figures
├── tests/                  # Tests unitaires (env)
├── README.md
├── requirements.txt
└── setup.py
```

---

## Lancement rapide

1. **Installer les dépendances** :

```bash
pip install -r requirements.txt
```

2. **Entraîner un agent PPO** :

```bash
python -m src.trainer
```

3. **Évaluer et comparer aux baselines** :

```bash
python -m src.evaluate
```

---

## Exemple de sortie

- Reward cumulée sur 500 steps
- P&L final (cash + inventaire × prix)
- Comparaison PPO vs Random vs Buy & Hold

*Figure générée automatiquement dans `evaluate.py`.*

---

## Améliorations prévues

- [ ] Ajout de données réelles (Binance, LOBSTER, etc.)
- [ ] Environnement multi-actifs ou multi-niveaux
- [ ] Agent avec mémoire (LSTM, Transformer)
- [ ] Logging avancé (TensorBoard, MLflow)
- [ ] Backtest + métriques réalistes (Sharpe, Drawdown)

---

## Auteur

Projet initié par **Elliot Piet**, dans le cadre d'une recherche appliquée en trading algorithmique.

---

## Licence

MIT License
