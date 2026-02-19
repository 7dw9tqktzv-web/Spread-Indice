# Spread Trading Indicator — Project CLAUDE.md

## Project Overview
Système complet de spread trading sur futures d'indices US (NQ, ES, RTY, YM).
- **Phase 1** : Moteur de backtest Python (optimisation & validation)
- **Phase 2** : Indicateur Sierra Charts temps réel (ACSIL C++)

Le biais directionnel journalier est discrétionnaire — le système time l'entrée avec précision statistique.

## Univers de Trading
- **Instruments** : NQ, ES, RTY, YM (contrats continus, Volume Rollover Back-Adjusted)
- **6 paires** : NQ/ES, NQ/RTY, NQ/YM, ES/RTY, ES/YM, RTY/YM

## Données
- Source : Sierra Charts / Denali Feed (tick data)
- Historique : 5 ans
- Granularité : 1min brut → 5min agrégé (backtest)
- Session : 17h30–15h30 CT (Globex), exclusion 30min début/fin
- Fenêtre trading : 4h00–14h00 CT

## Hedge Ratio Methods
1. OLS Rolling (bêta dynamique)
2. Filtre de Kalman (bêta adaptatif)
3. Dollar Neutral (notionnel)
4. Volatility Neutral (vol réalisée)

## Métriques Statistiques
- Z-score (barres 5min)
- ADF Test (stationnarité)
- Hurst Exponent (mean-reversion, H < 0.5)
- Half-Life (Ornstein-Uhlenbeck)
- Corrélation glissante
- Filtre de Kalman (spread filtré)

## Phase 1 — Python Pipeline
1. Ingestion données CSV (Sierra Charts export)
2. Nettoyage : filtrage horaire, sessions, gaps
3. Construction spread (6 paires)
4. Calcul métriques statistiques
5. Optimisation paramètres (fenêtres, seuils z-score, hedge ratio)
6. Backtest avec coûts de transaction
7. Performance : Sharpe, Calmar, Hit Ratio, Profit Factor
8. Rapport comparatif

## Phase 2 — Sierra Charts (ACSIL C++)
1. Réplication modèles validés
2. Calcul temps réel : OLS Rolling + Kalman + Z-score
3. Sizing en contrats
4. Dashboard textbox visuel
5. Inputs paramétrables dynamiquement

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, statsmodels, scipy, filterpy
- **Phase 2** : C++ (ACSIL Sierra Charts API)

## Conventions
- Toujours travailler en venv
- Données en Chicago Time (CT)
- Paramètres optimisés en Phase 1 avant implémentation Phase 2
