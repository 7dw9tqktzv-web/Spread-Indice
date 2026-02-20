# Spread Trading Indicator — Project CLAUDE.md

## Project Overview
Système complet de spread trading intraday sur futures d'indices US (NQ, ES, RTY, YM).
- **Phase 1** : Moteur de backtest Python (optimisation & validation)
- **Phase 2** : Indicateur Sierra Charts temps réel (ACSIL C++)

Le biais directionnel journalier est discrétionnaire — le système time l'entrée avec précision statistique sur des billets macroéconomiques.

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
1. **OLS Rolling** (bêta dynamique sur log-prix, fenêtre 7200 bars = 30j)
2. **Filtre de Kalman** (bêta adaptatif sur log-prix, Q = α×R×I, z-score innovation)

## Sizing
- **Dollar Neutral × β** : N_b = round((Notionnel_A / Notionnel_B) × β × N_a)
- Leg A fixe = 1 contrat, leg B calculée dynamiquement
- Formule ACSIL v1.4 compatible

## Métriques Statistiques (informatives, affichées dans textbox)
- Z-score OLS (rolling 12 bars = 1h) ou Kalman (innovation ν/√F, auto-adaptatif)
- ADF Test : simplifié Sierra (rolling 24 bars) + statsmodels complet
- Hurst Exponent (R/S, rolling 64 bars)
- Half-Life AR(1) (rolling 24 bars)
- Corrélation glissante (Pearson sur log-prix, 12 bars, seuil configurable)

## Paramètres Validés (5min)
| Paramètre | Valeur |
|-----------|--------|
| OLS lookback | 7200 bars (30j) |
| Z-score OLS | 12 bars (1h) |
| Corrélation | 12 bars (1h) |
| ADF | 24 bars (2h) |
| Hurst | 64 bars (~5h20) |
| Half-life | 24 bars (2h) |
| Kalman alpha_ratio | 1e-5 (à optimiser: [1e-6, 1e-5, 1e-4]) |
| Kalman warmup | 100 bars |

## Phase 1 — Python Pipeline
1. Ingestion données CSV (Sierra Charts export)
2. Nettoyage : filtrage horaire, sessions, gaps
3. Construction spread (6 paires) sur log-prix
4. Calcul hedge ratio (OLS Rolling + Kalman)
5. Sizing dollar-neutral × β
6. Calcul métriques statistiques (ADF, Hurst, half-life, corrélation)
7. Génération signaux (z-score entry/exit/stop)
8. Backtest avec coûts de transaction
9. Performance : Sharpe, Calmar, Hit Ratio, Profit Factor
10. Optimisation paramètres (walk-forward, grid search)
11. Rapport comparatif

## Phase 2 — Sierra Charts (ACSIL C++)
1. Réplication modèles validés
2. Calcul temps réel : OLS Rolling + Kalman + Z-score
3. Sizing en contrats (dollar-neutral × β)
4. Dashboard textbox visuel
5. Inputs paramétrables dynamiquement

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, statsmodels, scipy
- **Phase 2** : C++ (ACSIL Sierra Charts API)

## Conventions
- Toujours travailler en venv
- Données en Chicago Time (CT)
- Tous les calculs sur **log-prix** (ln) pour OLS et Kalman
- Paramètres optimisés en Phase 1 avant implémentation Phase 2
- **Git** : utiliser l'agent GitHub (gh) pour tous les commits, push et opérations de branche — jamais de commandes git manuelles
- Valider chaque étape avec l'utilisateur avant de passer à la suivante
