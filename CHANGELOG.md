# CHANGELOG — Recherche Backtest Spread Indice

Historique chronologique des tests, resultats et decisions.
Derniere mise a jour : 2026-02-21

---

## Phase 0 — Backtest exploratoire par paire (NQ_ES focus)

### Test 0.1 — OLS rolling NQ_ES (toutes configs)
- **Configs** : z_entry 2.0-3.0, OLS window 10-30j, tous profils metrics
- **Resultat** : Toutes configs perdantes sur NQ_ES en OLS rolling
- **Conclusion** : OLS rolling ne fonctionne pas sur NQ_ES

### Test 0.2 — Kalman NQ_ES
- a=1e-6 entry=2.0 conf=50% : 24 trades, 79% win, +$8,575, PF 3.45 (echantillon faible)
- a=1e-7 entry=2.5 conf=50% : 55 trades, 56% win, +$6,123, PF 1.33
- **Conclusion** : NQ_ES seule paire avec edge symetrique long/short confirme

### Test 0.3 — Kalman NQ_RTY
- a=1e-7 : +$21,650, PF 2.67, 77% win
- **Conclusion** : REJET — profit vient du biais long (NQ surperforme structurellement)

### Test 0.4 — ES_YM
- **Resultat** : Systematiquement perdant sur toutes configs
- **Conclusion** : Paire non viable

---

## Phase 1 — Grid Search 43,200 configs

### Test 1.1 — Grid search complet (6 paires x 7,200 combos)
- **Script** : `scripts/run_grid.py --workers 20`
- **Parametres** :
  - OLS window : 1320, 2640, 3960, 5280, 6600, 7920
  - ZW : 12, 24, 36, 48
  - z_entry : 2.0, 2.5, 3.0
  - z_exit : 0.0, 1.0, 1.5, 2.0
  - z_stop : 3.0, 3.5, 4.0
  - profil metrics : 3 profils
  - min_confidence : 50%, 60%, 70%, 80%
- **Champion brut** : NQ_YM, OLS=5280, ZW=12, e=2.0, x=0.0, s=3.0

### Findings grid search :
1. **z_exit=0.0 est un ARTEFACT** — exit=0 = ne jamais sortir sauf stop loss
2. **Confidence filtre elimine les BONS trades au-dela de 40%** (dans la config champion brute)
3. **Zone robuste** : OLS=2640, ZW=36, profil=tres_court, z_entry=3.0, z_exit=1.5
4. NQ_YM domine toutes les paires
5. Config champion fragile (pic isole en PnL, les voisins chutent)

---

## Phase 2 — Ablation des filtres (Approche A)

### Test 2.1 — Ablation filtres individuels NQ_YM
- **Script** : `scripts/analyze_filters.py`
- **Config robuste** : OLS=2640, ZW=36, e=3.0, x=1.5, s=4.0, profil tres_court
- **Seuils binaires** : ADF<-2.86, Hurst<0.45, Corr>0.80, HL in [5,120]

| Filtre | Trades | PnL | PF |
|--------|--------|-----|-----|
| Aucun filtre (z-score pur) | 1,859 | -$121k | 0.80 |
| Confidence >= 70% | 94 | +$13,650 | 2.22 |
| Regime ALL ON (binaire) | 0 | — | — |
| Hurst+Corr+HL (sans ADF) | 510 | +$4,700 | 1.03 |
| ADF seul | 48 | -$1,800 | 0.76 |
| Hurst seul | 1,038 | -$34k | 0.86 |
| Corr seule | 791 | +$9,000 | 1.08 |
| HL seul | 1,170 | -$52k | 0.83 |

### Test 2.2 — Grille seuils Hurst x Corr (16 combos)
- **Meilleur** : H<0.45 C>0.85 HL → 470 trades, +$9k, PF 1.08
- **Conclusion** : Filtres binaires insuffisants. Le scoring continu du confidence fait le travail.

### Test 2.3 — ADF kill switch 480 bars
- ADF 480 seul : 53 trades, PF 1.20
- ADF 480 + Hurst+Corr+HL : 0 trades (trop restrictif)
- **Conclusion** : ADF sur 12 bars = inutile (2.6% pass rate, median -0.69 vs seuil -2.86). Sur 480 bars = marginal.

### Insight cle Phase 2 :
Le confidence score (scoring continu + poids ADF 40%) selectionne 94/1859 trades avec PF 2.22.
Les filtres binaires individuels n'arrivent pas au meme resultat.
Le scoring continu capture des fenetres ou TOUTES les metriques sont simultanement favorables.

---

## Phase 3 — Validation quant complete

### Test 3.1 — Distribution temporelle (1A)
- **Script** : `scripts/validate_confidence.py --temporal`
- **Config** : e=3.0, x=1.5, c=70%
- 94 trades sur 6 ans (2021-2026)
- Long/short symetrique (les 2 cotes profitables)
- Concentration max 1 annee : 39.9% du PnL
- **VERDICT : GO**

### Test 3.2 — IS/OOS split 60/40 (3A)
- **Script** : `scripts/validate_confidence.py --isoos`
- IS (2020-12 → 2024-01) : 54 trades, PF 2.29, Sharpe 1.12
- OOS (2024-01 → 2026-02) : 40 trades, PF 2.14, Sharpe 1.08
- Degradation minimale IS → OOS
- **VERDICT : GO**

### Test 3.3 — Test de permutation 1000x (3B)
- **Script** : `scripts/validate_confidence.py --permutation`
- PF observe : 2.22 | PF permutations moyen : ~0.80
- 0/1000 permutations battent le PF observe
- p-value = 0.000
- **VERDICT : GO** — le confidence score capture un signal reel, non reproductible par hasard

### Test 3.4 — Decomposition composantes (1B)
- **Script** : `scripts/validate_confidence.py --decompose`
- Correlation de rang Spearman (sous-score vs PnL) : aucune significative
- Pas de composante individuelle predictive du PnL
- **VERDICT : NEUTRE** — le filtre agit comme gate (entre/n'entre pas), pas comme predicteur de magnitude

### Test 3.5 — Sensibilite parametrique (2A)
- **Script** : `scripts/validate_confidence.py --sensitivity`
- ~30 perturbations des params du confidence score
- 62% des configs gardent PF > 1.5
- **VERDICT : MARGINAL** — robuste mais pas insensible

### Test 3.6 — Walk-Forward (3C)
- **Script** : `scripts/validate_confidence.py --walkforward`
- IS=2 ans, OOS=6 mois, pas=6 mois → 6 fenetres
- 5/6 fenetres OOS profitables
- PF moyen OOS : 3.15
- PnL total OOS : ~$29,850
- **VERDICT : GO**

### Test 3.7 — Monte Carlo propfirm (4A)
- **Script** : `scripts/validate_confidence.py --propfirm`
- 10,000 simulations, 27 trades/an (realiste OOS)
- P(atteindre $75k) < 1% | PnL annuel moyen : ~$3,200
- **VERDICT : STOP** — edge reel mais volume insuffisant pour $300/jour

---

## Phase 4 — Recherche du sweet spot volume/qualite

### Test 4.1 — Sweep confidence 58-81% (granularite 1%)
- Transition nette a 66→67% : PF passe de ~1.03 a 1.20
- Sweet spot : 67-69% (98-128 trades, PF 1.20-1.67)
- Au-dela de 70% : PF monte mais <30 trades

### Test 4.2 — IS/OOS par palier de confidence

| Conf | IS Trades | IS PF | OOS Trades | OOS PF | OOS Sharpe |
|------|-----------|-------|------------|--------|------------|
| 67% | 75 | 1.21 | 53 | 1.17 | 0.28 |
| 68% | 65 | 1.67 | 45 | 1.21 | 0.32 |
| 69% | 57 | 2.16 | 41 | 1.29 | 0.42 |
| 70% | 54 | 2.29 | 40 | 2.14 | 1.08 |

### Test 4.3 — Mini-grid 84 combos (z_entry x z_exit x confidence)
- z_entry : 2.5, 3.0, 3.5
- z_exit : 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00
- confidence : 67%, 68%, 69%, 70%
- **z_entry=2.5 : TOUT negatif** — le spread doit etre a 3 sigma minimum
- **Decouverte z_exit=1.25** : plus de trades que 1.5 tout en gardant la qualite

### Top 5 configs mini-grid :

| # | Entry | Exit | Conf | Trades | PF | Sharpe | DD% | Score |
|---|-------|------|------|--------|-----|--------|-----|-------|
| 1 | 3.5 | 1.25 | 70% | 74 | 2.93 | 1.13 | 2.2% | 28.5 |
| 2 | 3.0 | 1.25 | 70% | 144 | 1.99 | 1.18 | 3.7% | 28.1 |
| 3 | 3.0 | 1.50 | 70% | 94 | 2.22 | 1.10 | 1.9% | 23.7 |
| 4 | 3.5 | 1.50 | 70% | 51 | 2.99 | 0.96 | 2.2% | 20.5 |
| 5 | 3.0 | 0.50 | 67% | 503 | 1.25 | 0.64 | 22.4% | 17.9 |

### Test 4.4 — Validation IS/OOS + Walk-Forward des top 5

| # | Config | OOS PF | OOS Sharpe | WF profitables | WF PnL total |
|---|--------|--------|------------|----------------|--------------|
| 1 | e=3.5 x=1.25 c=70 | 2.98 | 1.27 | 6/6 | $31,070 |
| 2 | e=3.0 x=1.25 c=70 | 1.56 | 0.68 | 6/6 | $30,535 |
| 3 | e=3.0 x=1.50 c=70 | 2.14 | 1.08 | 5/6 | $29,850 |
| 4 | e=3.5 x=1.50 c=70 | ~inf | 1.95 | 5/6 | $19,130 |
| 5 | e=3.0 x=0.50 c=67 | — | — | 5/6 | $35,490 |

Toutes les top 5 passent la validation. Edge confirme sur toute la grille z_entry >= 3.0.

---

## Phase 5 — Multi-timeframe (1min, 3min, 5min)

### Test 5.1 — Scaling lineaire des parametres
Parametres scales proportionnellement au timeframe (meme duree temporelle) :

| Param | 5min | 3min (x5/3) | 1min (x5) |
|-------|------|-------------|-----------|
| OLS | 2640 | 4400 | 13200 |
| ZW | 36 | 60 | 180 |
| ADF | 12 | 20 | 60 |
| Hurst | 64 | 107 | 320 |
| HL | 12 | 20 | 60 |
| Corr | 6 | 10 | 30 |

### Resultats multi-timeframe :

| TF | Config | Trades | PF | Sharpe | OOS PF |
|----|--------|--------|-----|--------|--------|
| 5min | e=3.0 x=1.25 c=70 | 144 | 1.99 | 1.18 | 1.56 |
| 5min | e=3.5 x=1.25 c=70 | 74 | 2.93 | 1.53 | 2.98 |
| 3min | e=3.0 x=1.25 c=70 | 156 | 1.29 | 0.43 | ~2.4 |
| 3min | e=3.0 x=1.50 c=70 | ~100 | ~1.1 | — | ~2.9 |
| 1min | e=3.0 x=1.25 c=70 | 393 | 0.84 | neg | neg |

### Conclusion multi-timeframe :
- **5min reste le meilleur** — edge valide
- **3min** : IS faible mais OOS fort → anomalie, probablement besoin de calibration propre (pas de scaling lineaire)
- **1min** : negatif — le bruit microstructure tue l'edge
- **Le scaling lineaire x5 est tautologique** — meme duree temporelle = meme information. Pour que 1min apporte quelque chose, il faudrait explorer des lookbacks plus courts (regimes intraday).

---

## Conclusion globale (etat au 2026-02-21)

### Edge valide :
- **Paire** : NQ_YM (5min)
- **Methode** : OLS rolling, confidence score >= 70%
- **Config de reference** : OLS=2640, ZW=36, e=3.0, x=1.25, c=70%
- **Performance** : 144 trades sur 5 ans, PF 1.99, Sharpe 1.18, DD 3.7%
- **Validation** : IS/OOS GO (PF OOS 1.56), Walk-Forward GO (6/6), Permutation GO (p=0.000)

### Probleme ouvert :
- **Volume insuffisant pour propfirm** : 27 trades/an (conf=70%) a 96 trades/an (conf=67%)
- Target propfirm = $300/jour = $75k/an, edge livre ~$3-6k/an en standalone

### Pistes non explorees :
1. **Mode overlay** : utiliser le confidence score comme filtre de timing sur un biais macro discretionnaire (le systeme ne genere pas les trades, il valide/invalide le timing)
2. **Calibration 3min independante** : les parametres 3min ne doivent pas etre un scaling lineaire du 5min — explorer des lookbacks plus courts
3. **Multi-paire** : agreger NQ_YM + NQ_ES pour augmenter le volume
4. **Parametres 1min inedits** : explorer des OLS/ZW courts (regimes intraday) au lieu du scaling x5
5. **Config #5 volume** : e=3.0 x=0.50 c=67 (503 trades, PF 1.25) — viable pour volume mais DD eleve (22%)
