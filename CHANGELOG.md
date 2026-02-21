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

### Config selectionnee : Config E (Principale)
- **Paire** : NQ_YM (5min)
- **Methode** : OLS rolling
- **Parametres** : OLS=3300, ZW=30, z_entry=3.15, z_exit=1.00, z_stop=4.50, conf>=67%, pas de time stop, window 02:00-14:00 CT, flat 15:30 CT
- **Performance** : 225 trades / 5.2 ans (43/an), PF 1.83, Sharpe 1.26, Calmar 0.96, WR 67.1%, PnL $25,790, MaxDD -$5,190
- **Validation** : IS/OOS GO (PF OOS 1.73 > IS 1.69), Walk-Forward 4/5, Permutation p=0.000

### Config backup : Config C (Volume+Filtre horaire)
- **Parametres** : OLS=2970, ZW=42, z_entry=2.95, z_exit=1.25, z_stop=4.00, conf>=67%, pas de time stop, window 02:00-15:00 CT + filtre horaire 8h-11h CT, flat 15:30 CT
- **Performance** : 351 trades / 5.2 ans (68/an), PF 1.53, Sharpe 1.16, WR 62.1%, PnL $25,780, MaxDD -$4,450
- **Avec filtre 8h-11h** : PF monte a 1.78, volume reduit mais qualite amelioree
- **Validation** : IS/OOS GO (PF OOS 1.67), Walk-Forward 4/5

### Pourquoi Config E :
- Meilleur compromis volume (43 trades/an) vs qualite (PF 1.83)
- OOS > IS (pas d'overfitting)
- Walk-Forward 4/5 solide
- Pas de time stop = moins de parametres = plus robuste

### Pistes non explorees :
1. **Mode overlay** : utiliser le confidence score comme filtre de timing sur un biais macro discretionnaire
2. **Calibration 3min independante** : explorer des lookbacks plus courts (pas de scaling lineaire du 5min)
3. **Multi-paire** : agreger NQ_YM + NQ_ES pour augmenter le volume
4. **Filtre horaire fin** : 8h-12h CT concentre 85% du profit — bloquer 4h-6h recupere ~$865

---

## Phase 6 — Grid Search Affine 1,080,000 combos (NQ_YM 5min)

### Optimisations implementees :
- **Numba JIT** : signal generator (451x), confidence filter, time stop, window filter
- **Vectorisation numpy** : compute_confidence (138x speedup)
- **run_backtest_grid** : backtest sans equity curve (~20x plus rapide)
- **Bug fix** : apply_time_stop re-entrait immediatement apres exit force → corrige
- **Validation** : 6/6 fonctions numba identiques aux implementations Python originales + 37/37 tests OK

### Test 6.1 — Grid search affine (1,080,000 combos, 141s)
- **Script** : `scripts/run_refined_grid.py --workers 5`
- **Parametres** :
  - OLS : 1980, 2310, 2640, 2970, 3300
  - ZW : 24, 30, 36, 42, 48
  - z_entry : 2.85 → 3.20 (step 0.05)
  - z_exit : 1.00, 1.10, 1.20, 1.25, 1.30, 1.40
  - z_stop : 3.75, 4.00, 4.25, 4.50
  - confidence : 67, 68, 69, 70, 72
  - time_stop : 0, 12, 18, 24, 36 bars
  - windows : 9 combinaisons (02:00-04:00 start × 12:00-15:00 end, flat 15:30)
- **Resultats** : 587,614 configs profitables (54% du total)
- **Vitesse** : 7,635 combos/s avec 5 workers

### Test 6.2 — Analyse approfondie (21 configs, full backtest)
- **Script** : `scripts/analyze_grid_results.py`

### Top 5 configs retenues :

| # | Profil | Config | Trd | WR% | PnL | PF | Sharpe | Calmar | MaxDD | Trd/an |
|---|--------|--------|-----|-----|-----|----|--------|--------|-------|--------|
| A | Balanced | OLS=2640 ZW=30 z=3.15/1.20/4.50 c=70 ts=18b 03:00-14:00 | 123 | 73.2% | $21,045 | 2.94 | 1.92 | 1.52 | -$2,665 | 24 |
| B | Volume+PnL | OLS=2640 ZW=48 z=3.15/1.00/4.25 c=68 ts=36b 04:00-14:00 | 288 | 68.1% | $26,400 | 1.56 | 1.11 | 0.57 | -$8,885 | 55 |
| C | Max Volume | OLS=2970 ZW=42 z=2.95/1.25/4.00 c=67 ts=none 02:00-15:00 | 351 | 62.1% | $25,780 | 1.53 | 1.16 | 1.11 | -$4,450 | 68 |
| D | Sharpe alt | OLS=2640 ZW=30 z=3.05/1.20/4.50 c=70 ts=18b 04:00-13:00 | 112 | 70.5% | $20,140 | 2.96 | 1.82 | 1.46 | -$2,655 | 22 |
| E | PnL+PF | OLS=3300 ZW=30 z=3.15/1.00/4.50 c=67 ts=none 02:00-14:00 | 225 | 67.1% | $25,790 | 1.83 | 1.26 | 0.96 | -$5,190 | 43 |

### Sensibilite parametrique (configs PF > 1.3) :
- **OLS=2640** domine (PF moy 1.58 vs 1.39-1.52 autres)
- **ZW=30** optimal (PF moy 1.56)
- **z_entry 3.05-3.15** zone plate (PF 1.56) — pas de pic isole
- **z_exit=1.20** meilleur (PF 1.57)
- **z_stop=4.25-4.50** equivalents
- **conf=70** cassure nette (PF 1.61 vs 1.43-1.46 pour 67-69)
- **ts=18** legerement meilleur (PF 1.54)
- **Window 03:00-14:00** meilleur (PF 1.55)

### Deep dive config A (balanced champion) :
- **Heures benefiques** : 8h-10h CT (85% du profit, WR 77-100%)
- **Heures critiques** : 5h-6h CT (perdantes, WR 50%)
- **Duree** : gagnants 4.1b (20min), perdants 7.1b (35min) — perdants 2.2x plus longs
- **Long/Short** : equilibre OK — Long $10,425 (49 trd) vs Short $8,145 (54 trd)
- **Stabilite** : 17/21 trimestres positifs (81%)

### Test 6.3 — Validation top 5 configs
- **Script** : `scripts/validate_top5_configs.py`
- **IS/OOS (60/40)** : 5/5 configs GO — Config B et E ont OOS > IS
- **Permutation (1000x)** : GO — p=0.000, PF observe 2.94 vs permute 0.78
- **Walk-Forward (IS=2y, OOS=6m, step=6m)** : A 4/5, B 2/5 (ELIMINE), C 4/5, D 3/5, E 4/5
- **Time stop** : ts=18b optimal pour Config A, pas de time stop optimal pour E
- **Filtre horaire** : 8h-12h boost Config A a PF 5.15 ; 8h-11h boost Config C a PF 1.78

### Selection finale :
- **Config E** : principale (meilleur compromis volume/qualite, OOS > IS)
- **Config C** : backup (plus de volume avec filtre horaire 8h-11h)
- **Config B** : eliminee (Walk-Forward 2/5, instable)

### Test 6.4 — Micro-contrats, multiplicateur de lots, SL dollar

**Implementes** : MNQ/MYM dans instruments.yaml, `find_optimal_multiplier()` (sizing),
`_apply_dollar_stop()` (numba), params `max_multiplier` + `dollar_stop` dans les 3 engines.

**Script** : `scripts/compare_micro_sizing.py`

| Config | Trd | WR% | PnL | PF | Sharpe | MaxDD |
|--------|-----|-----|-----|----|--------|-------|
| NQ/YM x1 (baseline) | 225 | 67.1% | $25,790 | 1.83 | 1.07 | -$5,190 |
| NQ/YM x2 | 225 | 69.3% | $35,485 | 1.71 | 0.93 | -$10,140 |
| NQ/YM x2 SL$500 | 225 | 56.0% | -$5,840 | 0.92 | -0.20 | -$17,175 |
| NQ/YM x2 SL$1000 | 225 | 63.1% | $3,685 | 1.05 | 0.15 | -$16,265 |
| NQ/YM x3 | 225 | 68.9% | $53,015 | 1.75 | 0.87 | -$16,405 |
| MNQ/MYM x3 | 225 | 63.1% | $3,754 | 1.50 | 0.63 | -$1,817 |
| MNQ/MYM x3 SL$500 | 225 | 63.1% | $4,801 | 1.74 | 0.87 | -$1,029 |

**Conclusions** :
- **SL dollar inadapte au NQ/YM standard** : coupe les trades avant retour a la moyenne (tous SL < $1000 perdants)
- **Micros** : PF inferieur (commissions pesent plus proportionnellement), pas d'avantage vs standard
- **mm=2 viable** sur compte perso (MaxDD $10k), **mm=1 pour propfirm** ($5k DD)
- **mm=3** : PnL x2 mais MaxDD x3 ($16k) — trop risque
- **Decision** : rester sur NQ/YM standard x1, SL dollar desactive. Multiplicateur et micros disponibles pour scaling futur
