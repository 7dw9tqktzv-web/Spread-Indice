# CHANGELOG — Recherche Backtest Spread Indice

Historique chronologique des tests, resultats et decisions.
Derniere mise a jour : 2026-02-24

---

## Phases 0-5 — Exploration et validation initiale (resume)

Exploration exhaustive sur 6 paires, 43,200 configs OLS broad grid, ablation des filtres, validation quant complete, multi-timeframe. Conclusions cles :

- **NQ_YM domine** toutes les paires en OLS. NQ_ES : Kalman marginal. ES_YM : systematiquement perdant.
- **z_exit=0.0 est un artefact** (ne jamais sortir sauf stop). **z_entry=2.5 tout negatif** (3+ sigma minimum).
- **Confidence scoring continu >> filtres binaires** : 94/1859 trades selectionnes, PF 2.22 vs 0.80 sans filtre.
- **Validation IS/OOS GO, Walk-Forward 5/6, Permutation p=0.000** sur la config exploratoire.
- **Monte Carlo propfirm : STOP** -- edge reel mais volume insuffisant (27 trades/an) pour $300/jour.
- **5min optimal**, 1min negatif (bruit microstructure), 3min anomalie IS/OOS.
- **Sweet spot confidence : 67-69%** (transition nette a 67%).
- **8h-12h CT concentre 85% du profit** (heures US open).

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

---

## Phase 7 — Grid Search Kalman NQ_YM (indicateur complementaire)

Objectif : evaluer si le filtre de Kalman apporte une information complementaire a OLS pour affichage en textbox Sierra (biais discretionnaire). Kalman n'avait JAMAIS ete teste sur NQ_YM.

### Differences cles OLS vs Kalman :
- OLS : z-score rolling (zscore_window), beta fixe sur fenetre
- Kalman : z-score innovation nu/sqrt(F) auto-adaptatif (N(0,1) par construction), beta adaptatif
- Consequence : z_entry optimal Kalman = 1.0-2.5 (vs 3.15 OLS), z_stop = 2.5-2.8 (vs 4.5 OLS)

### Test 7.1 — Grid v1 (290,304 combos, 95s)
- **Script** : `scripts/run_grid_kalman.py`
- **Parametres** : 12 alpha_ratio x 13 z_entry x 9 z_exit x 8 z_stop x 12 min_conf x 3 profils
- **Alpha sweet spot** : 1e-7 a 5e-7 (tres lent = beta stable)
- **z_entry optimal** : 1.0-2.25 (beaucoup plus bas que OLS)
- **z_stop = 2.5 domine** (correct pour innovation z N(0,1))
- **19 configs battent OLS PF** (1.83), meilleur PF 2.35 (59 trades)
- **Conclusion v1** : Kalman complementaire, pas remplacement. Signaux rares mais fiables.

### Test 7.2 — Grid v2 raffine (1,482,624 combos, 12min)
- **Script** : `scripts/run_grid_kalman_v2.py --workers 10`
- **Nouveaux parametres testes** (recommandation agent expert-spread) :
  - warmup : [100, 200, 500, 750] — convergence Kalman
  - gap_P_multiplier : [2.0, 5.0, 10.0, 25.0] — inflation P aux gaps overnight
- **Alpha** : [5e-8, 1e-7, 1.5e-7, 2e-7, 3e-7, 5e-7, 7e-7, 1e-6]
- **z_entry** : 1.0 -> 2.5 step 0.125, **z_exit** : 0.0 -> 1.5, **z_stop** : 2.0 -> 3.5
- **min_confidence** : [50, 60, 63, 65, 67, 69, 70, 72, 75]
- **Resultat cle** : warmup et gap_P_mult n'ont AUCUN impact (Kalman converge vite pour alpha <= 1e-6)
- **5,448 configs battent OLS PF** (vs 19 en v1)

### Test 7.3 — Rapport detaille Kalman vs OLS (trade-level)
- **Script** : `scripts/analyze_kalman_report.py`
- **Configs re-run avec moteur complet** (equity + trades individuels)

#### Tableau comparatif complet :

| Config | Trades | WR% | PnL | PF | Sharpe | Calmar | MaxDD$ | PropFirm |
|--------|--------|-----|-----|----|--------|--------|--------|----------|
| OLS ConfigE | 176 | 68.2% | $23,215 | 1.86 | 1.00 | 0.90 | $4,905 | 20.5 |
| K_BestPF (a=1.5e-7 ze=2.0 zx=0.75 zs=2.75 c=67 moyen) | 53 | 84.9% | $36,425 | **5.05** | **1.37** | 0.99 | $7,705 | 18.9 |
| K_BestPnL (a=3e-7 ze=1.375 zx=0.25 zs=2.75 c=75 tc) | 260 | 73.5% | **$73,580** | 1.64 | 1.18 | 0.88 | $16,730 | 26.1 |
| K_BestVolume (a=1.5e-7 ze=1.0 zx=0.75 zs=2.75 c=69 tc) | **395** | 70.9% | $52,610 | 1.32 | 0.77 | 0.37 | $26,195 | **27.3** |
| K_RunnerPF (a=3e-7 ze=2.125 zx=0.25 zs=2.8 c=65 court) | 59 | 74.6% | $37,970 | 2.90 | 1.09 | **1.46** | $5,925 | 19.2 |

#### Analyse des trades perdants :

**OLS perd a 8h-9h (apres open US), Kalman perd a 4h (pre-marche) et 13h (fin de fenetre).**

| Aspect | OLS ConfigE | K_BestPF | K_BestPnL |
|--------|-------------|----------|-----------|
| Heures perdantes | 8h-9h | 4h | 4h + 13h |
| Jours de perte communs | — | **0%** | **3%** |
| Confidence perdants | 70.2% | 39.5% | 58.0% |
| |Z| entree perdants | 2.40 | 1.30 | 1.10 |
| Duree perdants | 48min | 439min | 402min |
| Side desequilibre | SHORT perd 2x | Equilibre | Equilibre |
| Pire jour semaine | Lundi (45%) | Jeudi (33%) | Jeudi (36%) |

#### Chevauchement temporel OLS vs Kalman :
- K_BestPF : 4 jours communs / 169 OLS (2%) — **quasiment independants**
- K_BestPnL : 37 jours communs / 169 OLS (22%) — concordance 54% les 2 gagnent, 5% les 2 perdent
- K_BestVolume : 58 jours communs (34%) — NON complementaire (perd aussi quand OLS perd)

#### Complementarite (Kalman quand OLS perd) :
- **K_BestPnL : COMPLEMENTAIRE** — sur 55 jours de perte OLS, Kalman trade 8 fois et gagne 75%
- **K_BestPF : pas de chevauchement** — ne trade jamais les jours ou OLS perd
- **K_BestVolume : NON complementaire** — perd $17k sur les memes jours

#### Decomposition annuelle :
- **K_BestPnL** : brille 2024 ($30k) + 2025 ($19k), annee noire 2023 (-$11k)
- **K_BestPF** : negatif aussi en 2023 (-$4k), tous les autres ans positifs
- **OLS** : plus regulier, aucune annee negative, gains modestes

### Conclusions Phase 7 :

1. **Warmup et gap_P_mult negligeables** : Kalman converge vite, ces params n'impactent pas les resultats
2. **Alpha sweet spot : 1.5e-7 a 3e-7** (beta tres stable, adaptation lente)
3. **Innovation z-score est N(0,1)** : z_entry=1.5-2.0 et z_stop=2.5-2.8 sont les bons seuils (pas 3.15/4.5 comme OLS)
4. **OLS et Kalman perdent sur des jours/heures/regimes differents** = complementarite structurelle
5. **Config Kalman retenue pour textbox Sierra : K_RunnerPF**
   - Meilleur Calmar (1.46), MaxDD contenu ($5,925), PF 2.90
   - alpha=3e-7, z_entry=2.125, z_exit=0.25, z_stop=2.8, conf=65%, profil court
6. **Decision** : OLS reste le moteur de signaux. Kalman affiche beta + z-score + direction dans textbox Sierra comme indicateur complementaire discretionnaire

### Test 7.4 — Grid v3 definitif (1,009,800 combos, 218s)
- **Script** : `scripts/run_grid_kalman_v3.py --workers 10`
- **Parametres** (recommandation agent expert-spread) :
  - Alpha : [1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 5e-7] (6 valeurs, warmup/gap fixes)
  - z_entry : 1.25 -> 2.25 step 0.0625 (17 valeurs)
  - z_exit : [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5] (9 valeurs)
  - z_stop : [2.25, 2.5, 2.625, 2.75, 2.875, 3.0, 3.25] (7 valeurs)
  - min_confidence : [50, 60, 63, 64, 65, 66, 67, 68, 69, 70, 75] (11 valeurs)
  - 5 entry windows : [03:00-12:00, 04:00-12:00, 04:00-13:00, 04:00-14:00, 05:00-12:00]
- **Resultats** : 545,075 configs profitables (54%), 63,897 battent OLS PF

#### Top config par profil propfirm (5 profils) :

| Profil | Criteres | Configs | Best PnL | PF | Trades | WR% |
|--------|----------|---------|----------|-----|--------|-----|
| Sniper | PF>=2.5, WR>=75%, 30-150 trd | 12,275 | $64,990 | 3.24 | 91 | 76.9% |
| Steady | PF>=1.5, 100-300 trd | 14,618 | $84,825 | 1.84 | 238 | 78.2% |
| Volume | PF>=1.2, >=200 trd | 23,621 | $84,825 | 1.84 | 238 | 78.2% |
| Risk-Adj | PF>=1.8, >=50 trd | 25,453 | $84,825 | 1.84 | 238 | 78.2% |
| Balanced | PF>=1.3, 80-250 trd | 67,514 | $84,825 | 1.84 | 238 | 78.2% |

### Test 7.5 — Validation IS/OOS + Walk-Forward top 5 Kalman
- **Script** : `scripts/validate_kalman_top.py`

#### IS/OOS (60/40) — 5/5 GO :
- 4 configs sur 5 ont OOS PF > IS PF (pas d'overfitting)
- K_Balanced: IS 1.33 -> OOS 1.66 (+25%)
- K_ShortWin: IS 1.46 -> OOS 1.58 (+8%)

#### Permutation (1000x) — 2/2 GO :
- K_Balanced: p=0.000 (PF 2.09 vs permut 1.19)
- K_Sniper: p=0.000 (PF 2.45 vs permut 1.34)

#### Walk-Forward (IS=2y, OOS=6m) — 5/5 GO :

| Config | WF prof. | PnL total | PF moy | Verdict |
|--------|----------|-----------|--------|---------|
| K_Balanced (03:00-12:00) | **5/5** | **$32,000** | **2.34** | GO |
| K_ShortWin (05:00-12:00) | **5/5** | **$31,255** | **2.16** | GO |
| K_Quality (04:00-13:00) | 4/5 | $22,905 | 2.01 | GO |
| K_BestPnL (03:00-12:00) | 4/5 | $16,540 | 2.14 | GO |
| K_Sniper (05:00-12:00) | 3/5 | $6,805 | 0.88 | GO |

#### Yearly : 2023 seule annee negative pour tous. 2024-2026 tres fort.

### Test 7.6 — MaxDD et risk propfirm (full engine)
- **Script** : `scripts/find_safe_kalman.py`
- **Constat** : Les configs a volume (200+ trades) ont MaxDD $13k-$45k en E-mini — INCOMPATIBLES propfirm (trailing DD $4,500)
- **460/1994 configs safe** (MaxDD < $4,500 en E-mini) mais toutes ont < 50 trades (5-8 trades/an)
- **Streak max = 1** pour les meilleures configs safe (jamais 2 pertes consecutives)
- **Seule config balanced safe** (E-mini) : a=2.5e-7, 04:00-13:00, ze=1.562, zx=1.0, zs=2.75, c=75 → 56 trades, PF 2.31, MaxDD $-4,080

### Test 7.7 — Micro-contrats MNQ/MYM (propfirm-safe)
- **Script** : `scripts/find_safe_kalman_micro.py`
- **Multipliers micro** : MNQ=2.0$/pt, MYM=0.5$/pt (1/10 du standard), commission $0.62 RT

#### Scaling micro — Config champion (a=3e-7, tres_court, 03:00-12:00, ze=1.375) :

| Scale | Trades | WR% | PnL | PF | MaxDD | Streak | ConsDD | MaxLos | $/jour | Status |
|-------|--------|-----|-----|----|-------|--------|--------|--------|--------|--------|
| Micro x1 | 238 | 77.7% | $7,968 | 1.78 | $-1,916 | 3 | $-1,005 | $-735 | $6 | **SAFE** |
| **Micro x2** | **238** | **77.7%** | **$15,936** | **1.78** | **$-3,832** | **3** | **$-2,010** | **$-1,470** | **$12** | **SAFE** |
| Micro x3 | 238 | 77.7% | $23,905 | 1.78 | $-5,747 | 3 | $-3,015 | $-2,206 | $18 | WARN |
| E-mini x1 | 238 | 78.2% | $84,825 | 1.84 | $-18,425 | 3 | $-9,985 | $-7,330 | $65 | DANGER |

#### OLS ConfigE en micro :

| Scale | PnL | MaxDD | $/jour | Status |
|-------|------|-------|--------|--------|
| Micro x1 | $2,062 | $-536 | $2 | SAFE |
| Micro x2 | $4,125 | $-1,072 | $3 | SAFE |
| E-mini x1 | $25,790 | $-5,190 | $20 | WARN |

#### Distribution MaxDD micro x2 : 92.6% des configs ont MaxDD < $4,500

### Conclusions Phase 7 (finales) :

1. **Kalman valide comme indicateur complementaire** — 5/5 configs passent IS/OOS + Walk-Forward + Permutation
2. **Config Kalman champion** : a=3e-7, tres_court, ze=1.3125, zx=0.375, zs=2.75, c=75%, window 03:00-12:00
   - 218 trades, PF 2.09, WR 78.4%, $80,805, Sharpe 3.95, Calmar 5.71
   - Walk-Forward 5/5, OOS PF 1.66 > IS 1.33
3. **MaxDD incompatible propfirm en E-mini** : configs volume ($13k-$18k DD) vs trailing DD $4,500
4. **Micro x2 est propfirm-safe** : MaxDD $3,832, streak 3, mais revenu faible ($12/jour)
5. **Le trade-off est fondamental** : volume → DD eleve ; DD safe → peu de trades
6. **Decision** : Kalman en textbox Sierra (beta + z-score + direction). Pas de signaux automatiques.
   - E-mini : pour compte perso (MaxDD $18k acceptable)
   - Micro x2 : pour propfirm eval (MaxDD < $4,500)

---

## Phase 8 — Optimisation Kalman (Insights VIX) — 2026-02-22

Objectif : extraire et tester les ameliorations du filtre de Kalman issues d'un document d'analyse VIX (modele Ornstein-Uhlenbeck). Plan en 6 priorites (P1-P6), test sequentiel avec validation avant chaque etape.

### Implementation : P1 (R adaptatif EWMA) + P2 (Diagnostics P_trace/K_beta)

**Code modifie** :
- `src/hedge/kalman.py` : ajout `r_ewma_span`, `adaptive_Q` dans `KalmanConfig` + boucle R(t) = lambda*R(t-1) + (1-lambda)*nu^2 + stockage P_trace/K_beta/R_history dans diagnostics
- `src/hedge/base.py` : ajout `diagnostics: dict` dans `HedgeResult`
- `tests/test_hedge/test_kalman.py` : +13 tests (non-regression, convergence R, diagnostics, adaptive_Q)
- **Total tests** : 62 (49 originaux + 13 nouveaux)

### Test 8.1 — P1 : R adaptatif EWMA (55 backtests)

**Script** : `scripts/test_adaptive_r.py`
- **Configs testees** : 5 top Kalman (K_Sniper, K_BestPnL, K_Balanced, K_Quality, K_ShortWin)
- **R combos** : r_ewma_span = [0, 200, 500, 1000, 2000, 5000] x adaptive_Q = [False, True]
- **Baseline** : r_ewma_span=0 (R fixe, comportement champion actuel)

#### Resultats :

| Config | Baseline MaxDD | Pire EWMA MaxDD | Degradation | R ratio max |
|--------|---------------|-----------------|-------------|-------------|
| K_Sniper | -$10,995 | -$27,645 | 2.5x | 98,713x (EWMA=200) |
| K_BestPnL | -$19,155 | -$42,455 | 2.2x | 98,713x |
| K_Balanced | -$19,155 | -$131,335 | 6.9x | 98,713x |
| K_Quality | -$19,155 | -$90,170 | 4.7x | 98,713x |
| K_ShortWin | -$19,155 | -$84,805 | 4.4x | 98,713x |

**VERDICT : P1 INVALIDE** — R adaptatif degrade le MaxDD de 2x a 7x sur TOUTES les configs, TOUS les spans.

**Mecanisme identifie** : EWMA(nu^2) capture bruit + signal. Quand le spread devie legitimement, R augmente → filtre anesthesie (plus confiance au modele vs observations) → beta ne s'adapte plus → z-score decolle → cascade de pertes. Le R ratio atteint 98,713x (EWMA=200), soit 5 ordres de magnitude d'oscillation.

**Difference cle avec le papier VIX** : Le VIX est directement observe (1 variable d'etat). Notre Kalman estime [alpha, beta] a partir de log-prix — les innovations ne refletent PAS directement le bruit de mesure. EWMA sur nu^2 melange inadaptation du modele et bruit reel.

**Decision** : r_ewma_span reste dans le code (defaut=0 = desactive, non-regression OK). Ne JAMAIS activer sur cette strategie.

### Test 8.2 — P2 : Diagnostics P_trace/K_beta + Analyse Spearman

**Script** : `scripts/analyze_kalman_diagnostics.py`

#### Phase 1 — Spearman global :

| Config | Rho (P_trace vs PnL) | p-value | Significant? |
|--------|----------------------|---------|--------------|
| K_Sniper | -0.151 | 0.172 | Non |
| K_BestPnL | -0.206 | 0.001 | Oui |
| K_Balanced | -0.187 | 0.004 | Oui |
| K_Quality | -0.151 | 0.020 | Oui |
| K_ShortWin | -0.146 | 0.025 | Oui |

Signal apparent : rho negatif = P_trace eleve → PnL plus faible.

#### Phase 2 — Deconfounding (identification du biais temporel) :

**Probleme** : P_trace diminue monotoniquement avec le temps (convergence Kalman). Les annees recentes performent mieux. Le Spearman global capture potentiellement time→P_trace ET time→performance, PAS P_trace→performance.

| Mesure | K_Balanced | K_Sniper |
|--------|------------|----------|
| Rho global (P_trace vs PnL) | -0.187 | -0.151 |
| Rho (Time vs P_trace) | **-1.000** | **-1.000** |
| Rho intra-annee moyen | -0.121 | **+0.037** |
| Rho partiel (ctrl. time) | NaN (P_trace = f(time)) | NaN |
| Annees significatives | 0/6 | 0/6 |

**VERDICT : P3 INVALIDE** — P_trace est un pur proxy temporel (rho = -1.000 avec le temps). Signal intra-annee trop faible (rho ~-0.12, aucune annee individuellement significative). Un filtre P_trace serait un "filtre temporel deguise" — inutile en walk-forward ou le Kalman redemarre a chaque fenetre.

**Decision** : P_trace et K_beta restent dans les diagnostics (cout zero, visualisation utile). P3 (filtre P_trace) abandonne. P4 (z-score OU) deprioritise.

### Test 8.3 — Autopsie 2023 (10 dimensions, 5 configs Kalman + OLS)

**Script** : `scripts/analyze_2023_losses.py`

2023 est la SEULE annee negative pour les 5 configs Kalman. Autopsie exhaustive :

#### Resultats K_Balanced 2023 :

| Dimension | Finding |
|-----------|---------|
| 1. Taille | 45 trades, 32 W ($20,435) vs 13 L ($-26,135) = **-$5,700** |
| 2. Type sortie | **FLAT = le tueur** : 9 trades, **-$19,440 (86% des pertes)** |
| 3. Side | Short detruit : **-$6,630** (NQ +38.2% vs YM +8.9%, tech rally) |
| 4. Heure | 5h-8h = zone de perte (pre-ouverture US) |
| 5. Jour | Mercredi catastrophique : 4 trades, 25% WR, **-$10,465** |
| 6. Duree | Perdants durent 18.5 barres (1h30) vs gagnants 11.8 barres (1h) |
| 7. Spread regime | Drift **+29.2%** (plus haut en 6 ans), corr **0.664** (plus basse), std **0.071** (plus haute), ACF(1) = **0.000** (zero mean reversion!) |
| 8. Confidence | **INVERSEE** : perdants confidence 51.6% > gagnants 45.8% |
| 9. Yearly comparison | 2023 = anomalie sur TOUTES les metriques (drift, corr, vol, ACF) |
| 10. OLS overlay | Kalman -$5,700 + OLS +$5,860 = **+$160 (break-even)**. Mais 0/11 jours de perte Kalman compenses par OLS |

#### Root cause 2023 :

Le regime macro 2023 (tech rally NQ +38.2% vs value YM +8.9%) a detruit la cointegration :
- Spread drift historique (+29.2%)
- Correlation a son minimum (0.664)
- Volatilite a son maximum (0.071)
- **Zero mean reversion** (ACF1 = 0.000)
- Le confidence scoring est AVEUGLE a ce regime (metriques bar-by-bar, pas de detection de tendance daily)
- Les FLAT exits (force-close 15:30) representent 86% des pertes : le spread diverge, ne revient jamais, position tenue jusqu'a la fermeture forcee

#### Complementarite OLS/Kalman en 2023 :
- 17% de chevauchement de jours de trading (vs 2% global)
- OLS compense au niveau annuel (+$5,860 vs Kalman -$5,700)
- Mais PAS au niveau trade : sur les 11 jours de perte Kalman, OLS ne compense aucun (0/11)
- La complementarite est structurelle (mecanismes differents) mais pas synchrone

### Implications pour Phase 2 (Sierra Charts) :

1. **Overlay discretionnaire** : afficher les metriques de regime (corr rolling, spread drift, ACF) pour que le trader detecte les periodes type 2023
2. **Gestion FLAT** : le trailing stop ou un stop PnL intraday limiterait les pertes de FLAT exit ($19k → cible $5k)
3. **Biais directionnel** : en tech rally, le cote short du spread est structurellement perdant — le discretionnaire doit le filtrer
4. **Confidence scoring** : a completer par un indicateur de regime daily (trend vs mean-reversion) — le scoring actuel est bar-by-bar et aveugle aux trends multi-jours

### Scripts ajoutes Phase 8 :
- `scripts/test_adaptive_r.py` — Test P1 R adaptatif (55 backtests)
- `scripts/analyze_kalman_diagnostics.py` — Analyse P2 + deconfounding P_trace
- `scripts/analyze_2023_losses.py` — Autopsie 2023 (10 dimensions)

---

## Phase 9 — Grid Search NQ/RTY + ES/RTY (2026-02-22)

Objectif : tester si NQ/RTY ou ES/RTY offrent un edge exploitable (complement a NQ/YM).

### Etape 0 — Diagnostic spread

Diagnostics sur 3 paires (Hurst, ADF, correlation, drift, ACF) :
- **NQ/RTY** : Hurst 0.362, ADF significatif 10.5%, Corr 0.609, drift -0.020 → MARGINAL
- **ES/RTY** : Hurst 0.350, ADF significatif 11.4%, Corr 0.734, beta plus stable → MARGINAL
- **NQ/YM** (reference) : Hurst 0.410, ADF significatif 15.6%, Corr 0.910

Les deux paires passent le gate pour l'exploration.

### Etape 1 — Exploration rapide (6 OLS + 6 Kalman par paire)

**NQ/RTY** :
- OLS : 0/6 profitable — non viable
- Kalman : 3/6 profitable, symetrie L/S confirmee (47-53% long)
- Meilleur : K balanced a=3e-7, $9,810, PF 1.76, WR 75.6%

**ES/RTY** :
- OLS : 0/6 profitable
- Kalman : 1/6 a peine positif ($1,620)
- **DECISION : ES/RTY REJET** — spread trop stable pour mean-reversion

### Etape 1b — Exploration approfondie Kalman (18 configs x 5 windows)

**NQ/RTY** : 75/90 = 83% profitable. Champion $53,270 PF 1.34, L/S symetrique.
**ES/RTY** : 23/90 = 25.6% profitable, TOUS avec biais directionnel massif (80-97% long).

**DECISION DEFINITIVE : ES/RTY REJETE** — aucun edge mean-reversion, que du beta directionnel.
**NQ/RTY continue en grid search (Kalman uniquement).**

### Etape 2 — Grid Search Kalman NQ/RTY (856,800 combos)

Script : `scripts/run_grid_kalman_NQ_RTY.py`
- 7 alphas x 17 z_entry x 9 z_exit x 7 z_stop x 8 confidences x 3 profils x 5 fenetres
- Resultat : 856,800 combos en 180s, 413,524 profitable (48.2%)
- CSV : `output/grid_results_kalman_NQ_RTY.csv`

### Etape 3 — Analyse dimensionnelle

Script : `scripts/analyze_grid_NQ_RTY.py`

Sweet spots identifies :
- **Alpha** : 1.5e-7 domine PnL ($20,387 moyen), 5e-7 domine PF (2.97, peu de trades)
- **Window** : 05:00-12:00 unanimement meilleur (PF 2.14, PnL $17,380)
- **z_entry** : 1.8125 sweet spot (PF 2.04), zone plate 1.75-2.0
- **z_exit** : 0.125 maximise PnL ($24k), 1.5 maximise PF (4.64) mais peu de PnL
- **z_stop** : 3.25 domine PnL ($20,613), 2.25 domine PF (2.47)
- **Profile** : moyen domine PF (2.38), tres_court meilleur PnL
- **Confidence** : 75% domine PF (2.54), 50-60% meilleur PnL

Difference cle vs NQ/YM : z_stop optimal 3.25 (vs 2.75 NQ/YM), z_entry plus haut (1.8125 vs 1.375).

### Etape 4 — Top 5 selection + Validation IS/OOS/WF/Permutation

Script : `scripts/validate_top_NQ_RTY.py`

Configs testees :

| Config | alpha | profil | window | ze | zx | zs | conf | Trd | WR% | PnL | PF |
|--------|-------|--------|--------|-----|-----|-----|------|-----|-----|-----|-----|
| K_Balanced | 1.5e-7 | moyen | 05:00-12:00 | 1.8125 | 0.125 | 3.25 | 55 | 171 | 68.4% | $91,200 | 1.75 |
| K_Quality | 2.5e-7 | tres_court | 03:00-12:00 | 1.375 | 1.25 | 3.25 | 70 | 83 | 74.7% | $32,470 | 3.10 |
| K_Volume | 2.5e-7 | court | 05:00-12:00 | 1.3125 | 0.25 | 3.25 | 50 | 355 | 67.0% | $88,020 | 1.32 |
| K_Sniper | 2.5e-7 | moyen | 04:00-13:00 | 1.875 | 0.25 | 3.25 | 70 | 53 | 75.5% | $51,685 | 2.96 |
| K_PropFirm | 2.5e-7 | tres_court | 05:00-12:00 | 1.5625 | 1.5 | 2.25 | 60 | 61 | 90.2% | $25,195 | 4.54 |

**RESULTATS VALIDATION :**

| Config | IS PF | OOS PF | OOS Degrad | Perm p | WF | WF PnL | Verdict |
|--------|-------|--------|------------|--------|-----|--------|---------|
| K_Balanced | 2.09 | 0.76 | -63.6% | 0.000 | 3/5 | $1,285 | MARGINAL |
| K_Quality | 1.93 | 0.71 | -63.2% | 0.000 | 1/5 | -$17,455 | REJETE |
| K_Volume | 1.48 | 0.90 | -39.2% | 0.000 | 2/5 | -$6,940 | REJETE |
| K_Sniper | 4.02 | 0.72 | -82.1% | 0.000 | 3/5 | $12,900 | MARGINAL |
| K_PropFirm | 2.50 | 0.67 | -73.2% | 0.000 | 3/5 | $4,590 | MARGINAL |

**IS/OOS : 0/5 GO** — Toutes les configs ont PF OOS < 1.0.
**Walk-Forward** : 3 configs atteignent 3/5 (seuil minimum), PF WF miserable (1.08-1.65).
**Permutation** : 5/5 GO (p=0.000) — le signal est reel mais ne persiste pas OOS.

**Biais directionnel** :
- K_Balanced : 82% long **BIAS
- K_Sniper : 82% long **BIAS
- K_Volume : 41% long (seule config symetrique)
- K_Quality : 73% long *bias
- K_PropFirm : 74% long *bias

**Annee 2026** : catastrophique pour la plupart des configs (-$6k a -$16k sur 2 mois).
**Exception** : K_PropFirm a toutes les annees positives en full-sample (2021-2026), mais echoue en IS/OOS car le Kalman repart de zero.

### Etape 5 — Complementarite NQ/RTY vs NQ/YM

Script : `scripts/analyze_complementarity.py`

| Combo | PnL total | Trading days | Losing days | Max DD daily | Sharpe daily |
|-------|-----------|-------------|-------------|-------------|-------------|
| NQ_YM + NQ_RTY K_Sniper | $136,510 | 264 | 23% | -$20,780 | 4.49 |
| NQ_YM + NQ_RTY K_PropFirm | $110,020 | 260 | 21% | -$18,070 | 4.24 |
| NQ_YM + NQ_RTY K_Volume | $172,845 | 453 | 30% | -$29,505 | 2.42 |

Points cles :
- NQ_YM et NQ_RTY K_Sniper n'ont que **9% de jours communs** — faible chevauchement
- Correlation PnL jours communs : 0.12 (quasi nulle) — bonne diversification
- **Seulement 2 jours de perte communs** (NQ_YM 52 losing days, K_Sniper 13 losing days)
- 2023 reste la seule annee negative (-$2,785 avec K_Sniper), mais nettement amelioree vs NQ_YM seul

NQ_YM K_Balanced seul : PnL $84,825, 35% long (excellent short bias compensant le long bias NQ_RTY)

### VERDICT FINAL NQ/RTY

**NQ/RTY Kalman : REJETE comme strategie standalone** — echoue IS/OOS (PF OOS < 1.0 sur 5/5 configs), biais directionnel long sur 4/5 configs, 2024+ systematiquement en perte.

**NQ/RTY comme complement discretionnaire : POSSIBLE** — correlation quasi nulle avec NQ_YM, 2 jours de perte communs seulement. K_Sniper ou K_PropFirm peuvent etre affiches en Sierra comme indicateur secondaire pour les jours ou NQ_YM ne trade pas. Mais NE PAS automatiser — utiliser uniquement en overlay discretionnaire.

**ES/RTY : DEFINITIVEMENT REJETE** — aucun edge mean-reversion detecte. Tous les profits proviennent du biais directionnel long (80-97%).

### Scripts ajoutes Phase 9 :
- `scripts/diagnostic_spread.py` — Diagnostics spread 3 paires
- `scripts/explore_pairs.py` — Exploration rapide 6 OLS + 6 Kalman
- `scripts/explore_pairs_v2.py` — Exploration approfondie 18 Kalman x 5 windows
- `scripts/run_grid_kalman_NQ_RTY.py` — Grid search 856,800 combos
- `scripts/analyze_grid_NQ_RTY.py` — Analyse dimensionnelle grid
- `scripts/validate_top_NQ_RTY.py` — Validation complete top 5
- `scripts/analyze_complementarity.py` — Complementarite cross-pair

---

## Phase 10 — Grid Raffine OLS NQ/RTY + Validation (2026-02-23)

Objectif : affiner le grid search OLS sur NQ/RTY avec 17 profils metrics, 6 fenetres, ranges etendus. Selectionner et valider les meilleures configs.

### Differences cles NQ/RTY vs NQ/YM (OLS) :
- **Confidence weights** : HL retire du scoring (ablation : +155% trades, PnL quasi identique) → ADF 50%, Hurst 30%, Corr 20%, HL 0%
- **Profil dominant** : p36_96 (ADF lent 36 + Hurst rapide 96) vs tres_court (NQ/YM)
- **ZW optimal** : 20 bars (1h40) vs 30 bars (NQ/YM) — plus reactif
- **z_entry** : 3.00-3.50 (similaire a NQ/YM 3.15)
- **z_exit** : 0.75 (vs 1.00 NQ/YM) — sort plus tot
- **Multipliers** : MULT_A=20 (NQ), MULT_B=50 (RTY), TICK_A=0.25, TICK_B=0.10

### Etape 1 — Grid raffine (14,374,656 combos)

**Script** : `scripts/refine_ols_balanced.py`
- **OLS** : [2640, 3300, 3960, 5280, 6600, 7920, 9240]
- **ZW** : [16, 20, 24, 28, 36, 48, 60]
- **z_entry** : 2.75 → 3.75 (step 0.25)
- **z_exit** : [0.25, 0.50, 0.75, 1.00, 1.25]
- **z_stop** : [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
- **confidence** : [55, 60, 65, 70, 75, 80]
- **17 profils metrics** : p12_64 a p48_128 (incluant cross-profiles p18_192, p24_256, etc.)
- **6 fenetres** : 02:00-14:00, 04:00-14:00, 06:00-14:00, 08:00-14:00, 08:00-12:00, 06:00-12:00
- **Resultat** : 14,374,656 backtests en ~47 min, 62,037 filtres (PnL>0, PF>1.3, trades>150)

### Etape 2 — Sweet spots identifies

| Dimension | Optimal | PnL moyen | PF moyen | Notes |
|-----------|---------|-----------|----------|-------|
| OLS | 7920-9240 (30-35j) | $10-12k | 1.80-2.05 | Plus long que NQ/YM |
| ZW | 20 (1h40) | $9,200 | 1.82 | Plus reactif que NQ/YM |
| Profil | p36_96 | $10,500 | 1.95 | ADF lent + Hurst rapide |
| Window | 06:00-14:00 | $8,900 | 1.85 | Session europeenne + US |
| z_entry | 3.00 | $9,800 | 1.78 | Zone plate 3.00-3.25 |
| z_exit | 0.75 | $8,500 | 1.85 | Sort tot |
| z_stop | 5.0-5.5 | $9,000 | 1.90 | Plus large que NQ/YM |
| Confidence | 75 | $7,500 | 2.00 | Strict |

### Etape 3 — MaxDD check (15 candidats diversifies)

**Script** : `scripts/check_maxdd_refined_NQ_RTY.py`
- 4/15 SAFE (MaxDD < $8,000), 2 avec MaxDD < $5,000
- High volume (500+ trades) = systematiquement MaxDD $10k-$32k
- Safe configs : 150-205 trades

### Etape 4 — Ranking multi-criteres (78 candidats, 5 groupes)

**Script** : `scripts/rank_top_configs_NQ_RTY.py`
- 78 candidats pre-selectionnes (diversite de patterns)
- 5 groupes : SHARPE, CALMAR, PNL, PROPFIRM, EQUILIBRE
- **3 configs recurrentes** (apparaissent dans 3+ groupes) :
  - A_RTY : OLS=9240, ZW=20, p36_96, 06:00-14:00
  - B_RTY : OLS=7920, ZW=20, p36_96, 06:00-14:00
  - C_RTY : OLS=7920, ZW=20, p42_224, 04:00-14:00

### Etape 5 — Overlap analysis (10 configs)

**Script** : `scripts/compare_candidates_NQ_RTY.py`
- A_RTY vs B_RTY : **74% overlap** (quasi-identiques)
- F_narrow : **11% overlap** avec A+B+C (le plus complementaire)
- D_court : **20% overlap** (pattern different)
- G_sniper : **20% overlap** (pattern different)
- I_volume : 81% overlap → elimine (redondant)

### Etape 6 — Validation complete (6 configs)

**Script** : `scripts/validate_NQ_RTY_top6.py`

#### 6 configs selectionnees :

| Config | OLS | ZW | Profil | Window | ze | zx | zs | conf |
|--------|-----|----|--------|--------|-----|-----|-----|------|
| A_RTY | 9240 | 20 | p36_96 | 06:00-14:00 | 3.00 | 0.75 | 5.0 | 75 |
| B_RTY | 7920 | 20 | p36_96 | 06:00-14:00 | 3.00 | 0.75 | 3.5 | 75 |
| C_RTY | 7920 | 20 | p42_224 | 04:00-14:00 | 3.00 | 0.75 | 3.5 | 75 |
| D_court | 3960 | 28 | p28_144 | 02:00-14:00 | 3.00 | 1.25 | 5.5 | 80 |
| F_narrow | 6600 | 60 | p28_144 | 08:00-12:00 | 3.25 | 0.75 | 5.5 | 80 |
| G_sniper | 3960 | 24 | p16_80 | 06:00-14:00 | 3.50 | 0.50 | 4.5 | 70 |

#### Resultats validation (6/6 GO) :

| Config | Trades | WR% | PnL | Full PF | IS PF | OOS PF | Degrad | Perm p | WF | WF PnL |
|--------|--------|-----|-----|---------|-------|--------|--------|--------|-----|--------|
| A_RTY | 163 | 65% | $19,205 | 2.10 | 1.61 | 3.48 | -116% | 0.000 | 5/5 | $28,530 |
| B_RTY | 189 | 63% | $17,140 | 2.00 | 1.63 | 2.70 | -66% | 0.000 | 4/5 | $25,105 |
| C_RTY | 205 | 62% | $17,370 | 1.98 | 1.92 | 2.34 | -22% | 0.000 | 5/5 | $17,900 |
| D_court | 195 | 59% | $14,710 | 1.95 | 1.71 | 2.25 | -32% | 0.000 | 5/5 | $17,410 |
| F_narrow | 87 | 68% | $9,420 | 1.91 | 2.36 | 1.78 | +25% | 0.000 | 5/5 | $9,000 |
| G_sniper | 152 | 66% | $20,350 | 2.17 | 1.37 | 3.93 | -187% | 0.000 | 4/5 | $23,925 |

Notes :
- **5/6 configs OOS > IS** (degradation negative = OOS meilleur que IS = PAS d'overfitting)
- F_narrow seule config avec degradation normale IS→OOS (+25%)
- **Permutation p=0.000** pour les 6 — signal reel, non reproductible par hasard
- **Walk-Forward** : 4 configs 5/5, 2 configs 4/5 — robustesse confirmee
- Toutes utilisent CONF_WEIGHTS = ADF 50%, Hurst 30%, Corr 20%, HL 0%

### TOP 3 configs selectionnees :

**1. A_RTY** (Champion all-around)
- OLS=9240, ZW=20, p36_96, 06:00-14:00, ze=3.00, zx=0.75, zs=5.0, conf=75
- PF 2.10, WR 65%, $19,205, WF 5/5 ($28,530), OOS PF 3.48
- Meilleur WF PnL, OOS exceptionnel, profil tres stable

**2. G_sniper** (Meilleur PF + PnL)
- OLS=3960, ZW=24, p16_80, 06:00-14:00, ze=3.50, zx=0.50, zs=4.5, conf=70
- PF 2.17, WR 66%, $20,350, WF 4/5 ($23,925), OOS PF 3.93
- Pattern unique (OLS court + z_entry haut + z_exit bas), 20% overlap avec A_RTY

**3. D_court** (Volume + Diversite)
- OLS=3960, ZW=28, p28_144, 02:00-14:00, ze=3.00, zx=1.25, zs=5.5, conf=80
- PF 1.95, WR 59%, $14,710, WF 5/5 ($17,410), OOS PF 2.25
- Pattern tres different (OLS court, ZW moyen, z_exit haut, conf 80), 20% overlap
- Fenetre la plus large (02:00-14:00), capture opportunites hors heures US

### Scripts ajoutes Phase 10 :
- `scripts/refine_ols_balanced.py` — Grid raffine OLS NQ/RTY (14.4M combos)
- `scripts/check_maxdd_refined_NQ_RTY.py` — MaxDD check top candidats
- `scripts/rank_top_configs_NQ_RTY.py` — Ranking multi-criteres (78 candidats, 5 groupes)
- `scripts/compare_candidates_NQ_RTY.py` — Overlap analysis (10 configs)
- `scripts/validate_NQ_RTY_top6.py` — Validation IS/OOS + WF + Permutation (6 configs)

---

## Phase 11 — Deep Validation NQ/RTY OLS (7 etapes) — 2026-02-23

Objectif : pipeline de validation approfondie sur les configs NQ/RTY OLS retenues en Phase 10. 7 etapes : ablation, grid re-analyse, selection parametrique, time stop, validation complete, autopsie, micro/propfirm.

### Etape 1 — Ablation (HL weight)

Confirmation : retrait du poids HL dans confidence scoring (ADF 50%, Hurst 30%, Corr 20%, HL 0%) augmente trades +155% sans degrader PnL.

### Etape 2 — Grid re-analyse + MaxDD

Re-analyse du grid Phase 10 (62k configs filtrees) avec scoring multi-criteres. Ajout tier classification :
- **SAFE** : MaxDD < $5,000 (propfirm compatible)
- **WARN** : MaxDD $5,000-$7,000
- **DANGER** : MaxDD > $7,000

### Etape 3 — Selection parametrique (10 configs)

**Script** : `scripts/step3_maxdd_overlap_NQ_RTY.py`

Remplacement du filtre overlap (50% trading-day overlap trop agressif → n'eliminait tout sauf 4 DANGER) par selection a diversite parametrique :
- Score = 60% base_score + 40% diversity_bonus
- Diversite = distance parametrique sur 8 dimensions (OLS, ZW, profile, window, z_entry, z_exit, z_stop, conf)
- MaxDD weight augmente de 15% a 25%
- Resultat algorithmique : 8 configs (0 SAFE, 2 WARN, 6 DANGER) — SAFE trop similaires au WARN #27
- Ajout manuel de 2 SAFE (#8 et #6) → **10 configs finales**

| Tier | Configs | Labels |
|------|---------|--------|
| SAFE (2) | MaxDD -$4,605 | #8 (06:00-14:00), #6 (04:00-14:00) |
| WARN (2) | MaxDD -$5,410 to -$5,790 | #27, #23 |
| DANGER (6) | MaxDD -$5,115 to -$14,030 | #2, #21, #10, #12, #1, #32 |

### Etape 4 — Time Stop + Hourly Deep-Dive

**Script** : `scripts/step4_timestop_hourly_NQ_RTY.py`

6 valeurs de time stop testees (off, 60min, 90min, 120min, 180min, 240min) + decomposition PnL par heure d'entree CT.

#### Resultats cles :

| Config | Best time stop | Impact |
|--------|---------------|--------|
| #8, #6 (SAFE) | **off** | Trades deja courts (avg 7.1 bars = 35min), time stop inutile |
| **#2 (DANGER)** | **60min** | **PF 2.08→2.37, MaxDD -$8,435→-$5,115** (DANGER→quasi-WARN) |
| #23 (WARN) | 90min | PF 2.15→2.23 (leger) |
| #21 (DANGER) | 90min | PF 1.77→1.80, MaxDD -$10,020→-$9,000 (leger) |
| Autres | off | Aucune amelioration |

**Pattern horaire universel** : 7h-8h CT = money hours (70-80% PnL). 6h CT = TOXIC (perdant).

### Etape 5 — Validation Complete (IS/OOS + Permutation + Walk-Forward)

**Script** : `scripts/step5_validate_NQ_RTY.py`

IS/OOS 60/40, Permutation 1000x, Walk-Forward IS=2y OOS=6m step=6m. Time stops de l'etape 4 appliques.

#### Resultats : **10/10 GO**

| Config | Tier | TS | Trades | WR% | PnL | PF | MaxDD | IS PF | OOS PF | Perm | WF | Neg years |
|--------|------|----|--------|-----|-----|----|-------|-------|--------|------|----|-----------|
| #8 | SAFE | off | 182 | 70.9% | $40,730 | 2.10 | -$4,605 | 1.61 | 3.48 | 0.000 | 5/5 | 2026 |
| #6 | SAFE | off | 199 | 70.4% | $41,500 | 2.08 | -$4,605 | 1.61 | 3.34 | 0.000 | 5/5 | 2026 |
| #27 | WARN | off | 198 | 67.7% | $42,330 | 2.01 | -$5,410 | 1.62 | 2.71 | 0.000 | 4/5 | — |
| #23 | WARN | 18b | 160 | 70.6% | $40,445 | 2.23 | -$5,790 | 1.97 | 2.72 | 0.000 | 5/5 | 2026 |
| #2 | DANGER | 12b | 181 | 72.4% | $40,305 | 2.37 | -$5,115 | 2.14 | 2.66 | 0.000 | 5/5 | 2026 |
| #21 | DANGER | 18b | 341 | 65.7% | $64,670 | 1.80 | -$9,000 | 1.47 | 2.30 | 0.000 | 5/5 | — |
| #10 | DANGER | off | 164 | 68.9% | $40,955 | 2.17 | -$8,045 | 1.56 | 5.20 | 0.000 | 5/5 | — |
| #12 | DANGER | off | 305 | 71.1% | $54,465 | 1.66 | -$13,925 | 1.36 | 2.10 | 0.000 | 3/5 | 2021 |
| #1 | DANGER | off | 413 | 68.0% | $65,255 | 1.61 | -$11,875 | 1.12 | 2.67 | 0.000 | 5/5 | 2021 |
| #32 | DANGER | off | 307 | 70.7% | $53,155 | 1.61 | -$14,030 | 1.37 | 1.90 | 0.000 | 4/5 | 2021 |

Notes :
- **10/10 OOS PF > IS PF** — pas d'overfitting, periode recente (2024-2026) tres favorable
- **Permutation p=0.000** pour les 10 — signal reel
- Walk-Forward 3/5 a 5/5 — robustesse confirmee

### Etape 6 — Autopsie des annees faibles

**Script** : `scripts/step6_autopsy_NQ_RTY.py`

#### Annees negatives par config :

| Annee | Configs touchees | Pattern |
|-------|-----------------|---------|
| 2021 | #32 (-$16K), #10 (-$12K), #1 (-$1.1K) | Regime post-COVID, divergence NQ/RTY |
| 2023 | 6 configs negatives | Annee universellement faible, Jan+Aug toxiques |
| 2024 | **#10 catastrophe cachee** (-$27K) | Single trade 2024-12-18 perd -$32,938 |
| 2026 | 4 configs negatives (2 mois seulement) | Trop tot pour juger |

**SAFE configs (#8, #6)** : annees negatives tres legeres — 2023 max -$792 seulement.

**Config #10 alerte** : step3 montrait 0 neg years (masque par recuperation intra-annee), mais autopsie revele un trade catastrophique unique en Dec 2024. Profil a risque malgre stats globales flatteuses.

### Etape 7 — Micro Contracts + Propfirm

**Script** : `scripts/step7_micro_propfirm_NQ_RTY.py`

#### CRITICAL : Micro NON viable pour NQ/RTY

| Config | Size | Trades | PF | PnL | MaxDD | $/jour | Status |
|--------|------|--------|-----|-----|-------|--------|--------|
| #8 | E-mini | 182 | 2.10 | $40,730 | -$4,605 | $31 | WARN |
| #8 | Mx1 | 182 | 1.26 | $1,214 | -$832 | $1 | SAFE |
| #8 | Mx2 | 182 | **0.70** | **-$2,965** | -$4,520 | -$2 | WARN |
| #6 | E-mini | 199 | 2.08 | $41,500 | -$4,605 | $32 | WARN |
| #6 | Mx1 | 199 | 1.21 | $1,055 | -$851 | $1 | SAFE |
| #6 | Mx2 | 199 | **0.65** | **-$3,602** | -$4,862 | -$3 | WARN |

**Toutes les configs Mx2 ont PF < 1.0 (perdantes).** Commission $2.48 (Mx1) a $4.96 (Mx2) par RT est disproportionnee vs PnL micro NQ/RTY. Contrairement a NQ/YM Kalman (Mx2 PF 1.67, viable), le PnL/trade NQ/RTY OLS est trop faible pour absorber les commissions micro.

**Decision : NQ/RTY = E-mini only.** SAFE configs #8/#6 a MaxDD -$4,605 (92% du trailing DD $5K propfirm).

### Configs production retenues :

**#8 (principale)** : OLS=9240, ZW=20, p36_96, 06:00-14:00, ze=3.00, zx=0.75, zs=5.0, conf=75, ts=off
**#6 (wide window)** : OLS=9240, ZW=20, p36_96, 04:00-14:00, ze=3.00, zx=0.75, zs=5.5, conf=75, ts=off
**#2 (backup)** : OLS=9240, ZW=32, p48_128, 02:00-14:00, ze=3.25, zx=0.75, zs=5.5, conf=80, ts=12 bars

### Scripts ajoutes Phase 11 :
- `scripts/step3_maxdd_overlap_NQ_RTY.py` — Selection parametrique + diversite
- `scripts/step4_timestop_hourly_NQ_RTY.py` — Time stop + hourly deep-dive
- `scripts/step5_validate_NQ_RTY.py` — Validation complete IS/OOS + WF + Permutation
- `scripts/step6_autopsy_NQ_RTY.py` — Autopsie annees faibles
- `scripts/step7_micro_propfirm_NQ_RTY.py` — Micro contracts + propfirm check

---

## Phase 12 — Grid Kalman v2 NQ/RTY + Validation Textbox (2026-02-23)

Objectif : trouver une config Kalman NQ/RTY pour la textbox Sierra (biais discretionnaire), equivalente a K_Balanced pour NQ/YM.

### Root cause echec Phase 9

La Phase 9 (856,800 combos) utilisait `ConfidenceConfig()` par defaut = **poids NQ_YM** (ADF 40%, Hurst 25%, Corr 20%, **HL 15%**). L'ablation Phase 11 avait montre que NQ/RTY necessite ADF 50%, Hurst 30%, Corr 20%, **HL 0%** (+155% trades, PnL identique). Le HL a 15% donnait de la fausse confiance.

Les signaux du grid Phase 9 etaient contamines a la source — les 0/5 GO n'etaient PAS un echec du Kalman NQ/RTY, mais un echec du confidence scoring.

### Grid Kalman v2 (290,304 combos, 5min)

**Script** : `scripts/grid_kalman_v2_NQ_RTY.py --workers 10`

Parametres :
- **Poids corrects** : ADF 50%, Hurst 30%, Corr 20%, HL 0%
- Alpha : [1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 5e-7]
- z_entry : 1.25 -> 2.25 step 0.125 (9 valeurs)
- z_exit : [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
- z_stop : [2.25, 2.5, 2.75, 3.0, 3.25]
- Confidence : [50, 55, 60, 65, 70, 75]
- **7 profils** : tres_court, p16_80, court, p28_144, p36_96, moyen, p48_128
- **4 fenetres** : 03:00-12:00, 04:00-13:00, 05:00-12:00, 06:00-14:00
- **Resultat** : 290,304 combos en 295s, **101,846 profitable** (35%)

### Sweet spots Kalman v2 NQ/RTY

| Dimension | Optimal | Notes |
|-----------|---------|-------|
| Alpha | 1.5e-7 (PnL), 2.5-3e-7 (PF) | 3e-7 domine les configs sniper |
| Window | 05:00-12:00 (PF 1.40) et 06:00-14:00 (PnL) | |
| z_entry | 1.75-1.875 (PF 1.40) | Zone plate |
| z_exit | 1.5 (PF 1.70) et 0.0 (PF 1.60) | Bimodal |
| z_stop | 3.0-3.25 (PnL + volume) | |
| Profil | court (PF 1.40), p36_96 (PF 1.40) | |
| Confidence | 60 (meilleur compromis) | |

### Full analysis top 16 (L/S + yearly + IS/OOS)

2 familles identifiees :
- **Sniper (a=3e-7, zx=1.5)** : 38-62 trades, PF 5-18, WR 83-93%, MaxDD $3.5-5.2K
- **Volume (a=2e-7, zx=0.25)** : 400+ trades, PF 1.28-1.30, L/S 47-48%, MaxDD $22-28K

**16/16 IS/OOS GO** (vs 0/5 Phase 9). **9/16 L/S symetriques** (35-65%).

### Validation complete (7 configs, trades>=50, MaxDD>-$5K)

**Script** : `scripts/validate_kalman_v2_NQ_RTY.py`

IS/OOS 60/40, Walk-Forward IS=2y OOS=6m step=6m, Permutation 1000x.

#### Resultats : **7/7 GO**

| Config | Profil | Window | Trd | WR | PF | MaxDD | L/S | IS PF | OOS PF | WF | Perm |
|--------|--------|--------|-----|-----|-----|-------|------|-------|--------|-----|------|
| **K4_tc** | tres_court | 05:00-12:00 | 51 | 86.3% | 5.07 | -$3,715 | 63% | 2.48 | 22.89 | **6/6** | 0.000 |
| **K1_p16_80** | p16_80 | 05:00-12:00 | 54 | 85.2% | 8.03 | -$3,585 | 65% | 4.52 | 17.89 | 5/6 | 0.000 |
| K2_c55 | p16_80 | 05:00-12:00 | 61 | 83.6% | 5.52 | -$4,855 | 62% | 2.60 | 19.35 | 5/6 | 0.000 |
| K3_s325 | p16_80 | 05:00-12:00 | 55 | 83.6% | 7.47 | -$3,585 | 66% | 4.52 | 13.92 | 5/6 | 0.000 |
| K5_p36 | p36_96 | 04:00-13:00 | 59 | 86.4% | 8.07 | -$3,585 | 68% | 3.03 | 31.45 | 3/6 | 0.000 |
| K6_p36 | p36_96 | 04:00-13:00 | 62 | 83.9% | 6.66 | -$3,585 | 69% | 3.18 | 13.73 | 3/6 | 0.000 |
| K7_p36 | p36_96 | 04:00-13:00 | 58 | 86.2% | 7.55 | -$3,585 | 67% | 3.03 | 28.54 | 3/6 | 0.000 |

#### Yearly K4_tc (champion, 0 neg years) :
2021: $2,455 (3t) | 2022: $6,220 (10t) | 2023: $840 (15t) | 2024: $10,180 (11t) | 2025: $6,515 (8t) | 2026: $3,880 (4t)

#### Yearly K1_p16_80 (runner-up, 0 neg years) :
2021: $955 (2t) | 2022: $7,285 (12t) | 2023: $3,025 (14t) | 2024: $9,575 (9t) | 2025: $5,380 (12t) | 2026: $4,300 (5t)

### Configs textbox retenues :

**K4_tc (champion)** : alpha=3e-7, tres_court, 05:00-12:00, ze=1.75, zx=1.5, zs=3.0, conf=60
- WF 6/6 (seul), L/S 63%/37%, 0 neg years, profil tres_court = plus reactif

**K1_p16_80 (runner-up)** : alpha=3e-7, p16_80, 05:00-12:00, ze=1.625, zx=1.5, zs=3.0, conf=60
- PF 8.03, WF 5/6, yearly tres stable, 0 neg years

### Lecons Phase 12 :

1. **Poids confidence pair-specific = critique**. `ConfidenceConfig()` default = NQ_YM. Toujours passer les poids explicitement pour NQ_RTY.
2. **z_exit=1.5 = regime NQ_RTY Kalman**. Exit quasi au seuil d'entree = trades tres courts, tres fiables.
3. **Profils p36_96 ont biais L/S plus fort** (67-69%) et WF plus faible (3/6) que tres_court/p16_80.
4. **Kalman NQ/RTY valide pour textbox** mais pas standalone trading (~10 trades/an).

### Scripts ajoutes Phase 12 :
- `scripts/grid_kalman_v2_NQ_RTY.py` — Grid Kalman v2 (290K combos, poids corrects)
- `scripts/validate_kalman_v2_NQ_RTY.py` — Validation complete 7 configs
