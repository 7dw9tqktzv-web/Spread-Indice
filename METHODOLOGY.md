# Framework de Validation Config -- Spread Trading

Pipeline reproductible en 8 etapes pour valider une config de spread trading intraday sur futures US.
Developpe et valide sur NQ/YM OLS (Phase 13c, Config D). Applicable a toute paire.

**Source de verite** : ce document decrit la methodologie. CLAUDE.md decrit le code. MEMORY.md decrit les resultats.

---

## Vue d'ensemble

```
Etape 1 : Grid Massif       ->  CSV brut (~15M+ configs)
Etape 2 : Analyse dimensionnelle  ->  3-5 candidats
Etape 3 : Deep Analysis     ->  Overlay + Autopsy + WF recency
Etape 4 : Grid Chirurgical  ->  Config optimale (protection, exits)
Etape 5 : Final Checks      ->  Monte Carlo + Slippage
Etape 6 : Rapport HTML      ->  Documentation complete
Etape 7 : Verdicts          ->  GO / WARN / KILL
Etape 8 : Adaptation        ->  Parametres pair-specific
```

Duree totale estimee : 1-2 jours (grid ~4h, analyse ~2-3h, rapport ~30min).

---

## Prerequis

### Modules requis (`src/`)
- `src/data/cache.py` : `load_aligned_pair_cache()` -- donnees alignees Parquet
- `src/hedge/factory.py` : `create_estimator()` -- OLS/Kalman
- `src/signals/generator.py` : `generate_signals_numba()` -- machine a 4 etats
- `src/signals/filters.py` : `apply_time_stop()`, `apply_window_filter_numba()`
- `src/backtest/engine.py` : `run_backtest_vectorized()` -- backtest + arrays de trades
- `src/validation/gates.py` : `compute_gate_mask()`, `apply_gate_filter_numba()`
- `src/validation/cpcv.py` : `run_cpcv()` -- 45 chemins CPCV(10,2)
- `src/validation/neighborhood.py` : `compute_neighborhood_robustness()`
- `src/validation/propfirm.py` : `compute_propfirm_metrics()`
- `src/validation/deflated_sharpe.py` : `compute_dsr_for_config()`

### Donnees
- Barres 5min alignees (Parquet cache dans `output/cache/`)
- Source : raw/*.txt (Sierra Charts 1min exports, resamplees a 5min)

### Constantes pair-specific
```python
# A adapter par paire :
MULT_A, MULT_B = 20.0, 5.0      # NQ, YM (multipliers contrat)
TICK_A, TICK_B = 0.25, 1.0      # NQ, YM (tick sizes)
COMMISSION = 2.50                 # RT par jambe
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930                    # 15:30 CT en minutes depuis minuit
```

---

## Etape 1 : Grid Massif

**Objectif** : couvrir l'espace de parametres exhaustivement avec CPCV.

**Script reference** : `scripts/phase13c_grid_massif.py`

### Architecture optimisee (critique pour la performance)

```python
Pour chaque OLS_window :                       # 10 valeurs
    OLS estimate -> spread + beta (ONCE)
    Pour chaque ADF_window :                    # 9 valeurs
        compute_gate_mask() -> bool[] (ONCE)
        Pour chaque ZW :                        # 11 valeurs
            Recalcul zscore (ONCE)
            Pour chaque (z_entry, delta_tp, delta_sl, time_stop, window) :
                generate_signals -> apply_filters
                run_backtest_vectorized() ONCE
                run_cpcv() : 45 filtrages du trade array
```

**Cle** : CPCV ne re-run PAS le backtest par chemin. Un seul backtest par config, puis 45 filtrages triviaux du tableau de trades. C'est ca qui rend 24.7M configs faisables en ~4h.

### Parametrisation delta sigma

```python
z_exit = max(z_entry - delta_tp, 0.0)
z_stop = z_entry + delta_sl
```

Avantage : zero exclusions logiques (z_exit < z_entry et z_stop > z_entry garantis par construction). Toutes les combinaisons sont valides.

### Grid axes (reference NQ_YM, a adapter)

| Axe | Valeurs NQ_YM | Notes |
|-----|--------------|-------|
| OLS_window | 2000-8000 (10 pts) | Sweet spot 5000-7000 |
| ADF_window | 12-128 (9 pts) | CRITIQUE : bimodal (30 et 96) |
| ZW | 10-60 (11 pts) | Sweet spot 25-35 |
| z_entry | 2.00-3.75 (8 pts) | 3+ sigma pour NQ_YM |
| delta_tp | 1.00-4.00 (13 pts) | Large, le grid tranchera |
| delta_sl | 0.50-2.50 (8 pts) | |
| time_stop | 0-50 (10 pts) | En barres 5min |
| window | 2 fenetres | 02:00-14:00 et 04:00-14:00 |

### Binary gates (fixes, hors grid)

```python
ADF  : adf_stat < -2.86     # 5% critical value
Hurst: < 0.50                # mean-reversion par definition
Corr : > 0.70                # correlation forte requise
```

Gate rolling windows : ADF variable (dans grid), Hurst=64, Corr=24.

### CPCV(10,2)

```python
CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)
```

- 10 blocs de ~33,000 barres (~6 mois chacun)
- C(10,2) = 45 chemins test
- Purge : 100 barres (~8h) a chaque frontiere (2x conservateur vs duree moyenne trade)
- **Sharpe = mean(pnl) / std(pnl)** -- PAS annualise, pas de sqrt(N)
- Trade attribue si entree ET sortie dans les blocs test (hors purge)

### Parallelisation

```python
ProcessPoolExecutor(max_workers=N)  # groupes par OLS_window
```

Chaque worker charge les donnees et fait son batch autonome. Pas de shared state.

### Sortie

CSV avec colonnes : `ols, adf_w, zw, window, z_entry, delta_tp, delta_sl, time_stop, trades, win_rate, pnl, pf, avg_pnl, max_dd, cpcv_median_sharpe, cpcv_mean_sharpe, cpcv_std_sharpe, cpcv_pct_positive, cpcv_valid_paths`

Filtre : `trades >= 10` (configs avec moins = bruit).

---

## Etape 2 : Analyse Dimensionnelle

**Objectif** : filtrer le CSV brut vers 3-5 candidats viables.

**Script reference** : `scripts/phase13c_analysis.py`

### Filtrage par tiers propfirm

Le filtrage en cascade par tiers est critique pour capturer des profils differents (sniper haute precision vs volume). Appliquer du plus strict au plus large :

```python
TIERS = {
    "Tier 1 (sniper)":  {"trades": 300, "pf": 1.30, "max_dd": -5000, "cpcv_pct": 70},
    "Tier 2 (balanced)": {"trades": 250, "pf": 1.20, "max_dd": -5500, "cpcv_pct": 65},
    "Tier 3 (edge)":     {"trades": 200, "pf": 1.15, "max_dd": -6000, "cpcv_pct": 60},
    "Tier 4 (wide)":     {"trades": 150, "pf": 1.10, "max_dd": -7000, "cpcv_pct": 55},
}
```

Pour chaque tier, filtrer et trier par `cpcv_median_sharpe` decroissant. Les candidats finaux viennent souvent de tiers differents (ex: Config D = Tier 4 sniper, T1 = Tier 1 volume).

### Analyses

1. **Top N par CPCV par tier** : les 10-20 meilleures configs de chaque tier
2. **Scatter par dimension** : cpcv_median vs chaque axe -> identifier les sweet spots
3. **Clusters** : configs proches en parametres = meme "famille"
4. **Deflated Sharpe** : garde-fou, pas critere (FAIL a grande echelle)
5. **Neighborhood** : lookup des voisins (+/-1 step) dans le CSV pour chaque candidat

### Criteres de selection

- CPCV median Sharpe elevee
- CPCV pct_positive > 80% (44+/45 chemins)
- Neighborhood ROBUST (>60% profitable, degradation <50%)
- Propfirm COMPLIANT (MaxDD < $5,000)
- Diversite : au moins 2 "familles" de parametres differentes (tiers differents)

---

## Etape 3 : Deep Analysis

**Objectif** : analyser les candidats sous tous les angles avant de choisir le champion.

**Script reference** : `scripts/phase13c_deep_analysis.py`

### 3a. Overlay temporel (si plusieurs candidats)

Pour chaque paire de candidats, calculer le % de trades simultanes :
```python
overlap = (entry_A < exit_B) & (entry_B < exit_A)  # chevauchement temporel
```

| Overlap | Interpretation |
|---------|---------------|
| > 60% | DOUBLON -- garder le meilleur |
| 30-60% | PARTIAL -- potentiellement complementaires |
| < 30% | COMPLEMENT -- trading independant, diversification |

Verifier aussi : meme direction = redondant, direction opposee = hedge naturel.

**Trades uniques combines** : pour deux configs COMPLEMENT, calculer le nombre de trades uniques si on les execute ensemble. C'est ce qui revele si deux configs complementaires resolvent un probleme de volume (ex: A=124 + D=153 - overlap=24 = 253 trades uniques).

### 3b. Autopsie des trades

Pour chaque candidat, reconstruire le backtest et analyser :

1. **Direction** : % Long vs Short, PnL par direction
2. **Exit types** : classifier chaque trade (priorite : FLAT > TIME_STOP > Z_STOP > Z_EXIT)
   ```python
   if exit at flat_min:     "FLAT"
   elif duration >= ts:     "TIME_STOP"
   elif |z| >= z_stop:      "Z_STOP"
   else:                    "Z_EXIT"
   ```
3. **PnL par exit type** : si time_stop > 15% des exits ET PnL negatif = DESTRUCTIF
4. **Worst trade** : montant + date + contexte
5. **Max consecutive losses** : sequence la plus longue de trades perdants
6. **Avg win / avg loss ratio** : W/L < 1.0 = **WARN, pas KILL** pour mean-reversion.
   **Raisonnement** : en mean-reversion, le sigma se comprime quand le trade revert (le spread revient vers la moyenne = gain modere), mais s'elargit quand il diverge (le spread s'eloigne = perte plus grande). W/L < 1.0 est donc structural, pas un defaut. L'edge est dans le win rate, pas dans le W/L ratio.
   - Breakeven WR = 1 / (1 + W/L ratio). Ex: W/L=0.97 -> breakeven WR = 50.7%
   - Verifier que WR observe > breakeven WR + buffer (>10 pts = SAFE)
   - **Lecon NQ_YM** : Config D a W/L=0.97, WR=68.6%, buffer +18 pts. Reclasse de KILL a WARN.
7. **Calendarite** : PnL par jour de semaine, par heure
8. **Duration** : scatter duree vs PnL, distribution des durees

### 3c. Walk-Forward temporel

WF IS=2 ans, OOS=6 mois, step=6 mois. Genere ~29 combinaisons.

**Mapper chaque fold a ses dates calendrier** pour interpreter les resultats (ex: "fold 2021-H1 = COVID recovery").

**Recency score** : % GO sur les 6 derniers folds (~3 annees les plus recentes).

| Recency | Verdict |
|---------|---------|
| >= 50% | GO |
| 33-49% | WARN |
| < 33% | KILL |

### Decision matrix (seuils FIXES avant de voir les resultats)

```python
THRESHOLDS = {
    "overlap_doublon": 60,        # > 60% = DOUBLON
    "overlap_partial": 30,        # 30-60% = PARTIAL
    "worst_trade_kill": -3500,    # pire trade
    "worst_trade_warn": -2000,
    "max_consec_kill": 7,
    "max_consec_warn": 5,
    "win_loss_warn": 1.5,         # W/L ratio (WARN, pas KILL)
    "recency_go": 50,             # % GO derniers folds
    "recency_warn": 33,
}
```

---

## Etape 4 : Grid Chirurgical

**Objectif** : affiner la config championne sur les axes de protection.

**Script reference** : `scripts/phase13c_surgical_grid.py`

### Quand l'utiliser

- Apres avoir choisi le champion a l'etape 3
- Sur **1 seul config** (pas sur les elimines)
- Focus : time_stop et delta_sl (les deux axes de protection)

### Axes

```python
TIME_STOPS = [0, 12, 15, 18, 20, 25, 30, 40, 50]  # 9 valeurs
DELTA_SLS = [1.00, 1.25, 1.50, 1.75, 2.00]          # 5 valeurs
# Total : 45 configs
```

### Per-config : exit type breakdown

Pour chaque config du grid, calculer le breakdown complet des exits :
```
% exits Z_EXIT, Z_STOP, TIME_STOP, FLAT
PnL par type, WR par type
```

**Critere cle** : si time_stop > 15% des exits ET WR ts_exits = 0% => DESTRUCTIF, ne pas utiliser.

### Phase optionnelle : sweep gate windows

Sur la config optimale, tester 1 axe a la fois :
```python
CORR_WINDOWS = [18, 20, 22, 24, 26, 28, 30]    # 7 configs
HURST_WINDOWS = [32, 48, 64, 96, 128]           # 5 configs
ADF_WINDOWS = [72, 80, 88, 96, 104, 112, 120]   # 7 configs
```

Si marginal (<5% difference), garder les defauts.

---

## Etape 5 : Final Checks

**Objectif** : stress-test final avant validation.

**Script reference** : `scripts/phase13c_final_checks.py`

### 5a. Monte Carlo Drawdown

```python
def run_monte_carlo(pnls, n_sims=10_000):
    rng = np.random.default_rng(42)
    for i in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled)
        running_max = np.maximum.accumulate(equity)
        dd = equity - running_max
        max_dds[i] = dd.min()
    return max_dds
```

**ATTENTION percentiles sur valeurs negatives** :
- `np.percentile(mc_dds, 5)` = worst 5% (le plus negatif)
- `np.percentile(mc_dds, 95)` = best 5% (le moins negatif)
- Pour "P95 worst-case", utiliser `np.percentile(mc_dds, 5)`

| Critere | GO | WARN | KILL |
|---------|-----|------|------|
| Prob(DD < -$5,000) | < 10% | 10-25% | > 25% |

### 5b. Slippage Sensitivity

Tester avec 0, 1, 2, 3 ticks de slippage. 1 tick = defaut backtest.

| Critere | GO | WARN | KILL |
|---------|-----|------|------|
| PF a 2 ticks | > 1.5 | 1.2-1.5 | < 1.2 |

Calculer aussi le breakeven (nombre de ticks ou PF = 1.0).

---

## Etape 6 : Rapport HTML

**Objectif** : document de reference permanent pour la config validee.

**Script reference** : `scripts/phase13c_report_d.py`

### Sections obligatoires

1. **Parametres** : tous les hyperparametres de la config
2. **Metriques globales** : trades, WR, PnL, PF, DD, avg duration
3. **CPCV** : median Sharpe, paths+, distribution des Sharpe par chemin
4. **Equity curve + drawdown** : graphique matplotlib en base64
5. **Distribution temporelle** : PnL par annee, heatmap heure x jour de semaine
6. **Analyse trades** : scatter duree vs PnL, histogramme PnL, exit types
7. **Propfirm** : compliance, max daily loss, trailing DD
8. **Voisinage** : tableau des voisins, % profitable, degradation
9. **Monte Carlo** : distribution DD, percentiles, prob breach
10. **Slippage** : tableau 0-3 ticks, PF/CPCV/paths+

Sortie : `output/<PAIR>/config_<NAME>_reference.html`

---

## Etape 7 : Verdicts

Tableau de decision final. Tous les seuils doivent etre definis AVANT de voir les resultats.

| Test | GO | WARN | KILL |
|------|----|------|------|
| CPCV paths+ | > 80% | 60-80% | < 60% |
| Voisinage profitable | > 60% | 40-60% | < 40% |
| Voisinage degradation | < 50% | 50-70% | > 70% |
| WF recency (6 derniers) | >= 50% | 33-49% | < 33% |
| Monte Carlo breach $5K | < 10% | 10-25% | > 25% |
| Slippage 2 ticks PF | > 1.5 | 1.2-1.5 | < 1.2 |
| Worst trade | > -$2,000 | -$2k to -$3.5k | < -$3,500 |
| Max consecutive losses | <= 4 | 5-6 | >= 7 |
| W/L ratio (mean rev) | > 1.0 | 0.70-1.0 (si WR > 60%) | < 0.70 |

**Decision** :
- Aucun KILL = **GO**
- 1+ KILL = **REVIEW** (analyser si structurel ou corrigeable)
- 3+ WARN = **REVIEW**

---

## Etape 8 : Adaptation par Paire

Pour appliquer ce framework a une nouvelle paire :

### Parametres a modifier

| Parametre | Ou le changer | Notes |
|-----------|--------------|-------|
| MULT_A, MULT_B | Constantes script | Multipliers du contrat |
| TICK_A, TICK_B | Constantes script | Tick sizes |
| COMMISSION | Constantes script | Commission RT |
| FLAT_MIN | Constantes script | Heure flat (930 = 15:30 CT) |
| Instrument enum | `SpreadPair(leg_a=..., leg_b=...)` | |
| Grid ranges | Axes du grid | Adapter aux sweet spots connus |

### Sweet spots connus par paire

| Paire | OLS_window | ZW | z_entry | Notes |
|-------|-----------|-----|---------|-------|
| NQ/YM | 5000-7000 | 25-35 | 3.00-3.50 | ADF_w bimodal (30, 96) |
| NQ/RTY | 7000-10000 | 15-25 | 2.75-3.25 | OLS plus long, ZW plus reactif |

### Gate thresholds et windows

Les seuils ADF < -2.86, Hurst < 0.50, Corr > 0.70 sont bases sur la theorie statistique (5% ADF, definition mean-reversion, correlation forte). Ils devraient etre universels, mais verifier sur chaque paire que les gates ne bloquent pas >80% des barres (sinon trop restrictif).

**Gate windows = pair-specific, TOUJOURS dans le grid.** C'est la lecon principale de Phase 13a -> 13c : fixer ADF_window a 24 barres semblait raisonnable mais rendait la config fragile (changer a 48 = PF 2.0 -> 1.2). En mettant ADF_window dans le grid (12-128), on a decouvert que ADF_w=96 etait optimal pour NQ_YM -- pas 24. Sur NQ_RTY, la valeur optimale sera probablement differente. Hurst_window et Corr_window sont moins sensibles mais meritent un sweep en Etape 4 (grid chirurgical).

### Pieges connus

1. **Confidence weights pair-specific** : si on utilise le scoring continu (Phases 6-12), les poids ADF/Hurst/Corr/HL sont pair-specific. NQ_YM = 40/25/20/15, NQ_RTY = 50/30/20/0.
2. **Micro contracts** : PnL/trade plus petit = commission disproportionnee. Tester Mx1 et Mx2 explicitement.
3. **ADF window sensitivity** : TOUJOURS mettre dans le grid, jamais fixer a priori.
4. **Time stop** : peut etre DESTRUCTIF sur certaines paires. Toujours verifier avec exit type breakdown.
5. **W/L ratio < 1.0** : structural pour mean-reversion. Ne pas eliminer sur ce critere seul. Verifier le buffer WR > breakeven WR.

---

## Reference : Config D NQ_YM (validee)

La premiere config produite par ce framework. Parametres et resultats complets dans CHANGELOG.md Phase 13c.

```
OLS=7000, ADF_w=96, ZW=30, 02:00-14:00
ze=3.25, zx=0.50, zs=4.75, ts=0
Gates: ADF<-2.86, Hurst<0.50, Corr>0.70
153 trades, PF 2.13, WR 69%, PnL $24,880, DD -$4,595
CPCV 0.265 median, 97.8% paths+
Monte Carlo 90% safe, Slippage PF 1.85 at 2 ticks
Verdict: GO
```
