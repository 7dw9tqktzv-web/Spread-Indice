# Expert Engine — Knowledge Base

## Architecture Actuelle (`src/backtest/engine.py`)

### 3 modes de backtest
1. **`BacktestEngine`** : bar-by-bar loop Python, reference, single backtests. Mark-to-market equity.
2. **`run_backtest_vectorized`** : numpy vectorise, avec equity curve numba. Pour validation individuelle.
3. **`run_backtest_grid`** : lightweight, pas d'equity curve. Pour grid search (20x plus rapide).

### Shared helpers (deja factorise)
- `_detect_and_pair_trades(sig)` : detecte transitions signal, paire entries/exits
- `_compute_trades(...)` : sizing, slippage, dollar stop, PnL vectorise
- `_compute_summary_stats(pnl_net, durations)` : metriques resumees
- `_build_equity_curve(...)` : numba JIT, mark-to-market
- `_apply_dollar_stop(...)` : numba JIT, avance les exit bars si unrealized < -dollar_stop

### Signal generator (`src/signals/generator.py`)
- `generate_signals_numba()` : 4-state machine (FLAT/LONG/SHORT/COOLDOWN), numba JIT
- Input: z-score array + thresholds. Output: int8 array {+1, 0, -1}
- COOLDOWN = etat interne, emet 0 (pas de signal)

### Pipeline
```
AlignedPair (cache) → OLS/Kalman (hedge) → Metrics (ADF/Hurst/HL/Corr)
→ SignalGenerator (4-state) → GateFilter / ConfidenceFilter → WindowFilter
→ BacktestEngine → PerformanceMetrics
```

### Sizing
Dollar-neutral beta-weighted: `N_b = round((Not_A / Not_B) × |beta| × N_a)`, min 1.
Micro: `find_optimal_multiplier()` teste 1x..Nx pour minimiser erreur d'arrondi.

---

## Pattern: Moteur Hybride (Reference GC/SI)

### Architecture deux couches
- **Couche lente** (1min ou 5min) : calcul indicateurs (z-score, beta, gates, correlation, hurst)
- **Couche rapide** (5s ou 1s) : scanning PnL intra-trade pour dollar exits precis
- Les indicateurs ne sont JAMAIS recalcules sur la couche rapide
- Seuls les prix bruts de la couche rapide servent au calcul PnL

### Boucle principale (pseudo-code)
```python
for i in range(n_slow_bars):
    # 1. Si en position → scanner les barres rapides entre [t_prev, t_curr]
    #    a. SL_DOLLAR (priorite max)
    #    b. TP_DOLLAR
    # 2. Si pas de trigger rapide → verifier sorties z-score sur barre lente
    #    a. SL_ZSCORE
    #    b. TP_ZSCORE
    #    c. MAX_HOLD (bars en position)
    #    d. FLAT_EOD (fin de session)
    # 3. Cooldown reset
    # 4. Check entrees (filtre horaire + regime + quality)
```

### Cursor 5s/1s (pattern cle)
```python
idx_5s_start = 0  # avance lineairement, jamais en arriere

# Dans la boucle lente:
while idx_5s_start < len(dt_5s) and dt_5s[idx_5s_start] < t_prev:
    idx_5s_start += 1  # skip barres deja traitees

j = idx_5s_start
while j < len(dt_5s) and dt_5s[j] <= t_curr:
    pnl = calc_pnl_inline(...)
    if pnl <= SL: exit; break
    if pnl >= TP: exit; break
    j += 1
```
**Complexite** : O(n_fast) total, PAS O(n_slow × n_fast). Le curseur avance une seule fois.

### Dollar exit → PnL fixe au seuil
Quand SL_DOLLAR trigger : `pnl_gross = pnl_sl` (valeur exacte du seuil, PAS le prix 5s).
Meme logique pour TP_DOLLAR. Plus realiste (slippage deja inclus dans le seuil).

### Reentry sur meme barre
Apres un dollar exit + cooldown reset, le code verifie les conditions d'entree sur la MEME barre lente.
Permet de capturer un retournement immediat sans attendre la barre suivante.

### MFE/MAE tracking
`max_pnl_intra` et `min_pnl_intra` trackes sur TOUTES les barres rapides (pas juste les barres lentes).
Donne le vrai Maximum Favorable/Adverse Excursion pour analyse post-backtest.

### Pure z-score mode
Si `pnl_tp >= 50000 and pnl_sl <= -50000` : skip le scan rapide entierement.
Optimisation quand seuls les z-score exits sont actifs (pas de dollar exits).

---

## Pattern: Config Vector Numba (Reference GC/SI)

### Probleme resolu
Les kernels Numba ne supportent pas les dicts. Ajouter un parametre = changer la signature du kernel = recompilation + bug-prone.

### Solution: Config Vector
```python
# --- Index des parametres (constantes globales) ---
CFG_Z_ENTRY = 0
CFG_Z_EXIT  = 1
CFG_Z_STOP  = 2
CFG_PNL_TP  = 3
CFG_PNL_SL  = 4
# ... ajouter ici
CFG_SIZE    = 35  # incrementer

# --- Packing: dict → numpy array ---
def pack_config(config: dict) -> np.ndarray:
    cfg = np.zeros(CFG_SIZE, dtype=np.float64)
    cfg[CFG_Z_ENTRY] = float(config["z_entry"])
    cfg[CFG_Z_EXIT]  = float(config["z_exit"])
    # ... 1 ligne par param
    return cfg

# --- Kernel: lit cfg[IDX] ---
@njit(cache=True)
def _run_kernel(data_arrays..., cfg):
    z_entry = cfg[CFG_Z_ENTRY]  # acces direct
```

### Regles d'extension
1. Ajouter `CFG_XXX = N` (incrementer)
2. Ajouter 1 ligne dans `pack_config()`
3. Utiliser `cfg[CFG_XXX]` dans le kernel
4. Incrementer `CFG_SIZE`
5. **La signature du kernel ne change JAMAIS**

### Helpers Numba inlines
```python
@njit(cache=True, inline='always')
def _check_entry(z, corr, score, state, hurst, cfg):
    ...
```
- `inline='always'` : pas d'overhead d'appel de fonction
- Toutes les petites fonctions (entry, exit, costs, sizing, pnl, record_trade) sont inlinees
- Le kernel principal les appelle comme des fonctions normales mais Numba les fusionne

### Resultats pre-alloues
```python
_MAX_TRADES = 5000
results = np.empty((_MAX_TRADES, 17), dtype=np.float64)
# ... remplir trade par trade ...
return results[:trade_no], trade_no  # slice final
```
- Pre-allocation fixe (pas de list.append() dans numba)
- Warning si MAX_TRADES atteint

### Etats et raisons comme entiers
```python
_FLAT = 0; _LONG = 1; _SHORT = -1; _CD_LONG = 2; _CD_SHORT = -2
_TP_ZSCORE = 0; _SL_ZSCORE = 1; _TP_DOLLAR = 2; _SL_DOLLAR = 3
EXIT_REASON_MAP = {0: 'TP_ZSCORE', 1: 'SL_ZSCORE', ...}  # mapping inverse
```
- Numba ne gere pas les strings → entiers dans le kernel, mapping dans le wrapper

### Timestamps int64
```python
ts_ns = pd.to_datetime(df['DateTime']).values.astype(np.int64)
# Comparaison dans numba: dt_5s_ns[j] <= t_curr (int64 vs int64)
```
- Pandas datetime → int64 nanoseconds pour passage a numba

### Warmup
```python
def warmup_numba():
    """Pre-compile le kernel avec dummy data (~3-5s)."""
    cfg = np.zeros(CFG_SIZE, dtype=np.float64)
    cfg[CFG_GC_PV] = 10.0; ...
    _run_kernel(dummy_arrays..., cfg)
```
- Appeler 1 fois au demarrage du script
- Evite la latence de compilation au premier vrai appel

### Wrapper Python
Le kernel retourne raw numpy → le wrapper Python convertit en DataFrame avec noms de colonnes, timestamps, direction strings, exit reasons, PnL_Cumul, etc.

---

## Pattern: Grid sur courbes 1s pre-calculees (grid_dollar_tpsl.py)

### Architecture en 4 etapes
1. **Reconstruct** : relancer le backtest 5min, recuperer entries/exits/sides/pnls
2. **Load 1s** : charger CSV 1s uniquement pour les dates de trades (chunked, ~2M rows/chunk)
3. **Pre-compute curves** : pour chaque trade, calculer la courbe `unrealized[t]` en 1s
4. **Grid** : boucle TP×SL, pour chaque config scanner les courbes → `np.where(curve >= TP)`

### Pattern de chargement 1s optimise
```python
for chunk in pd.read_csv(fname, chunksize=2_000_000):
    date_mask = chunk["Date"].isin(trade_dates)  # filtre par dates de trades
    if not date_mask.any(): continue
    # ... filtre par intervalle [entry-1min, exit+6min]
```
- Ne charge que les dates avec des trades
- Buffer temporel autour de chaque trade pour couvrir le decalage bar_timestamp vs bar_close
- **Bug corrige** : `+5min` offset car l'entree se fait au CLOSE du bar (timestamp = debut du bar)

### Sanity check
```python
implied_pnl = curve[-1] - exit_cost
actual_pnl = trade["pnl_5min"]
err = abs(implied_pnl - actual_pnl)
```
Verifie que le PnL 1s en fin de trade ≈ PnL 5min original. Ecart = slippage de discretisation.

---

## Optimisations Appliquees

### Hurst Rolling — x300 faster
- Avant: boucle Python 331k × hurst_exponent() = ~13 min
- Apres: precompute rolling std par lag (pandas C), puis polyfit loop = 2.7s
- Fichier: `src/stats/hurst.py`

### Half-life Rolling — x400 faster
- Avant: boucle Python 331k × np.linalg.lstsq() = 39s
- Apres: rolling Cov/Var vectorise = <0.1s
- Methode: `b = Cov(Z(t),Z(t-1)) / Var(Z(t-1))`, `HL = -ln(2)/ln(b)`
- Fichier: `src/stats/halflife.py`

### Signal Generator — x451 (numba JIT)
- 4-state machine compilee par numba
- `@njit(cache=True)` → pas de recompilation entre sessions

### Grid Search — hedge ratio factorise
- Beta/spread calcules 1 seule fois par ols_window (pas N× pour chaque zscore_window)
- Z-score recalcule a partir du spread existant: `(spread - rolling_mean) / rolling_std`

### Performance.py — warnings supprimes
- `np.errstate(divide="ignore", invalid="ignore")` pour eviter spam I/O
- `np.nan_to_num()` pour returns quand equity = 0

---

## Bottlenecks & Solutions

### RESOLU — Backtest Grid
- `run_backtest_grid` : pas d'equity curve, detection vectorisee des transitions signal
- Grid 24.7M combos en 3h50m (20 workers)

### A FAIRE — Moteur Hybride 1s
- Notre engine actuel ne supporte que bar-close entries/exits
- Le moteur hybride doit combiner couche 5min (indicateurs) + couche 1s (scanning z_live)
- **Differences vs GC/SI** : notre scanning 1s sera sur le z-score (pas dollar PnL)
  - `spread_1s = ln(NQ_1s) - beta_5min × ln(YM_1s) - alpha_5min`
  - `z_live = (spread_1s - mu) / sigma`
  - Entry: z_live franchit ze → entre immediatement
  - Exit: z_live franchit zx/zs → sort immediatement
- Le dollar stop reste en surcouche optionnelle (mais deja prouve DESTRUCTIF)

### A FAIRE — Config Vector pour grid 1s
- Adapter le pattern CFG_XXX de GC/SI pour notre projet
- Permettra de faire un grid search sur le moteur hybride 1s sans changer la signature

---

## Erreurs Passees & Corrections

### Sizing bug grid_dollar_tpsl.py
- Bug: `n_b = abs(b) * MULT_A / MULT_B` (ratio des multipliers, FAUX)
- Fix: `n_b = (px_a*MULT_A) / (px_b*MULT_B) * |beta|` (ratio des notionnels × beta)

### Timing offset 1s
- Bug: bar timestamp = DEBUT du bar, entry au CLOSE = timestamp + 5min
- Fix: `start = entry_time + pd.Timedelta(minutes=5)` pour aligner 1s avec le moment reel d'entree

### Double-buffer session
- Sierra 17:30-15:30 EST deja le buffer vs Globex 17:00-16:00
- `buffer_minutes=30` ajoutait un 2e buffer → 18:00-15:00 (perdait 14k barres)
- Fix: `buffer_minutes=0`

### ADF p-value vs statistic
- `adf_rolling()` retourne des p-values (0-1), `adf_statistic_simple()` retourne des statistiques (~-3.5)
- Le filtre comparait a -2.86 (seuil de statistique) avec des p-values → filtre mort

### R/S Hurst biaise
- R/S sur niveaux de spread → cumsum d'un cumsum → H toujours ~0.99
- Fix: variance-ratio sur niveaux directement → H median 0.41

### stdout bufferise en multiprocessing
- `ProcessPoolExecutor` buffer stdout → logs n'apparaissent pas en temps reel
- Fix: `sys.stdout.flush()` apres chaque log

---

## Checklist: Implementation Moteur 1s

### Infra existante a reutiliser
- `scripts/grid_dollar_tpsl.py` : `load_1s_for_trades()` pour charger CSV 1s
- `scripts/mfe_mae_analysis.py` : `load_1s()` alternative
- `src/signals/generator.py` : `generate_signals_numba()` (4-state machine)
- `src/signals/filters.py` : `apply_gate_filter_numba()`, `apply_window_filter_numba()`
- `src/backtest/engine.py` : `_detect_and_pair_trades()`, `_compute_trades()`

### Ce qui doit etre code
1. **`run_hybrid_1s_backtest()`** : boucle principale couche 5min + scanning 1s
2. **Config vector** pour le kernel numba
3. **Load/align 1s data** : NQ + YM synchronises, forward-fill gaps
4. **Z-score 1s** : `spread_1s = ln(NQ_1s) - beta × ln(YM_1s) - alpha`, `z = (s - mu) / sigma`
5. **Comparaison directe** : Config D 5min vs Config D 1s (memes params)
