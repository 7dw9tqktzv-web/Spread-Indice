# Pipeline V1 -- Reference Complete

## Philosophie Globale

### BLOQUANT vs OBSERVATION
- **BLOQUANT** : modele mathematiquement invalide -> arret complet
- **OBSERVATION** : metrique stockee -> score de confiance -> decision trader

L'objectif est de CONTROLER les biais, pas de les supprimer.
Le pipeline qui bloque tout n'est pas robuste -- c'est un pipeline qui ne trade jamais.

### Score de Confiance
Chaque paire accumule un score issu des observations (etapes 2-5).
Seuil minimum parametre et calibre sur donnees historiques.

---

## Etape 1 -- Data Validation & Cleaning

### Aggregation 5min Session-Aware
```python
# CORRECT : groupby session AVANT resampling
df.groupby('session_id').resample('5min').agg({...})
# INTERDIT : traverse les sessions
df.resample('5min')  # <- ne jamais utiliser
```
Barre finale incomplete : supprimee si < 3 barres 1min sur 5.

### Session markers
```
session_id = YYYYMMDD
session_break = True  # premiere barre de chaque session (17h30 CT)
REGLE ABSOLUE : aucun calcul ne traverse ce flag
log_return = NaN sur session_break OBLIGATOIREMENT
```

### Price Type par actif
- HIGH (>100K ADV) : ES, NQ, CL, NG, GC, HO, RTY, RB, ZC, ZW -> Close
- MID (20-80K) : SI, YM, HG, PL -> Typical Price (H+L+C)/3
- LOW (~5K) : PA -> Typical Price, seuil low_liq 50%

### Back-adjustment Sierra Chart
Convention ADDITIVE : `back_adjust = settlement_nouveau(J-1) - settlement_ancien(J-1)`
Biais en log < 1% sur 30j, absorbe par alpha_OLS.

---

## Etape 2 -- Tests de Stationnarite (ADF + KPSS)

Objectif : valider que chaque actif est I(1).
Downsampling proportionnel -> target ~1000 obs.
Multi-scale : 3 frequences, I(1) confirme sur >= 2 dont la plus basse.
**BLOQUANT** : actif non-I(1) sur 30j ET 60j, confirme sur 2+ frequences.

---

## Etape 3 -- Cointegration OLS + AR(1)

### Construction du spread
```
log(A)_t = alpha + beta * log(B)_t + epsilon_t
Spread_t = log(A)_t - alpha_OLS - beta_OLS * log(B)_t
```
Definition canonique AVEC alpha. Z-score = Spread_t / sigma_eq.
Asymetrie : tester A->B et B->A, retenir le plus stationnaire.
Valeurs critiques MacKinnon (pas DF standard).

### Test AR(1) sur le spread
```
Spread_t = c + phi * Spread_{t-1} + eta_t
H0 : phi = 1 (non stationnaire)
t_DF = (phi_hat - 1) / SE(phi_hat)  # distribution Dickey-Fuller
```

### Sorties vers etape 4
beta_OLS, alpha_OLS, phi_hat, c, sigma_eta, SE(alpha), SE(beta), mu_B

---

## Etape 4 -- Ornstein-Uhlenbeck : Qualification

### Conversion AR(1) -> OU
```
kappa = -ln(phi) / dt
theta_OU = c / (1 - phi)            # ~ 0 par OLS
sigma_eq = sigma_eta / sqrt(1 - phi^2)    # denominateur Z-score
sigma_diffusion = sigma_eta * sqrt(-2*ln(phi) / (1 - phi^2))  # usage interne
```

### Assertions BLOQUANTES
```
assert phi in ]0, 1[
assert kappa > 0
assert sigma_eq > 0
assert isfinite(sigma_eq)
assert Ratio_HL <= 3.0
assert abs(theta_OU) < 0.01 * sigma_eq
```

### Cross-check sigma_eq
```
z_modele = (Spread_t - theta_OU) / sigma_eq
z_empirique = (Spread_t - mean(Spread)) / std(Spread)
assert median(|Delta_z|) < 0.02 AND percentile_99(|Delta_z|) < 0.15
```

### Q_OU -- Usage interne UNIQUEMENT, ne sort PAS vers etape 5

---

## Etape 5 -- Kalman, Risk Management & Execution

### Flux par barre
```
[EVENT]      BarClose(t)
[PARALLELE]  Phase 2 (Signal) + Phase 3 (Kalman)
[BARRIERE]   -> creation BarState_t IMMUABLE
[SEQUENTIEL] Phase 4 lit BarState_t -> produit FilterVerdict_t
[CONDITIONNEL] Phase 5 lit BarState_t + FilterVerdict_t -> sizing
```

### Phase 1 -- Init Session (17h30 CT)
```python
x_0 = [alpha_OLS, beta_OLS]
P_0 = diag(max(SE(alpha)^2, 1e-4), max(SE(beta)^2, 1e-4))
is_session_killed = False
beta_kalman_prev = beta_OLS  # evite faux Filtre B sur gap overnight
```

### Phase 2 -- Signal Engine (Monde Statique)
sigma_eq FIXE toute la session. N'accede JAMAIS a beta_Kalman.
```
Spread_t = log(A)_t - alpha_OLS - beta_OLS * log(B)_t
Z_t = (Spread_t - theta_OU) / sigma_eq
```

**Machine a etats LONG (acheter A, vendre B) :**
```
ARMEMENT     : Z_t < -2.5 -> is_armed_long = True
DECLENCHEMENT: is_armed_long ET Z_t > -2.0 -> Signal LONG
SUR Z_t < -3.0 : position ouverte -> STOP LOSS / arme -> DESARMEMENT
TAKE PROFIT  : |Z_t| < 0.5 -> fermer
```

**Machine a etats SHORT (vendre A, acheter B) :**
```
ARMEMENT     : Z_t > +2.5 -> is_armed_short = True
DECLENCHEMENT: is_armed_short ET Z_t < +2.0 -> Signal SHORT
SUR Z_t > +3.0 : STOP LOSS / DESARMEMENT
TAKE PROFIT  : |Z_t| < 0.5 -> fermer
```

**Time-Lock :**
```
T_limite = min(T_close_session - HL_empirique, T_close_pit_paire)
Si Heure >= T_limite : desarmement force
```

### Phase 3 -- Kalman Engine (tache de fond permanente)
Etat cache : x_t = [alpha_t, beta_t]
```
H_t = [1, log(B)_t]
e_t = log(A)_t - H_t * x_pred
S_t = H_t * P_pred * H_t' + R
K_t = P_pred * H_t' / S_t
x_t = x_pred + K_t * e_t
P_t = (I - K_t*H_t) * P_pred * (I - K_t*H_t)' + K_t*R*K_t'  # JOSEPH
NIS_t = e_t^2 / S_t
```

### Phase 4 -- Risk Manager (Filtres A/B/C)
Lit BarState_t, produit FilterVerdict_t. Ne modifie JAMAIS BarState_t.

**Filtre A -- Qualite Innovation (NIS) :** SUSPENSION TEMPORAIRE
```
filtre_A_ok = (NIS_t < 9.0) AND (rolling_NIS_20 < 3.0)
```

**Filtre B -- Vitesse Beta :** SUSPENSION TEMPORAIRE
```
filtre_B_ok = |beta_K_t - beta_K_{t-1}| / |beta_K_{t-1}| < 0.005
```

**Filtre C -- Derive Macro :** COUPE-CIRCUIT DEFINITIF
```
sigma_derive = sqrt(264 * Q_beta_classe)
seuil_C_abs = 4 * sigma_derive
filtre_C_ok = |beta_Kalman_t - beta_OLS| < seuil_C_abs
Si filtre_C_ok == False : is_session_killed = True (IRREVERSIBLE)
```

Seuils Filtre C par classe :
| Classe | Q_beta | seuil_C_abs |
|--------|--------|-------------|
| Metals | 1e-7 | 0.02056 |
| Equity | 2e-7 | 0.02906 |
| Grains | 2e-7 | 0.02906 |
| Energy | 5e-7 | 0.04597 |

**Arbre de decision :**
- Entree : filtre_A AND filtre_B AND NOT is_session_killed
- Sortie TP/SL : IMMEDIATEMENT (filtres ignores)
- Filtre C + position : Backtest auto / Reel alerte

### Phase 5 -- Sizing Beta-Neutral
```
notional_A = raw_price_A * (Q_A_std * multiplier_A_std + Q_A_micro * multiplier_A_micro)
target_notional_B = notional_A * abs(beta_Kalman_t)
Q_B_total_micros = max(round(target_notional_B / (raw_price_B * multiplier_B_micro)), 1)
Q_B_std = Q_B_total_micros // ratio_B
Q_B_micro = Q_B_total_micros % ratio_B
```

**DISTINCTION CRITIQUE : multiplier != tick_value**
```
multiplier = tick_value / tick_size
notional = Prix * multiplier = 2000 * 100 = $200,000  (GC)
notional = Prix * tick_value = 2000 * 10 = $20,000     (FAUX, facteur 10)
```

---

## Regles Anti-Look-Ahead

1. Fenetre calibration = [T-30 sessions propres ... T-1 a 15h30 CT]
2. Parametres OLS/OU figes AVANT premier tick de session T
3. Kalman reinitialise a x_0 = [alpha_OLS, beta_OLS] a 17h30 CT
4. Fenetre trading : [T 17h30 ... T 15h30 CT], burn-in ~90 barres
5. PnL calcule sur beta_Kalman_ENTREE (pas _SORTIE)
6. Trois colonnes PnL_net : 1x, 1.5x, 2x slippage

---

## Couts de Transaction

### Convention RT (Round-Trip = entree + sortie)
```python
cout_RT(sym, mult) = COMM_RT[sym] + SLIP_RT[sym] * slippage_multiplier
spread_cost_rt(legs, mult=1.0) = sum(qty * cout_RT(sym, mult))
pnl_net = pnl_brut - spread_cost_rt(legs, mult)  # UNE SEULE soustraction
```

### Analyse de sensibilite obligatoire
```
pnl_net_1x   = pnl_brut - spread_cost_rt(legs, 1.0)     # base
pnl_net_1.5x = pnl_brut - spread_cost_rt(legs, 1.5)     # robustesse
pnl_net_2x   = pnl_brut - spread_cost_rt(legs, 2.0)     # pessimiste
Critere minimal : Sharpe_1.5x > 0
Critere resilience : Sharpe_2.0x > 0
```

### Couts RT par paire (config minimale)
| Paire | Total RT |
|-------|---------|
| GC/SI (std) | $78.40 |
| GC/SI (micro) | $53.70 |
| NQ/RTY | $27.60 |
| CL/HO | $36.40 |
| CL/NG | $48.20 |
| ZC/ZW | $60.60 |
| GC/PA | $128.40 |

---

## Metriques de Performance

| Metrique | Seuil V1 |
|----------|---------|
| Sharpe (PnL_net_daily) | > 1.0 a 1.5x slip |
| Win Rate | > 55% |
| Avg Win/Loss | > 1.2 |
| Max Drawdown | < 20% capital |
| Freq Sortie Forcee | < 15% |
| Slippage Robustness | Sharpe_1.5x > 0 |
