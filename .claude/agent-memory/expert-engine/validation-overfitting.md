# Validation, Overfitting & Anti-Biais -- Knowledge Base

## 1. Overfitting -- Le Plus Grand Probleme en Finance Quantitative

### Definition
```
D = D_T + epsilon
```
D = donnees observees, D_T = signal vrai, epsilon = bruit aleatoire.
L'overfitting = ajuster le modele au BRUIT au lieu du signal.

### Deux causes principales
1. Petit echantillon : bruit et tendance indistinguables
2. Modele trop complexe : se contorsionne pour fitter le bruit

### Manifestations en finance
- **Trop de parametres** : "Il vaut mieux expliquer 60% des donnees avec 2-3 params que 90% avec 10"
- **Fit parfait = red flag** : "There is almost always noise in real data, a perfect fit is almost always overfitting"
- **Comparaisons multiples** : 190 paires testees -> ~10 "significatives" par hasard (p=0.05)
- **Optimisation de fenetre** : tester toutes les fenetres et prendre la meilleure = overfitting
- **Look-ahead bias** : calculer beta/z-score sur TOUTE la serie = utiliser info future

### Polynomial curve fitting (analogie)
- Lineaire : sous-fit
- Quadratique : bon fit
- Polynome degre 9 : passe par chaque point mais oscille aux extremites (prediction desastreuse)

---

## 2. Solutions Anti-Overfitting

### 2.1 Parcimonie des parametres
Moins de parametres = moins de chances d'overfitter.

### 2.2 Out-of-sample testing
"The most important way to avoid overfitting: test on data not used in constructing the model."

### 2.3 PIEGE : Abuser de l'out-of-sample
"Repeatedly fit and compare on the same OOS data defeats the purpose."
Chaque consultation de l'OOS "contamine" ces donnees.

### 2.4 Information Criterion (AIC/BIC)
Mesure le "bang-for-buck" de chaque parametre supplementaire.

### 2.5 Kalman Filter
Elimine le choix de fenetre -> reduit l'overfitting sur ce parametre.

### 2.6 Walk-Forward Analysis (WFA)
```
Training : 12 mois | Blind test : 3 mois
Si Return Degradation < 50% et Sharpe OOS > 0.5 : valider
Sinon : rejeter, iterer
```

### Degradation mesuree dans la litterature
| Strategie | Rendement affiche | Rendement WFA | Degradation |
|-----------|------------------|---------------|-------------|
| RSI + SMA(20) | +199.2% | +5.8% | **97%** |
| Butterworth + ATR | +1,500% | +7% | **99.5%** |

---

## 3. Look-Ahead Bias -- Filtres

### Filtres causaux vs non-causaux
| Fonction | Biais |
|----------|-------|
| `scipy.signal.filtfilt` | Utilise les prix FUTURS -> **INTERDIT** en backtesting |
| `scipy.signal.lfilter` | Strictement causal -> **OBLIGATOIRE** |

---

## 4. CPCV (Combinatorial Purged Cross-Validation)

### Configuration du projet
CPCV(10,2) : 45 chemins combinatoriaux.
- Sharpe = mean/std des PnL (PAS annualise, pas de sqrt(N))
- Trade attribue si entree ET sortie dans les blocs test
- Purge 100 barres (~8h)

### Pourquoi CPCV
- Train/test classique : un seul split arbitraire
- CPCV : 45 splits differents -> distribution du Sharpe OOS
- Purge : elimine la contamination de la fuite d'info autour des frontieres train/test

---

## 5. Deflated Sharpe Ratio (DSR)

Correction pour le nombre de strategies testees (multiple testing).
Si on teste 100 strategies, la meilleure aura un Sharpe gonfle par le hasard.
DSR ajuste le Sharpe observe pour le nombre de tentatives.

---

## 6. Binary Gates

### Architecture
ADF/Hurst/Corr thresholds. Toutes doivent passer (AND).
`apply_gate_filter_numba()` bloque les ENTREES quand gate=False.
Ne bloque JAMAIS les sorties (TP/SL toujours executes).

---

## 7. Confidence Scoring

### Architecture
Scores 0->1 via interpolation lineaire.
ADF gate a -1.00 -> 0%.
Poids pair-specific.
Philosophie : la confiance module l'aggressivite, pas le go/no-go binaire.

---

## 8. Neighborhood Robustness

### Principe L1
Tester les parametres voisins du point optimal.
Si la performance s'effondre dans le voisinage -> overfitting probable.
Si la performance est stable -> parametres robustes.

---

## 9. Propfirm Metrics

Metriques calibrees sur compte $150K.
Criteres specifiques aux challenges propfirm (drawdown max, consistency, etc.).

---

## 10. Pipeline Complet Recommande

```
1. Modelisation (OU + AR)    -> modele sur spread
2. Validation (CPCV + WFA)   -> performance OOS, 45 chemins
3. Si Return Degradation < 50% et Sharpe OOS > 0.5 :
     -> Transaction Cost Analysis -> Paper trading -> Deploiement reel
   Sinon :
     -> Rejeter, iterer
```

### Regles du projet
- sigma_eq FIXE toute la session (pas adaptatif)
- r_ewma_span et adaptive_Q INVALIDES (MaxDD 2-7x pire)
- Alpha Kalman : 1.5e-7 a 3e-7
- Z-score Kalman (innovation) : z_entry = 1.5-2.0 (pas meme echelle que OLS 3.15)
- Lecon transversale : l'elegance mathematique ne suffit pas, seule la perf OOS compte

---

## 11. Instruments Reference

### Multipliers critiques (Point Value)
| Sym | Tick Size | Tick Value | Multiplier |
|-----|-----------|-----------|-----------|
| NQ | 0.25 | $5.00 | 20 |
| ES | 0.25 | $12.50 | 50 |
| RTY | 0.10 | $5.00 | 50 |
| YM | 1.00 | $5.00 | 5 |
| CL | 0.01 | $10.00 | 1000 |
| NG | 0.001 | $10.00 | 10000 |
| HO | 0.0001 | $4.20 | 42000 |
| GC | 0.10 | $10.00 | 100 |
| SI | 0.005 | $25.00 | 5000 |
| HG | 0.0005 | $12.50 | 25000 |
| PA | 0.50 | $50.00 | 100 |
| ZC | 0.25 | $12.50 | 50 |
| ZW | 0.25 | $12.50 | 50 |
| MNQ | 0.25 | $0.50 | 2 |
| MGC | 0.10 | $1.00 | 10 |
| SIL | 0.005 | $5.00 | 1000 |
| MCL | 0.01 | $1.00 | 100 |

### COMM_RT / SLIP_RT par actif
| Sym | COMM_RT | SLIP_RT | Total RT |
|-----|---------|---------|---------|
| NQ | $3.80 | $10.00 | $13.80 |
| ES | $3.80 | $25.00 | $28.80 |
| RTY | $3.80 | $10.00 | $13.80 |
| CL | $4.00 | $20.00 | $24.00 |
| NG | $4.20 | $20.00 | $24.20 |
| HO | $4.00 | $8.40 | $12.40 |
| GC | $4.20 | $20.00 | $24.20 |
| SI | $4.20 | $50.00 | $54.20 |
| PA | $4.20 | $100.00 | $104.20 |
| ZC | $5.30 | $25.00 | $30.30 |
| ZW | $5.30 | $25.00 | $30.30 |
