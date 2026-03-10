# Cointegration Fundamentals -- Knowledge Base

## 1. Stationnarite

### Definition (wide-sense)
```
1. E[X_t] = mu, pour tout t              (moyenne constante)
2. Var(X_t) = sigma^2, pour tout t       (variance constante)
3. Cov(X_t, X_{t+h}) = gamma(h)          (autocovariance ne depend que de h)
```
Si la distribution sous-jacente change (regime shift), tous les parametres calibres deviennent invalides.

### Ordres d'integration
- **I(0)** : serie stationnaire (rendements d'actions, bruit blanc)
- **I(1)** : premiere difference stationnaire (prix d'actions = marche aleatoire)
- **I(d)** : necessite d differenciations

### Test ADF (Augmented Dickey-Fuller)
```
H0 : racine unitaire (non-stationnaire, marche aleatoire)
H1 : serie stationnaire
p-value < 0.05 => rejeter H0 => probablement stationnaire
```
Limites : sensible a la longueur d'echantillon, choix des lags, faux positifs a 5%.

### Test KPSS (complementaire a ADF)
```
H0 : serie stationnaire
H1 : non-stationnaire
```
Croiser ADF + KPSS pour robustesse :
| ADF | KPSS | Conclusion |
|-----|------|-----------|
| Non-rejet (p>0.05) | Rejet (p<0.05) | I(1) confirme |
| Rejet (p<0.05) | Non-rejet | I(0) stationnaire |
| Rejet | Rejet | Incoherent |
| Non-rejet | Non-rejet | Ambigu |

### Exposant de Hurst
- H < 0.5 : anti-persistant (mean-reverting) -> favorable
- H = 0.5 : marche aleatoire
- H > 0.5 : persistant (trending)
Methode : variance-ratio (pas R/S qui biaise a ~0.99 sur niveaux de spread).

---

## 2. Cointegration

### Definition formelle
Deux series I(1) sont cointegrees si une combinaison lineaire est I(0) :
```
z(t) = A(t) - beta * B(t)  ~  I(0)
```

### ATTENTION : Correlation != Cointegration
| Propriete | Correlation | Cointegration |
|-----------|------------|---------------|
| Mesure | Co-mouvement des RENDEMENTS | Stationnarite du SPREAD |
| Temporalite | Instantanee, ephemere | Relation de long terme |
| Implication | "Bougent ensemble" | "REVIENNENT l'un vers l'autre" |

Deux series tres correlees peuvent ne PAS etre cointegrees (correlation fallacieuse).
Deux series peu correlees peuvent ETRE cointegrees.

### Methode d'Engle-Granger (1987) -- 2 etapes
**Etape 1 :** Regression OLS : `Y_t = alpha + beta * X_t + epsilon_t`
- beta = hedge ratio (ratio de couverture)
- epsilon_t = residus = spread

**Etape 2 :** Test ADF sur les residus
- ADF rejette H0 => residus stationnaires => X et Y cointegres

### Convention de regression du projet
```
log(A)_t = alpha + beta * log(B)_t + epsilon_t
Spread_t = log(A)_t - alpha_OLS - beta_OLS * log(B)_t   <- DEFINITION CANONIQUE
```
Par construction OLS : E[Spread_t] = 0, donc theta_OU ~ 0.
Asymetrie : tester les deux sens (A->B et B->A), retenir le meilleur residus.

### Valeurs critiques MacKinnon -- OBLIGATOIRE pour residus de regression bivariee
| N observations | Seuil 5% MacKinnon | DF Standard 5% (FAUX) |
|---------------|-------------------|----------------------|
| ~2 640 (10j)  | -3.37 | -2.86 |
| ~7 920 (30j)  | -3.35 | -2.86 |
| ~15 840 (60j) | -3.34 | -2.86 |

DF Standard sous-estime les seuils => faux positifs massifs.

---

## 3. Probleme du Grand N (Biais intraday)

Sur 60 jours de barres 5min : N ~ 15 840 observations.
Erreur standard ADF ~ 1/sqrt(N) -> tend vers 0 -> rejet quasi-systematique de H0.

**SOLUTION :** Downsampling proportionnel session-aware -> target_obs ~ 1000 par test.
```
10j -> 2 640 barres 5min -> arrondi 15min -> ~880 obs
30j -> 7 920 barres 5min -> arrondi 30min -> ~1 320 obs
60j -> 15 840 barres 5min -> arrondi 60min -> ~1 320 obs
```

Multi-scale validation : 3 frequences testees en parallele. I(1) confirme sur >= 2 frequences.

---

## 4. Stabilite Structurelle

### Beta rolling
Fenetre 30j divisee en 3 blocs de 10j -> CV(beta) = std/mean :
- CV < 10% -> stable
- CV 10-20% -> derive moderee -> warning
- CV > 20% -> instable -> observable critique

### Variance du spread
Ratio_var = max(Var_b)/min(Var_b) :
- Ratio < 1.5 -> stable
- Ratio 1.5-2 -> heteroscedasticite moderee
- Ratio > 2.0 -> instable -> observable critique

---

## 5. Selection de Paires

### Critere fondamental
Lien economique demonstrable AVANT d'ouvrir les donnees. Sans lien economique, la cointegration statistique est probablement spurious.

### Piege des comparaisons multiples
20 actifs => 190 paires testees. Avec p = 0.05 : ~10 paires "significatives" par HASARD.
Solution : hypothese economique a priori + Benjamini-Hochberg + verification OOS.

### Diversification
JAMAIS trader une seule paire. Portefeuille multi-paires, multi-secteurs.

---

## 6. Univers de Trading (6 paires futures)

| Paire | Lien Economique | Classe | Beta OLS Typique |
|-------|----------------|--------|-----------------|
| GC/SI | Metaux precieux, reserve de valeur | Metals | 1.3-1.5 |
| GC/PA | Metaux precieux, catalyse industrielle | Metals | 2.0-2.5 |
| NQ/RTY | Indices US, macro commune | Equity | 0.4-0.6 |
| CL/HO | Petrole/Fioul, raffinage direct | Energy | 0.8-1.0 |
| CL/NG | Petrole/Gaz, substitution energetique | Energy | 0.3-0.5 |
| ZC/ZW | Mais/Ble, cereales substituables | Grains | 0.7-0.9 |

GC/PA : cout friction eleve ($128.40 RT), filtree par non-rentabilite.
ZC/ZW : T_close_pit 13h20 CT, aucune entree apres.
