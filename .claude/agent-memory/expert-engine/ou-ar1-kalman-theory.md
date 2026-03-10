# OU, AR(1) & Kalman Filter -- Theory Reference

## 1. Modele AR(1) -- Autoregressive d'Ordre 1

### Equation
```
S_t = c + phi * S_{t-1} + epsilon_t,   epsilon_t ~ N(0, sigma^2)
```
- phi : coefficient autoregressif (persistance)
- c : constante (drift)
- Condition de stationnarite : |phi| < 1

### Moyenne d'equilibre
```
mu = E[S_t] = c / (1 - phi)
```

### Demi-vie
```
t_{1/2} = -ln(2) / ln(phi)   (en nombre de periodes)
```

### Limites
1. Temps discret vs marche continue (slippage)
2. Homoscedasticite supposee (sigma constant)
3. Seuils heuristiques (+/-2 sigma = arbitraire)
4. Beta statique (meme probleme qu'Engle-Granger)

---

## 2. Processus d'Ornstein-Uhlenbeck (OU)

### EDS (Equation Differentielle Stochastique)
```
dX_t = theta * (mu - X_t) * dt + sigma * dW_t
```
- theta (>0) : vitesse de retour a la moyenne (mean reversion speed)
- mu : moyenne long terme
- sigma : volatilite du processus
- dW_t : increment brownien

### Deux forces en equilibre
- VENT (sigma * dW_t) : pousse aleatoirement
- ELASTIQUE (theta * (mu - X_t)) : ramene vers mu
- theta grand + sigma petit = processus concentre autour de mu
- theta petit + sigma grand = grandes excursions

### Solution analytique
```
X_t = X_0 * e^(-theta*t) + mu * (1 - e^(-theta*t)) + sigma * integral_0^t e^(-theta*(t-s)) dW_s
```
Decomposition :
- `X_0 * e^(-theta*t)` : oubli exponentiel de la condition initiale
- `mu * (1 - e^(-theta*t))` : convergence vers mu
- Integrale stochastique : fluctuations accumulees

### Discretisation d'Euler
```
X_{t+1} = X_t + theta * (mu - X_t) * dt + sigma * sqrt(dt) * epsilon_t
```

### Distribution de transition (gaussienne)
```
X_t | X_s ~ N(
    X_s * e^{-theta*(t-s)} + mu * (1 - e^{-theta*(t-s)}),
    sigma^2 / (2*theta) * (1 - e^{-2*theta*(t-s)})
)
```

### Demi-vie continue
```
t_{1/2} = ln(2) / theta
```

### Estimation des parametres
**Methode des Moments (rapide, bonne pour init) :**
```python
def method_moments(X, dt):
    dX = np.diff(X)
    mu = X.mean()
    exog = (mu - X[:-1]) * dt
    model = OLS(dX, exog).fit()
    theta = model.params[0]
    sigma = (dX - theta * exog).std() / np.sqrt(dt)
    return theta, mu, sigma
```

**MLE (plus precis, init par MoM) :**
```python
def ll(params, X, dt):
    theta, mu, sigma = params
    dX = np.diff(X)
    X_prev = X[:-1]
    return np.sum(np.log(sigma * np.sqrt(2*np.pi*dt)) +
                  (dX - theta*(mu - X_prev)*dt)**2 / (2*sigma**2*dt))
```
Strategie hybride : MoM pour init, puis MLE Nelder-Mead.

---

## 3. Isomorphisme AR(1) <-> OU

### Relation fondamentale
```
phi = e^{-theta * dt}
```

| AR(1) | OU | Relation |
|-------|-----|---------|
| phi | theta | phi = e^{-theta*dt} |
| c/(1-phi) | mu | Meme moyenne d'equilibre |
| sigma_AR | sigma_OU | sigma_AR = sigma_OU * sqrt((1-e^{-2*theta*dt}) / (2*theta)) |

### Conversion AR(1) -> Parametres OU (Formules de Paolucci)
```
dt = 1   (convention pas-a-pas)
kappa = -ln(phi) / dt                         # vitesse de retour
theta_OU = c / (1 - phi)                      # centre equilibre ~ 0 par OLS

# DEUX objets distincts issus de sigma_eta :
sigma_eq = sigma_eta / sqrt(1 - phi^2)        # ECART-TYPE distribution stationnaire
                                               # -> denominateur du Z-score
sigma_diffusion = sigma_eta * sqrt(-2*ln(phi) / (1 - phi^2))  # coeff diffusion SDE
                                               # -> usage interne Q_OU uniquement
```

### CORRECTION CRITIQUE : sigma_eq vs sigma_diffusion
```
Z-score = (Spread_t - theta_OU) / sigma_eq    <- CORRECT
Z-score = (Spread_t - theta_OU) / sigma_diffusion  <- FAUX (seuils faux x3)
```
Pour phi = 0.95 : sigma_eq ~ 3.2 * sigma_eta vs sigma_diffusion ~ 1.03 * sigma_eta.

### Condition de validite
```
phi in ]0, 1[  OBLIGATOIRE
phi = 1  -> kappa = 0 -> marche aleatoire
phi -> 0+ -> kappa -> +inf -> reversion instantanee (bruit blanc)
phi <= 0  -> ln(phi) non defini
phi >= 1  -> variance infinie, processus explosif
```

### Half-life : Modele vs Empirique
```
HL_modele = ln(2) / kappa * 5 minutes     # indicateur relatif
HL_empirique = mediane des temps |Z| > 2*sigma_eq -> |Z| < 0.5*sigma_eq
                                           # REFERENCE OPERATIONNELLE
Ratio = HL_empirique / HL_modele
```
- 0.8-1.5 : OU bien calibre
- 1.5-2.0 : biais modere
- > 2.0 : OU significativement biaise
- < 0.8 : kappa surestime (microstructure noise)
- **BLOQUANT : Ratio > 3.0 -> STOP**

---

## 4. Filtre de Kalman

### Concept
Detective bayesien : theorie (modele) + temoignages bruites (observations).
A chaque observation, ajuste sa croyance en fonction de la coherence.
Avantage vs moving average : reactif sans lag, pas de fenetre a choisir.

### Representation espace-etat
**Equation de mesure :** `Y_t = H * X_t + v_t,  v_t ~ N(0, R)`
**Equation d'etat :** `X_t = F * X_{t-1} + w_t,  w_t ~ N(0, Q)`

### Cycle Predict-Update
**Prediction (a priori) :**
```
X_t|t-1 = F * X_{t-1|t-1}
P_t|t-1 = F * P_{t-1|t-1} * F^T + Q
```

**Mise a jour (a posteriori) :**
```
e_t = Y_t - H * X_t|t-1                         # innovation
S_t = H * P_t|t-1 * H^T + R                     # variance innovation
K_t = P_t|t-1 * H^T / S_t                       # gain de Kalman
X_t|t = X_t|t-1 + K_t * e_t                     # etat corrige
P_t|t = (I - K_t*H) * P_t|t-1 * (I - K_t*H)^T + K_t*R*K_t^T   # JOSEPH FORM
```

### Gain de Kalman K_t
```
K_t -> 0 : confiance au modele (R grand, mesure bruitee)
K_t -> 1 : confiance aux donnees (Q grand, etat incertain)
```

### FORME DE JOSEPH OBLIGATOIRE pour P_t
```
P_t = (I - K*H) * P_pred * (I - K*H)^T + K*R*K^T    <- CORRECT
P_t = (I - K*H) * P_pred                              <- INTERDIT
```
Forme standard accumule erreurs d'arrondi sur 350+ iterations -> P perd symetrie/PSD -> filtre explose.

### NIS (Normalized Innovation Squared)
```
NIS_t = e_t^2 / S_t    (doit etre ~ Chi2(1), E[NIS] ~ 1)
```
Filtre A : `NIS_t < 9.0 ET rolling_NIS_20 < 3.0`

### Application : Beta Dynamique
Etat cache : x_t = [alpha_t, beta_t] (2x1)
```
H_t = [1, log(B)_t]           (1x2)
x_pred = x_{t-1}              (random walk)
e_t = log(A)_t - H_t * x_pred
K_t = P_pred * H_t^T / S_t
x_t = x_pred + K_t * e_t
```
Avantage : elimine le choix de fenetre (vs rolling OLS).

### Calibration Q et R
- Q_Kalman : matrice 2x2 covariance bruit d'etat [alpha_t, beta_t], ordre ~1e-7 a 1e-6
- R : variance residus OLS (etape 3)
- Q_OU (variance spread par barre, ~1e-4) != Q_Kalman
- **Si Q_OU passe comme Q_Kalman : P_pred explose -> K->1 -> beta surreagit -> session tuee**

### Q_Kalman par classe (V1)
| Classe | q_alpha | q_beta |
|--------|---------|--------|
| Metals (GC/SI, GC/PA) | 1e-6 | 1e-7 |
| Equity Index (NQ/RTY) | 2e-6 | 2e-7 |
| Grains (ZC/ZW) | 2e-6 | 2e-7 |
| Energy (CL/HO, CL/NG) | 5e-6 | 5e-7 |

---

## 5. Synergie Hybride Kalman + OU

### Architecture du projet
```
OLS / OU  ->  QUOI trader et QUAND  (signal statistique, monde statique)
KALMAN    ->  COMBIEN trader         (sizing dynamique, monde adaptatif)
KALMAN    ->  SI on a le droit       (risk manager barre par barre)
```
Ces deux mondes ne se melangent JAMAIS. Separation des responsabilites = pilier de l'architecture.

### Niveaux d'implementation
1. **Debutant** : Engle-Granger + Z-score rolling
2. **Intermediaire** : AR(1) + Half-life + Rolling beta
3. **Avance** : Kalman beta dynamique + OU seuils Bertram
4. **Institutionnel** : Kalman sur OU latent + HMM regime detection

---

## 6. Frontieres de Bertram (Seuils Optimaux)

### Probleme
Seuils z-score +/-2 = heuristiques arbitraires. Bertram donne l'optimum mathematique.

### Framework
- Seuils symetriques [-a, +a]
- LONG quand spread atteint -a, CLOTURE quand revient a mu
- Objectif : maximiser `E[Profit] / E[Temps de cycle]`
- Solution analytique fermee (pas de Monte Carlo)
- Seuils optimaux dependent de theta et sigma (estimes MLE)

### Extensions
Integration couts de transaction : probleme multi-regime (neutre / long / short).
Chaque transition a un cout (commission + slippage).
