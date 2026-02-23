# BRIEF — Session Claude Code : Grid Search NQ/RTY + ES/RTY

**Objectif** : Trouver les meilleures configs OLS et Kalman pour les paires NQ/RTY et ES/RTY, en suivant le workflow valide sur NQ/YM (7 phases). Ces configs alimenteront un indicateur Sierra Charts pour du trading discretionnaire (biais macro + timing statistique).

**Contexte** : Ce n'est PAS du stat arb pur. Le trader a un biais directionnel discretionnaire, le systeme time l'entree avec precision statistique. Une paire avec un edge modeste mais un signal clair est utile — on ne cherche pas PF 2.0+ obligatoirement.

**Mode de travail** : Travailler en AUTONOMIE sur tout le workflow. Presenter les resultats finaux a l'utilisateur avec les verdicts. Si une paire est clairement non-viable (spread non-stationnaire, 0% configs profitables), documenter et passer a la suivante sans attendre validation.

---

## 1. REFERENCE : CE QUI A MARCHE SUR NQ/YM

### Workflow complet (7 phases, resume)

```
Phase 0 : Exploration rapide OLS + Kalman (quelques configs manuelles)
    -> Identifier si la paire a un edge brut
    -> TEST CRITIQUE : verifier symetrie long/short

Phase 1 : Grid Search OLS large (43,200 configs)
    -> Identifier la zone de parametres robuste
    -> Eliminer les artefacts (z_exit=0.0, pics isoles)

Phase 2 : Ablation des filtres
    -> Tester chaque filtre individuellement
    -> Confirmer que le confidence scoring > filtres binaires

Phase 3 : Validation quantitative (IS/OOS, Walk-Forward, Permutation)
    -> Seul gate obligatoire : WF >= 4/5, Permutation p < 0.01

Phase 4 : Sweep confidence fin (pas de 1%)
    -> Trouver la transition exacte du seuil de confidence
    -> Sur NQ/YM : transition nette a 67%

Phase 6 : Grid Search affine (1M+ combos avec Numba)
    -> Parametres fins, time stops, windows horaires
    -> Selection top 5 configs par profil

Phase 7 : Grid Search Kalman (1M+ combos)
    -> Alpha, z_entry/exit/stop adaptes au z-score innovation N(0,1)
    -> Validation top 5 Kalman separement
```

### Configs champion NQ/YM (pour reference, ne PAS reutiliser ces valeurs)

**OLS Config E** : OLS=3300, ZW=30, z_entry=3.15, z_exit=1.00, z_stop=4.50, conf>=67%, window 02:00-14:00, profil tres_court
-> 225 trades, PF 1.83, Sharpe 1.26, WF 4/5, Permutation p=0.000

**Kalman K_Balanced** : alpha=3e-7, ze=1.3125, zx=0.375, zs=2.75, conf>=75%, window 03:00-12:00, profil tres_court
-> 218 trades, PF 2.09, WR 78.4%, WF 5/5

### Learnings cles a appliquer

1. **z_exit=0.0 est un artefact** — toujours exclure du grid (exit=0 = ne jamais sortir sauf stop)
2. **Confidence scoring continu >> filtres binaires** — ne pas perdre de temps sur les seuils binaires ADF/Hurst
3. **Le profil metrics "tres_court"** (ADF=12, Hurst=64, HL=12, Corr=6) a domine sur NQ/YM. Tester aussi "court" et "moyen" comme dimensions du grid
4. **OLS window 2640-3300** etait le sweet spot NQ/YM. Pour NQ/RTY et ES/RTY, la dynamique peut etre differente — ne pas restreindre a priori
5. **Kalman alpha sweet spot** : 1.5e-7 a 3e-7 sur NQ/YM. Tester une gamme plus large au debut
6. **Innovation z-score Kalman est N(0,1)** -> z_entry optimal = 1.0-2.5 (PAS 3.0+ comme OLS)
7. **Warmup et gap_P_multiplier Kalman** : aucun impact sur NQ/YM, fixer a warmup=200 et gap_P_mult=10.0
8. **Confidence sweep fin** : sur NQ/YM, la transition 66->67% etait invisible dans un grid a pas de 5%. Un sweep 1% par 1% apres le grid initial est obligatoire
9. **2023 est toxique** : tech rally NQ +38.2% vs YM +8.9% a detruit la cointegration. NQ/RTY sera probablement PIRE (RTY small-cap encore plus decouple de NQ tech). ES/RTY pourrait etre different (ES diversifie). Analyser 2023 en priorite
10. **FLAT exits = 86% des pertes 2023 sur NQ/YM** : positions tenues jusqu'a force-close 15:30 quand le spread diverge. A surveiller aussi sur les nouvelles paires
11. **P_trace Kalman est un proxy temporel** (rho=-1.000 avec le temps) — NE PAS utiliser comme filtre
12. **R adaptatif EWMA INVALIDE** : r_ewma_span=0 obligatoire (MaxDD 2-7x pire si active)

---

## 2. HISTORIQUE DES PAIRES — CE QU'ON SAIT DEJA

### NQ/RTY — REJETE en Phase 0 (a re-tester)
- Test 0.3 (ancien, parametres grossiers) : Kalman a=1e-7, +$21,650, PF 2.67, 77% WR
- **REJET** : profit venait quasi-exclusivement du cote LONG (NQ surperforme RTY structurellement)
- C'est du beta directionnel, pas de la mean-reversion
- MAIS : le pipeline a evolue enormement depuis (confidence scoring, grid affine, validation WF). Le re-test avec le workflow mature peut donner un resultat different
- **Attention** : NQ et RTY ont des dynamiques tres differentes (tech mega-cap vs small-cap). La correlation est plus faible que NQ/YM
- **Attention 2023** : NQ +38% tech rally vs RTY small-cap = probablement la pire annee pour cette paire. S'attendre a un drawdown massif en 2023

### ES/RTY — Jamais teste serieusement
- ES/YM teste en Phase 0 -> systematiquement perdant
- ES/RTY est une paire differente (large-cap vs small-cap). Plus de litterature academique sur cette paire
- A explorer sans a priori
- **Avantage potentiel** : ES est plus diversifie que NQ (moins tech-concentre), donc le regime 2023 pourrait etre moins destructeur

### Instruments
```yaml
NQ: multiplier=20.0, tick=0.25, tick_value=5.0, commission=2.50
ES: multiplier=50.0, tick=0.25, tick_value=12.50, commission=2.50
RTY: multiplier=50.0, tick=0.10, tick_value=5.0, commission=2.50
YM: multiplier=5.0, tick=1.0, tick_value=5.0, commission=2.50
```

### Regression convention
`log_a = alpha + beta * log_b + epsilon` — leg_a est dependante, leg_b est explicative.
- NQ/RTY : NQ dependant, RTY explicatif (beta ~ NQ/RTY ratio)
- ES/RTY : ES dependant, RTY explicatif (beta ~ ES/RTY ratio)

---

## 3. PRE-REQUIS CRITIQUE : SCRIPTS A CREER

**Les scripts existants sont 100% hardcodes pour NQ_YM.** Il faut creer des copies adaptees.

### Scripts a dupliquer et adapter :

| Script source | Script a creer | Modifications |
|---|---|---|
| `scripts/run_refined_grid.py` | `scripts/run_grid_ols_NQ_RTY.py` | MULT_A=20.0, MULT_B=50.0, TICK_A=0.25, TICK_B=0.10, SpreadPair(NQ, RTY) |
| `scripts/run_refined_grid.py` | `scripts/run_grid_ols_ES_RTY.py` | MULT_A=50.0, MULT_B=50.0, TICK_A=0.25, TICK_B=0.10, SpreadPair(ES, RTY) |
| `scripts/run_grid_kalman_v3.py` | `scripts/run_grid_kalman_NQ_RTY.py` | Idem MULT/TICK + SpreadPair(NQ, RTY), supprimer baseline OLS NQ_YM |
| `scripts/run_grid_kalman_v3.py` | `scripts/run_grid_kalman_ES_RTY.py` | Idem MULT/TICK + SpreadPair(ES, RTY) |
| `scripts/validate_kalman_top.py` | `scripts/validate_top_NQ_RTY.py` | MULT/TICK + SpreadPair + configs champions a definir apres grid |
| `scripts/validate_kalman_top.py` | `scripts/validate_top_ES_RTY.py` | Idem |

**Points a modifier dans CHAQUE copie** :
1. `MULT_A, MULT_B` — multiplieurs du contrat (cf. instruments.yaml)
2. `TICK_A, TICK_B` — taille du tick
3. `SpreadPair(leg_a=Instrument.XX, leg_b=Instrument.YY)` — paire
4. `PAIR = ("XX", "YY")` — label
5. Titre/description dans l'argparse
6. Fichier de sortie CSV : `grid_ols_NQ_RTY.csv`, etc.
7. `OLS_BASELINE` dans le script Kalman : supprimer ou remplacer par baseline OLS de la paire (a definir apres grid OLS)

**NE PAS modifier `src/`** — le code source est generique et fonctionne deja sur n'importe quelle paire.

---

## 4. WORKFLOW A EXECUTER

### Etape 0 — Diagnostic du spread (10 min par paire)

**AVANT tout grid search**, calculer les statistiques de base du spread pour chaque paire. Cela determine si le grid search a des chances d'aboutir et calibre les ranges de parametres.

```python
# Pour chaque paire, calculer :
from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
import numpy as np

pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
aligned = load_aligned_pair_cache(pair, "5min")

# OLS spread (window=2640 par defaut)
est = create_estimator("ols_rolling", ols_window=2640)
hr = est.estimate(aligned)
spread = hr.spread.dropna()

# Statistiques cles :
# 1. Hurst (variance-ratio) — doit etre < 0.50 pour mean-reversion
# 2. ADF rolling — combien de % du temps le spread est stationnaire ?
# 3. Correlation rolling NQ vs RTY — stabilite ?
# 4. Drift annuel du spread — tendance structurelle ?
# 5. ACF(1) du spread — y a-t-il de l'autocorrelation ?
```

**Decision gate** :
- Hurst median > 0.55 ET ADF jamais significatif -> paire NON VIABLE, documenter et stop
- Hurst median 0.45-0.55 -> paire MARGINALE, continuer avec prudence
- Hurst median < 0.45 -> paire PROMETTEUSE, continuer

Produire un tableau comparatif avec NQ/YM comme reference :

| Metrique | NQ/YM (ref) | NQ/RTY | ES/RTY |
|---|---|---|---|
| Hurst median | 0.41 | ? | ? |
| ADF % significant | ~60% | ? | ? |
| Correlation mean | ~0.85 | ? | ? |
| Spread drift annuel | variable | ? | ? |
| ACF(1) median | ~0.10 | ? | ? |

### Etape 1 — Exploration rapide (30 min par paire)

Lancer quelques configs manuelles pour voir si un edge brut existe :

```bash
# NQ/RTY - OLS (3 configs : tight, medium, loose)
python scripts/run_backtest.py --pair NQ_RTY --method ols_rolling --ols-window 2640 --z-entry 3.0 --z-exit 1.5 --z-stop 4.0 --min-confidence 50
python scripts/run_backtest.py --pair NQ_RTY --method ols_rolling --ols-window 2640 --z-entry 2.5 --z-exit 1.0 --z-stop 3.5 --min-confidence 60
python scripts/run_backtest.py --pair NQ_RTY --method ols_rolling --ols-window 3960 --z-entry 3.0 --z-exit 1.0 --z-stop 4.5 --min-confidence 50

# NQ/RTY - Kalman (3 configs)
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7 --z-entry 1.5 --z-exit 0.5 --z-stop 2.5 --min-confidence 50
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 1e-7 --z-entry 2.0 --z-exit 0.5 --z-stop 2.75 --min-confidence 60
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7 --z-entry 1.5 --z-exit 0.25 --z-stop 2.75 --min-confidence 70

# ES/RTY - memes tests (adapter --pair ES_RTY)
```

**TEST CRITIQUE OBLIGATOIRE** : Pour chaque config profitable, analyser la repartition long/short.
```
Si PnL_long > 80% du PnL total -> BIAIS DIRECTIONNEL, noter le warning
Si PnL_long / PnL_short ratio > 3:1 -> le profit vient du beta, pas du mean-reversion
```

Ce n'est PAS un rejet automatique (le trader a un biais discretionnaire), mais il faut le documenter clairement. Le trader doit savoir que sur cette paire, l'edge n'est fiable que dans une direction.

### Etape 2 — Grid Search OLS (1-2h par paire)

Creer le script dedie (cf. section 3) puis lancer.

**Parametres a tester** :
```python
OLS_WINDOWS = [1320, 1980, 2640, 3300, 3960, 5280]  # Plus large que NQ/YM
ZW_WINDOWS = [24, 30, 36, 42, 48, 60]
Z_ENTRIES = [2.50, 2.75, 3.00, 3.15, 3.30, 3.50]
Z_EXITS = [0.50, 0.75, 1.00, 1.20, 1.25, 1.50]  # PAS de 0.0 (artefact)
Z_STOPS = [3.50, 4.00, 4.25, 4.50, 5.00]
CONFS = [50, 60, 65, 67, 70, 75]  # Commencer plus bas (edge peut etre plus faible)
TIME_STOPS = [0, 12, 18, 24, 36]
PROFILES = ["tres_court", "court", "moyen"]  # 3 profils comme dimension
WINDOWS = [
    ("02:00-14:00", 2, 0, 14, 0),
    ("03:00-12:00", 3, 0, 12, 0),
    ("03:00-14:00", 3, 0, 14, 0),
    ("04:00-12:00", 4, 0, 12, 0),
    ("04:00-13:00", 4, 0, 13, 0),
    ("04:00-14:00", 4, 0, 14, 0),
    ("05:00-12:00", 5, 0, 12, 0),
]
FLAT_MIN = 930  # 15:30 CT
```

**Attention sizing** :
- NQ/RTY : MULT_A=20.0 (NQ), MULT_B=50.0 (RTY), TICK_A=0.25, TICK_B=0.10
- ES/RTY : MULT_A=50.0 (ES), MULT_B=50.0 (RTY), TICK_A=0.25, TICK_B=0.10

**Output** : Sauvegarder dans `output/results/grid_ols_NQ_RTY.csv` et `grid_ols_ES_RTY.csv`

### Etape 3 — Analyse des resultats OLS (30 min par paire)

Pour chaque paire :
1. Combien de configs profitables ? (comparaison : NQ/YM = 54% en Phase 6 affinee)
2. Quel est le sweet spot OLS window / ZW / z_entry ?
3. Y a-t-il une zone robuste (voisins du champion aussi profitables) ?
4. Sensibilite au confidence threshold — identifier la transition (sweep 1% si necessaire)
5. Sensibilite au profil metrics — quel profil domine ?
6. Analyse directionnelle long/short du top 5

**Si < 5% de configs profitables avec PF > 1.3** -> la paire OLS est probablement non-viable. Passer directement au Kalman.

### Etape 4 — Confidence sweep fin (si transition detectee)

Apres le grid initial, si une zone de transition du confidence est identifiee :

```python
# Sweep fin : confidence de X% a Y% par pas de 1%
CONFS_FINE = list(range(55, 82))  # 55% a 81% par pas de 1%
```

Sur NQ/YM, c'est cette etape qui a identifie la transition nette a 67%.

### Etape 5 — Grid Search Kalman (1-2h par paire)

Creer le script dedie (cf. section 3) puis lancer.

```python
ALPHAS = [5e-8, 1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 5e-7, 7e-7, 1e-6]
Z_ENTRIES = [1.0, 1.25, 1.375, 1.5, 1.625, 1.75, 2.0, 2.25, 2.5]  # Innovation z = N(0,1)
Z_EXITS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25]  # PAS de 0.0
Z_STOPS = [2.0, 2.25, 2.5, 2.625, 2.75, 3.0, 3.25]
CONFS = [50, 60, 63, 65, 67, 69, 70, 72, 75]
PROFILES = ["tres_court", "court", "moyen"]
WINDOWS = [memes que OLS]
FLAT_MIN = 930
FIXED_WARMUP = 200
FIXED_GAP_P_MULT = 5.0
# r_ewma_span = 0 obligatoire (INVALIDE sur NQ/YM, ne pas tester)
# adaptive_Q = False obligatoire
```

**Important** : les MULT et TICK doivent correspondre a la paire testee.

### Etape 6 — Top 5 selection + Validation (1-2h par paire)

Pour OLS et Kalman separement :

1. **Selectionner top 5 configs** par profil :
   - Balanced : meilleur compromis PF x trades
   - Volume : max trades avec PF > 1.3
   - Quality : meilleur PF avec trades > 50
   - Sniper : PF > 2.5, WR > 70%
   - PropFirm : MaxDD < $4,500 avec meilleur PnL

2. **Validation complete** (creer script dedie cf. section 3) :
   - IS/OOS 60/40 : PF OOS >= PF IS idealement
   - Walk-Forward (IS=2y, OOS=6m, step=6m) : minimum 4/5 profitable
   - Permutation 1000x : p < 0.01

3. **Analyse directionnelle** : pour chaque config validee, reporter PnL long vs PnL short

4. **Decomposition annuelle** : PnL par annee, identifier les annees negatives

### Etape 7 — Autopsie des annees perdantes

Si une ou plusieurs annees sont negatives (2023 probable pour NQ/RTY) :
- Adapter `analyze_2023_losses.py` pour la paire
- Analyser les trades perdants (type de sortie FLAT/STOP/EXIT, side, heure, jour)
- Identifier le regime macro (correlation, drift, volatilite du spread)
- Comparer avec le comportement NQ/YM la meme annee
- Documenter pour le trader discretionnaire

### Etape 8 — Complementarite cross-paire

**Apres avoir les configs validees pour les 2 paires** :
- Calculer le chevauchement temporel des signaux NQ/RTY vs ES/RTY vs NQ/YM (jours de trading communs)
- Analyser la complementarite : quand NQ/YM perd, est-ce que NQ/RTY ou ES/RTY compensent ?
- Produire un tableau annuel croise : PnL par paire par annee
- Objectif : verifier que les paires apportent de la diversification (pas juste le meme signal decore differemment)

---

## 5. LIVRABLES ATTENDUS

### Pour chaque paire (NQ/RTY et ES/RTY), produire :

1. **Diagnostic spread** (Etape 0) :
   - Tableau Hurst / ADF / Correlation / Drift / ACF compare a NQ/YM
   - Verdict : PROMETTEUSE / MARGINALE / NON VIABLE

2. **Resume CHANGELOG** (format identique aux phases existantes) :
   - Nombre de configs testees, % profitables
   - Top 5 OLS + Top 5 Kalman avec metriques completes
   - Resultats validation (IS/OOS, WF, Permutation)
   - Analyse directionnelle long/short
   - Warning si biais directionnel detecte
   - Decomposition annuelle (quelle annee perd, metriques du regime)

3. **Config retenue** (ou verdict "paire non viable") :
   - OLS : tous les parametres + metriques + WF result
   - Kalman : tous les parametres + metriques + WF result
   - Backup config si applicable

4. **Fichiers CSV des resultats grid** :
   - `output/results/grid_ols_NQ_RTY.csv`
   - `output/results/grid_kalman_NQ_RTY.csv`
   - `output/results/grid_ols_ES_RTY.csv`
   - `output/results/grid_kalman_ES_RTY.csv`

5. **Tableau complementarite cross-paire** (Etape 8)

6. **Mise a jour CHANGELOG.md** avec tous les resultats

---

## 6. REGLES CRITIQUES

### NE PAS FAIRE
- Ne pas modifier le code source (`src/`) — creer des scripts dedies dans `scripts/`
- Ne pas reutiliser les parametres NQ/YM — chaque paire a sa propre dynamique
- Ne pas ignorer l'analyse long/short — c'est LE test qui a tue NQ/RTY en Phase 0
- Ne pas valider une config sans Walk-Forward — les configs in-sample-only sont du bruit
- Ne pas presenter une config avec z_exit=0.0 — c'est un artefact connu (ne sort jamais sauf stop)
- Ne pas activer r_ewma_span > 0 ni adaptive_Q=True dans le Kalman (INVALIDE, MaxDD 2-7x pire)
- Ne pas utiliser P_trace comme filtre de trading (proxy temporel, rho=-1.000 avec le temps)
- Ne pas lancer un grid search sans diagnostic du spread d'abord (Etape 0 obligatoire)

### TOUJOURS FAIRE
- Verifier la symetrie long/short de chaque config retenue
- Documenter le comportement annuel (quelle annee perd, pourquoi)
- Comparer OLS vs Kalman : chevauchement temporel, complementarite
- Rapporter les resultats negatifs ("paire non viable" est un resultat valide et utile)
- Tester l'ablation du confidence score (avec vs sans) pour verifier qu'il ajoute de la valeur
- Inclure le profil metrics (tres_court/court/moyen) comme dimension du grid
- Sweep fin du confidence threshold (+/- 10% autour du sweet spot, pas de 1%)
- Comparer les resultats 2023 entre paires (regime commun ou specifique ?)

---

## 7. CRITERES DE DECISION

### Config VALIDEE si :
- Walk-Forward >= 4/5 fenetres profitables
- Permutation p < 0.01
- PF OOS >= 1.3 (seuil plus bas que NQ/YM car usage discretionnaire)
- Au moins 50 trades sur 5 ans (10/an minimum)
- Pas de biais directionnel > 80% (ou biais documente si accepte pour usage discretionnaire)

### Paire REJETEE si :
- Spread non-stationnaire (Hurst median > 0.55 ET ADF jamais significatif)
- Aucune config ne passe WF 4/5
- Toutes les configs profitables ont un biais directionnel > 90%
- 0% de configs profitables avec PF > 1.2

### Paire VIABLE AVEC RESERVES si :
- Configs validees mais biais directionnel 60-80%
- Edge modeste (PF 1.3-1.5) mais stable
- Peu de trades (30-80 sur 5 ans) mais qualite elevee
-> Utilisable en mode discretionnaire avec le warning appropriate

---

## 8. REFERENCES CODE

```bash
# Scripts principaux (NE PAS MODIFIER — les copier et adapter pour chaque paire)
scripts/run_backtest.py          # Backtest single config (deja generique via --pair)
scripts/run_refined_grid.py      # Grid search OLS (template, hardcode NQ_YM)
scripts/run_grid_kalman_v3.py    # Grid search Kalman (template, hardcode NQ_YM)
scripts/validate_kalman_top.py   # Validation Kalman (template, hardcode NQ_YM)
scripts/validate_top5_configs.py # Validation OLS (template, hardcode NQ_YM)
scripts/analyze_2023_losses.py   # Autopsie annee perdante (template, hardcode NQ_YM)

# Documentation (LIRE AVANT DE COMMENCER)
CLAUDE.md                        # Conventions projet + architecture
CHANGELOG.md                     # Historique des tests (NQ/YM phases 0-8)
config/instruments.yaml          # Specs instruments (multipliers, ticks, commissions)
config/pairs.yaml                # Definition des paires
```

### Constantes par paire a mettre dans les scripts :

| Paire | MULT_A | MULT_B | TICK_A | TICK_B | SpreadPair |
|---|---|---|---|---|---|
| NQ/YM (ref) | 20.0 | 5.0 | 0.25 | 1.0 | SpreadPair(Instrument.NQ, Instrument.YM) |
| NQ/RTY | 20.0 | 50.0 | 0.25 | 0.10 | SpreadPair(Instrument.NQ, Instrument.RTY) |
| ES/RTY | 50.0 | 50.0 | 0.25 | 0.10 | SpreadPair(Instrument.ES, Instrument.RTY) |

### Constantes universelles (ne pas modifier) :
```python
SLIPPAGE = 1          # 1 tick par leg par side
COMMISSION = 2.50     # $ par side par contrat (standard)
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930        # 15:30 CT = force close
```

---

## 9. ESTIMATION TEMPS

| Etape | NQ/RTY | ES/RTY |
|---|---|---|
| Diagnostic spread (Etape 0) | 10 min | 10 min |
| Creation scripts dedies | 30 min | 30 min |
| Exploration rapide (Etape 1) | 30 min | 30 min |
| Grid OLS (Etape 2) | 1-2h | 1-2h |
| Analyse OLS + sweep conf (Etapes 3-4) | 45 min | 45 min |
| Grid Kalman (Etape 5) | 1-2h | 1-2h |
| Top 5 + Validation (Etape 6) | 1-2h | 1-2h |
| Autopsie perdantes (Etape 7) | 30 min | 30 min |
| Complementarite cross-paire (Etape 8) | 30 min | — |
| Documentation | 30 min | 30 min |
| **Total par paire** | **~6-9h** | **~6-9h** |

Les grid searches sont le gros du temps. Les scripts Numba existants tournent a ~7,500 combos/s avec 5 workers.

**Optimisation** : les 2 paires peuvent etre traitees en parallele (pas de dependance) sauf l'Etape 8 (complementarite cross-paire) qui necessite les resultats des 2.
