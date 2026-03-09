# Specs Actifs — Futures

Source : CME Group, AMP Futures, IBKR, NinjaTrader, LP Futures (vérifié le 2026-03-08)

---

## 1. Indices US (E-mini)

### Contract Specs

| | NQ | ES | RTY | YM |
|---|---|---|---|---|
| **Nom complet** | E-mini Nasdaq-100 | E-mini S&P 500 | E-mini Russell 2000 | E-mini Dow Jones |
| **Exchange** | CME | CME | CME | CME |
| **Globex Code** | NQ | ES | RTY | YM |
| **Sierra Symbol** | `NQ?##_FUT_CME` | `ES?##_FUT_CME` | `RTY?##_FUT_CME` | `YM?##_FUT_CME` |
| **Multiplier ($/pt)** | $20 | $50 | $50 | $5 |
| **Tick Size** | 0.25 pts | 0.25 pts | 0.10 pts | 1.00 pt |
| **Tick Value** | $5.00 | $12.50 | $5.00 | $5.00 |
| **Settlement** | Cash | Cash | Cash | Cash |
| **Commission** | ~$2.50/side | ~$2.50/side | ~$2.50/side | ~$2.50/side |

### Notionnel par contrat (ordre de grandeur, mars 2026)

| | Prix approx | Notionnel (prix × multiplier) |
|---|---|---|
| NQ | ~21,500 | ~$430,000 |
| ES | ~6,100 | ~$305,000 |
| RTY | ~2,250 | ~$112,500 |
| YM | ~44,500 | ~$222,500 |

### Implications pour le sizing

- **NQ est le plus gros notionnel** (~$430k) → souvent leg_a avec N_a=1
- **RTY est le plus petit** (~$112k) → N_b sera élevé quand paired avec NQ (ratio ~3-4x)
- La formule : `N_b = round((Not_A / Not_B) × β × N_a)`
- Exemple NQ/RTY : `N_b ≈ round((430k / 112k) × 1.5 × 1) ≈ 6 contrats`

### Session Globex (Indices)
- **Ouverture** : Dimanche 17h00 CT → Vendredi 16h00 CT
- **Session quotidienne** : 17h00 CT → 16h00 CT (avec pause 15h15-15h30 CT pour ES/NQ/RTY/YM)
- **Notre fenêtre** : 17h30-15h30 CT (buffer 30min), trading 04h00-14h00 CT

### 6 paires tradées (Indices)
NQ/ES, NQ/RTY, NQ/YM, ES/RTY, ES/YM, RTY/YM

---

## 2. Énergie — Tous les produits NYMEX

### Contract Specs — Crude Oil

| | CL | BZ | MCL |
|---|---|---|---|
| **Nom complet** | Light Sweet Crude Oil (WTI) | Brent Last Day Financial | Micro WTI Crude Oil |
| **Exchange** | NYMEX | NYMEX | NYMEX |
| **Globex Code** | CL | BZ | MCL |
| **Sierra Symbol** | `CL?##_FUT_CME` | `BZ?##_FUT_CME` | `MCL?##_FUT_CME` |
| **Contract Size** | 1,000 barrels | 1,000 barrels | 100 barrels |
| **Multiplier ($/pt)** | $1,000 | $1,000 | $100 |
| **Tick Size** | $0.01/barrel | $0.01/barrel | $0.01/barrel |
| **Tick Value** | $10.00 | $10.00 | $1.00 |
| **Contract Months** | Mensuel (Y+10 ans+2) | Mensuel (Y+7 ans+3) | Mensuel (Y+10 ans+2) |
| **Settlement** | Physical Delivery | Cash (Financial) | Cash (Financial) |
| **Exchange Fee** | ~$1.60/side | ~$1.50/side | ~$0.50/side |
| **Maint. Margin** | ~$7,450 | ~$7,370 | ~$747 |
| **ADV 2024** | ~983,000 | ~200,000 (est.) | ~50,000 (est.) |

### Contract Specs — Refined Products

| | HO | RB |
|---|---|---|
| **Nom complet** | NY Harbor ULSD (Heating Oil) | RBOB Gasoline |
| **Exchange** | NYMEX | NYMEX |
| **Globex Code** | HO | RB |
| **Sierra Symbol** | `HO?##_FUT_CME` | `RB?##_FUT_CME` |
| **Contract Size** | 42,000 gallons (1,000 bbl) | 42,000 gallons (1,000 bbl) |
| **Multiplier ($/pt)** | $42,000 (par $/gal) | $42,000 (par $/gal) |
| **Tick Size** | $0.0001/gallon | $0.0001/gallon |
| **Tick Value** | $4.20 | $4.20 |
| **Contract Months** | 18 mois consecutifs | 50 mois consecutifs |
| **Settlement** | Physical (NY Harbor) | Physical |
| **Exchange Fee** | ~$1.60/side | ~$1.60/side |
| **Maint. Margin** | ~$11,175 | ~$8,080 |
| **ADV 2024** | ~180,000 | ~130,000 (est.) |
| **ADV 2025** | **197,000** (record) | ~140,000 (est.) |

### Contract Specs — Natural Gas

| | NG | QG |
|---|---|---|
| **Nom complet** | Henry Hub Natural Gas | E-mini Natural Gas |
| **Exchange** | NYMEX | NYMEX |
| **Globex Code** | NG | QG |
| **Sierra Symbol** | `NG?##_FUT_CME` | `QG?##_FUT_CME` |
| **Contract Size** | 10,000 mmBtu | 2,500 mmBtu |
| **Multiplier ($/pt)** | $10,000 | $2,500 |
| **Tick Size** | $0.001/mmBtu | $0.005/mmBtu |
| **Tick Value** | $10.00 | $12.50 |
| **Contract Months** | Mensuel (Y+5 ans) | Mensuel (Y+5 ans) |
| **Settlement** | Physical (Henry Hub, LA) | Cash |
| **Exchange Fee** | ~$1.60/side | ~$0.50/side |
| **Maint. Margin** | ~$3,992 | ~$1,003 |
| **ADV 2024** | **566,000** (record) | ~15,000 (est.) |

### Notionnel par contrat (ordre de grandeur, mars 2026)

| | Prix approx | Notionnel (prix × multiplier) |
|---|---|---|
| CL | ~$70 | ~$70,000 |
| BZ | ~$74 | ~$74,000 |
| MCL | ~$70 | ~$7,000 |
| HO | ~$2.20/gal | ~$92,400 |
| RB | ~$2.10/gal | ~$88,200 |
| NG | ~$4.00/mmBtu | ~$40,000 |
| QG | ~$4.00/mmBtu | ~$10,000 |

### Session Globex (Énergie)
- **Ouverture** : Dimanche 17h00 CT → Vendredi 16h00 CT
- **Session quotidienne** : 17h00 CT → 16h00 CT (pause 16h00-17h00 CT, 60 min)
- **Pas de pause 15h15-15h30** (contrairement aux indices)

### Paires tradées (Énergie)

**Crude Oil Spreads :**
- **CL/BZ** : WTI-Brent spread (generalement WTI < Brent)

**Crack Spreads (refinery margin) :**
- **CL/HO** : 1:1 crack (WTI vs ULSD). `Crack = HO x 42 - CL` (convertir $/gal en $/bbl)
- **CL/RB** : 1:1 crack (WTI vs RBOB). `Crack = RB x 42 - CL`
- **3:2:1 Crack** : 3 CL vs 2 RB + 1 HO (simule la marge d'une raffinerie)

**Autres :**
- **CL/NG** : crude vs natgas (energy ratio), mais cointegration faible (R² Kalman ~ 0)

### Notes importantes

- **CL/BZ** : Même contract size (1000 bbl) et même tick structure → sizing simplifié (ratio ~1:1 avant beta)
- **HO/RB cotés en $/gal** : Multiplier par 42 pour obtenir $/bbl et comparer avec CL
- **BZ et MCL sont cash-settled** : pas de risque de livraison physique
- **CL, HO, RB, NG sont physically delivered** : rollover OBLIGATOIRE avant First Notice Date (FND)
- **Micro Brent (MBZ) n'existe PAS** : la plus petite taille Brent = BZ standard (1,000 bbl)
- Front month : `CLJ26_FUT_CME` / `BZJ26_FUT_CME` (avril 2026)
