# Specs Actifs — Futures

Source : CME Group (vérifié le 2026-03-03)

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

## 2. Énergie — Crude Oil

### Contract Specs

| | CL | BZ |
|---|---|---|
| **Nom complet** | Light Sweet Crude Oil (WTI) | Brent Last Day Financial |
| **Exchange** | NYMEX | NYMEX |
| **Globex Code** | CL | BZ |
| **Sierra Symbol** | `CL?##_FUT_CME` | `BZ?##_FUT_CME` |
| **Contract Size** | 1,000 barrels | 1,000 barrels |
| **Multiplier ($/pt)** | $1,000 | $1,000 |
| **Tick Size** | $0.01/barrel | $0.01/barrel |
| **Tick Value** | $10.00 | $10.00 |
| **Settlement** | Physical Delivery | Cash (Financial) |
| **Exchange Fee** | ~$1.21/side | ~$0.77/side |

### Notionnel par contrat (ordre de grandeur, mars 2026)

| | Prix approx | Notionnel (prix × multiplier) |
|---|---|---|
| CL | ~$70 | ~$70,000 |
| BZ | ~$74 | ~$74,000 |

### Session Globex (Énergie)
- **Ouverture** : Dimanche 17h00 CT → Vendredi 16h00 CT
- **Session quotidienne** : 17h00 CT → 16h00 CT (pause 16h00-17h00 CT, 60 min)
- **Pas de pause 15h15-15h30** (contrairement aux indices)

### Paire tradée (Énergie)
CL/BZ (WTI vs Brent spread)

### Notes CL/BZ
- **Même contract size** (1000 barrels) et **même tick structure** → sizing simplifié (ratio ~1:1 avant beta)
- BZ est **cash-settled** (pas de risque de livraison), CL est **physically delivered** (rollover avant FND)
- Le spread CL-BZ reflète le **WTI-Brent differential** (généralement WTI < Brent → spread négatif)
- Contrats mensuels disponibles sur plusieurs années
- Front month mars 2026 : `CLJ26_FUT_CME` / `BZJ26_FUT_CME` (avril 2026)
