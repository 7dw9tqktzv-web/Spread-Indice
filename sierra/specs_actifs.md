# Specs Actifs — Futures Indices US (E-mini)

Source : CME Group (vérifié le 2026-02-20)

## Contract Specs

| | NQ | ES | RTY | YM |
|---|---|---|---|---|
| **Nom complet** | E-mini Nasdaq-100 | E-mini S&P 500 | E-mini Russell 2000 | E-mini Dow Jones |
| **Exchange** | CME | CME | CME | CME |
| **Globex Code** | NQ | ES | RTY | YM |
| **Multiplier ($/pt)** | $20 | $50 | $50 | $5 |
| **Tick Size** | 0.25 pts | 0.25 pts | 0.10 pts | 1.00 pt |
| **Tick Value** | $5.00 | $12.50 | $5.00 | $5.00 |
| **Commission** | ~$2.50/side | ~$2.50/side | ~$2.50/side | ~$2.50/side |

## Notionnel par contrat (ordre de grandeur, fév 2026)

| | Prix approx | Notionnel (prix × multiplier) |
|---|---|---|
| NQ | ~21,500 | ~$430,000 |
| ES | ~6,100 | ~$305,000 |
| RTY | ~2,250 | ~$112,500 |
| YM | ~44,500 | ~$222,500 |

## Implications pour le sizing

- **NQ est le plus gros notionnel** (~$430k) → souvent leg_a avec N_a=1
- **RTY est le plus petit** (~$112k) → N_b sera élevé quand paired avec NQ (ratio ~3-4x)
- La formule : `N_b = round((Not_A / Not_B) × β × N_a)`
- Exemple NQ/RTY : `N_b ≈ round((430k / 112k) × 1.5 × 1) ≈ 6 contrats`

## Session Globex
- **Ouverture** : Dimanche 17h00 CT → Vendredi 16h00 CT
- **Session quotidienne** : 17h00 CT → 16h00 CT (avec pause 15h15-15h30 CT pour ES/NQ/RTY/YM)
- **Notre fenêtre** : 17h30-15h30 CT (buffer 30min), trading 04h00-14h00 CT

## 6 paires tradées
NQ/ES, NQ/RTY, NQ/YM, ES/RTY, ES/YM, RTY/YM
