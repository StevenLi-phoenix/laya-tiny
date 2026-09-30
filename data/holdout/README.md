# Holdout

`tickets.jsonl` is used **only** for final evaluation — never for training, tokenizer fitting, early stopping or
calibration. The `split` stage also drops any corpus row whose word-3-gram Jaccard with a holdout row is ≥ 0.6.

| source | rows | languages |
|---|---|---|
| `laya-skill/tickets.jsonl@v0.2.0` | 40 | 21 en, 10 zh, 9 es |
| `laya-tiny/v1-authored` | 184 | en |

The 184 `v1-authored` rows were written and labelled by an AI assistant (Claude) against the rubric below while
building this project. They have **not** been independently re-labelled by a human; treat single-digit accuracy
differences accordingly, and prefer teacher agreement for fidelity questions.

English balance: department 52/52/51/50 (billing/technical/account/sales), urgency 89/75/41 (0/1/2), churn 39/205.

## Labelling policy

- **department** — billing: invoices, charges, refunds, payment methods · technical: bugs, crashes, errors, outages,
  performance · account: login, password, profile, permissions, deleting the account · sales: pricing, plans,
  upgrades, quotes. A failed payment that locks the workspace is billing; a crashed checkout page is technical.
- **urgency** — 0: a question or minor annoyance · 1: something is broken but there is a workaround, or a mis-charge
  · 2: work or money is stopped right now. A stated threat to leave counts as at least 1 (as in the original 40).
- **churn_risk** — true only when the customer says they will cancel, leave, not renew or switch. Prospects
  comparing vendors are false.

Row ids are `h-` + the first 12 hex chars of SHA-1 of the text (checked by `tests/test_holdout.py`).
