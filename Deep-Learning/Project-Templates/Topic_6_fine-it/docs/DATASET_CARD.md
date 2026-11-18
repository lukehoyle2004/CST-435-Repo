# Dataset card: FineTuneFleet routing corpus (Banking77 → 8 queues)

## Source

| | |
|---|---|
| Dataset | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), pinned to commit `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` |
| Licence | CC-BY-4.0 (Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence Encoders") |
| Content | 13,083 short English customer-service questions to a digital bank, each labelled with one of 77 intents |
| Who wrote them | Crowd-sourced to imitate real customers; **not** real customer messages and contains no personal data |

## What this project changed

1. **Labels: 77 intents → 8 queues.** [`shared/queues.py`](../shared/queues.py) maps every intent to
   exactly one support queue by three written rules, applied in order:
   1. anything the customer did not authorise, or a lost/stolen/compromised card or phone →
      `security_fraud`;
   2. any fee, charge or exchange-rate question → `fees_fx`;
   3. otherwise, the team that owns the product.

   Queue size in intents: card services 19, fees & FX 12, transfers 10, account & identity 10,
   top-ups 9, payments & disputes 6, security & fraud 6, cash & ATM 5.
2. **Cleaning.** Whitespace collapsed. Exact duplicates were removed *after* lowercasing and
   stripping punctuation: 31 duplicates within train, 4 within test, and **25 train messages that also
   appear in test** (so nothing the model trained on is scored as unseen).
3. **Splits.** The original test split is kept as **test** (3,076 messages). A stratified 10% of
   the original train (by intent, seed 42) becomes **validation** (995); the rest is **train**
   (8,952). Every intent appears in every split, and no message (after normalization) is in more
   than one split; `tests/test_data.py` checks both.

| Queue | Train | Validation | Test |
|---|---:|---:|---:|
| card_services | 1,887 | 210 | 760 |
| fees_fx | 1,557 | 173 | 479 |
| transfers | 1,265 | 142 | 400 |
| top_ups | 1,043 | 115 | 358 |
| account_identity | 1,026 | 113 | 400 |
| payments_disputes | 879 | 98 | 240 |
| security_fraud | 718 | 80 | 240 |
| cash_atm | 577 | 64 | 199 |
| **Total** | **8,952** | **995** | **3,076** |

**Lengths** (DistilBERT word-pieces): median 13, 95th percentile 37 (train); the longest message
is 98. The service's `max_length` of 128 truncates nothing.

## Where it lives

* `data/corpus.csv`: one row per message, with columns `doc_id, split, queue, intent, text,
  text_sha256, n_chars, n_tokens`. It is written by `python -m scripts.prepare_data`, which is
  reproducible from the pinned dataset revision.
* Supabase `ftf_documents`: the same 13,023 rows, loaded with `--push`. They are readable only with
  the service-role key; the public anon key is refused.
* [`docs/data/dataset_stats.json`](data/dataset_stats.json): every count on this page.

## Known limitations

* **Clean and single-intent.** Real support messages are longer, have typos, mix two problems
  ("my card was declined and I was charged a fee") and arrive in other languages. Scores on
  Banking77 are an upper bound for production.
* **Queue imbalance.** Card services has 3.3× the training examples of cash & ATM. Macro-F1
  (every queue counts equally) is the headline metric for that reason.
* **The mapping is a design decision.** "Charged a fee on a cash withdrawal" goes to fees & FX
  by rule 2, not to cash & ATM. A different bank might draw the line elsewhere, and the model
  would need retraining to match.
* **Synthetic customers.** The writing style is that of crowd workers imagining a bank, which
  may under-represent how stressed, elderly or non-native customers write.
