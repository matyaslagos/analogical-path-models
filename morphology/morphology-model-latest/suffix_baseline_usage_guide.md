# Usage guide for [`suffix_baseline.py`](https://github.com/matyaslagos/analogical-path-models/blob/main/suffix_baseline.py)

(`suffix_baseline.py` was vibe coded with Claude Code)

`suffix_baseline.py` implements and compares suffix-based methods for morphological
reinflection on the [SIGMORPHON 2016 shared task](https://github.com/ryancotterell/sigmorphon2016)
(task 1: given a lemma and a target morphosyntactic description, produce the
inflected form). It is a pure-stdlib, functional reimplementation and extension of
the suffix-based baseline of Liu & Mao (2016), section 2.

It learns suffix rewrite rules from `lemma -> form` pairs and applies the longest
rule that matches a word's ending. Four inflection methods are provided, differing
in how they break ties and whether they pivot through other paradigm cells:

- **`inflect`** — baseline: ties broken by token frequency, then alphabetically.
- **`inflect_reliable`** — ties broken by rule *reliability* (`hits / scope`,
  i.e. the fraction of applicable training cases the rule predicts correctly),
  then frequency, then alphabetically.
- **`inflect_indirect`** — guesses via the single most reliable *pivot path*
  through another paradigm cell (reliability = product of the two steps).
- **`inflect_indirect_weighted`** — pools reliability-weighted votes over all
  pivot paths.

The data is expected under `sigmorphon_data/<language>-task1-<split>`, where
`<split>` is one of `train`, `dev`, `test`, `test-covered`.

## Quick start: compare all methods

Run the module directly to train on every language's training set and print
dev-set accuracy for all four methods (~2 min):

```bash
python3 suffix_baseline.py
```

## Loading data

Each data file is tab-separated (`lemma <TAB> msd <TAB> form`); `read_data`
returns a list of `(lemma, msd, form)` triples. For `test-covered` files the form
is withheld and comes back as `None`.

```python
import suffix_baseline as sb
train = sb.read_data('sigmorphon_data/finnish-task1-train')
dev   = sb.read_data('sigmorphon_data/finnish-task1-dev')
train[0]
# ('ääkköstää', 'pos=V,polar=POS,mood=IMP,tense=PRS,per=3,num=SG', 'ääkköstäköön')
```

## Direct methods: `inflect` and `inflect_reliable`

```python
model = sb.train(train)                          # rule table, keyed by target MSD
reliabilities = sb.compute_reliabilities(train, model)

msd = 'pos=N,case=IN+ESS,num=PL'                 # inessive plural ("in the …s")
sb.inflect(model, 'talo', msd)                   # baseline           -> 'taloissa'
sb.inflect_reliable(model, reliabilities, 'talo', msd)  # reliability -> 'taloissa'
```

`train` returns a nested dict `msd -> source_suffix -> Counter(target_suffix -> token frequency)`;
`compute_reliabilities` returns `msd -> source_suffix -> target_suffix -> reliability`.

## Indirect methods: pivoting through other cells

`train_indirect` reconstructs each lemma's paradigm and learns `form -> form`
substitutions between every pair of cells (a special `LEMMA_CELL` cell holds the
citation form, so the ordinary direct guess is just the lemma-pivot path). It
returns everything the indirect methods need.

```python
cross_model, cross_reliabilities, paradigms, sources_by_target = sb.train_indirect(train)

# single most reliable pivot path:
sb.inflect_indirect(cross_model, cross_reliabilities, paradigms,
                    sources_by_target, 'talo', msd)
# reliability-weighted vote across all pivot paths:
sb.inflect_indirect_weighted(cross_model, cross_reliabilities, paradigms,
                             sources_by_target, 'talo', msd)
```

## Evaluating

`evaluate` scores any prediction function `(lemma, msd) -> form` against gold
`(lemma, msd, form)` triples, returning `(n_correct, n_total, errors)`:

```python
correct, total, errors = sb.evaluate(
    lambda lemma, msd: sb.inflect_reliable(model, reliabilities, lemma, msd), dev)
correct, total                     # (1438, 1598)
errors[0]
# {'lemma': 'adrenomedulliini', 'msd': 'pos=N,case=IN+ESS,num=PL',
#  'gold': 'adrenomedulliineissä', 'guess': 'adrenomedulliineissa'}
```

`run_language` does the whole pipeline for one language — load, train (direct and
indirect), and evaluate all four methods on a split:

```python
result = sb.run_language('finnish')              # eval_split='dev' by default
result['reliable_accuracy'], result['reliable_correct'], result['total']
# (0.8998748435544431, 1438, 1598)
# also: baseline_*, indirect_*, weighted_* ; and 'language', 'split'

sb.run_language('turkish', eval_split='test')    # evaluate on the test split
```

## Inspecting a method's reasoning

To see *why* a method made a prediction — which word forms contributed and by how
much — use `explain_direct` (for `inflect` / `inflect_reliable`) and
`explain_indirect` (for `inflect_indirect` / `inflect_indirect_weighted`). Both
return plain dicts that print nicely with `pprint`.

`explain_direct` shows the longest lemma suffix that matched a rule and every
target suffix competing for it, each with the word form it yields, its token
`frequency` and its `reliability`. Pass `train_triples` to also list the training
`(lemma, form)` pairs each substitution comes from:

```python
exp = sb.explain_direct(model, reliabilities, 'galaktoosi',
                        'pos=N,case=TRANS,num=PL', train_triples=train)
exp['source_suffix'], exp['applicable']        # ('si>', 9)
exp['baseline_choice'], exp['reliable_choice'] # ('galaktooeiksi', 'galaktooseiksi')
for c in exp['candidates']:
    print(c['form'], c['frequency'], round(c['reliability'], 3), len(c['contributing_forms']))
# galaktooseiksi 3 1.0   9     <- reliable_choice: 9 training words use -si -> -seiksi
# galaktooeiksi  5 0.0   0     <- baseline_choice: more frequent, but never correct here
# galaktooksi    1 0.0   0
exp['candidates'][0]['contributing_forms'][:2]
# [('fideikomissi', 'fideikomisseiksi'), ('jänöjussi', 'jänöjusseiksi')]
```

This is a good illustration of why reliability helps: the baseline picks the more
*frequent* `-eiksi>` substitution (wrong), while the reliable method picks
`-seiksi>`, which is borne out by 9 training word forms.

`explain_indirect` shows, for each candidate form, the pivot word forms that
voted for it. Each path reports the cell it came from (`via_msd`), the pivot form,
its combined `reliability`, `frequency` and `weighted_vote`. The candidate's
`best_reliability` is what `inflect_indirect` ranks by, and `weighted_vote` is what
`inflect_indirect_weighted` ranks by:

```python
exp = sb.explain_indirect(cross_model, cross_reliabilities, paradigms,
                          sources_by_target, 'talo', 'pos=N,case=IN+ESS,num=PL')
exp['weighted_choice']                         # 'taloissa'
top = exp['candidates'][0]
top['form'], round(top['weighted_vote'], 2), len(top['paths'])   # ('taloissa', 20.0, 19)
for p in top['paths'][:3]:
    print(p['via_msd'], p['pivot_form'], round(p['reliability'], 2), p['frequency'])
# pos=N,case=IN+ABL,num=PL  taloista   1.0  4    <- elative plural "taloista" contributes most
# pos=N,case=NOM,num=PL     talot      1.0  2
# pos=N,case=PRIV,num=PL    taloitta   1.0  2
```

## Reference: constants

- `sb.LANGUAGES` — all ten task-1 languages.
- `sb.SUFFIXING_LANGUAGES` — the seven largely suffixing ones (where the method is
  expected to do well).
- `sb.BEGIN`, `sb.END` — the `'<'` / `'>'` word-boundary markers used inside rules.
- `sb.LEMMA_CELL` — the sentinel MSD standing for the citation (lemma) form.
