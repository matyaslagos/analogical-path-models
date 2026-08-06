# Usage guide for [`morphology_model.py`](https://github.com/matyaslagos/analogical-path-models/blob/main/morphology_model.py)

## Setup and usage of some functions

Setting up the model:
```python
# File custom_io.py should be in same folder as morphology_model.py
import morphology_model as mor
model = mor.MorphModel()
model.setup() # ~10 secs, needs file with path 'corpora/sztaki_corpus_2017_2018_0001_clean.tsv'
```
Generating possible {Pl, Acc} word forms of the lemma "november":
```python
word_forms = mor.inflect(model, 'november', {'Pl', 'Acc'})
word_forms[0] # tuple of best word form and score
```
Producing a word form together with the ending substitutions that built it:
```python
>>> from pprint import pp # pretty printing
>>> result = mor.produce_word_inspected(model, 'november', {'Pl', 'Acc'})
>>> result['produced word'] # guessed word form ('' if the guess fails or ties)
'novembereket'
>>> result['score'] # goodness score of the produced word form
20.373859801942427
>>> pp(result['contributions'][0]) # substitution that contributed the most
{'tag': {'Nom'},
 'substitution': 'er -> ereket',
 'reliability': 1.0,
 'attested by': [('műszer', 'műszer', 'műszereket'),
                 ('per', 'per', 'pereket'),
                 ('hangszer', 'hangszer', 'hangszereket')]}
```
`'contributions'` is the list of ending substitutions that produced the winning
word form, sorted by `'reliability'` (highest first). Each entry records the
analogical source `'tag'`, the decoded ending `'substitution'` (`e -> e'`, with
`∅` for an empty ending), its `'reliability'` score `P_{t->g}(e -> e')`, and the
`'attested by'` list of `(lemma, source-tag form, target-tag form)` triples that
attest the substitution.

Carrying out testing:
```python
test_corpus = mor.import_test_data() # needs file with path 'corpora/sztaki_corpus_2017_2018_0002_clean.tsv'
results = mor.testing(model, test_corpus[:5000]) # ~1min
len(results[True]) # number of correct guesses
len(results[False]) # number of incorrect guesses
len(results['UNK']) # number of unguessable items
```
(Unguessable items are {Nom} word forms of those lemmas that are either unattested or only attested with their {Nom} word forms.)

Inspecting the test results:
```python
>>> from pprint import pp # pretty printing
>>> pp(results[False][0])
{'target word': 'rituáléhoz',
 'produced word': 'rituáléhez',
 'tag': {'All'},
 'lemma': 'rituálé',
 'lemmafreq': 7}
```

