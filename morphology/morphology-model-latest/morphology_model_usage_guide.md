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

