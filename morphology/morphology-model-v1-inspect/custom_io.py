import csv
import re
from collections import Counter
from collections import defaultdict
from string import ascii_lowercase
from pprint import pp

#> Importing SZTAKI .tsv corpus <#

def sztaki_tsv_nouns_import():
    """Import nouns from a SZTAKI cleaned corpus as frequency dict.

    Arguments:
        None, defaults to a particular corpus.

    Returns:
        freqs (Counter): dict of nouns and their frequencies
    """
    freqs = Counter()
    sztaki_corpus_path = 'corpora/sztaki_corpus_2017_2018_0001_clean.tsv'
    with open(sztaki_corpus_path, newline='') as f:
        reader = csv.reader(
            (row for row in f if row.strip() and not row.startswith('#')),
            delimiter='\t'
        )
        next(reader, None) # skip first line
        is_hun_char = lambda x: x.lower() in ascii_lowercase + 'áéíóúöőüű'
        is_hun_string = lambda x: all(map(is_hun_char, x))
        for row in reader:
            if len(row) >= 4 and row[3].startswith('[/N]') and is_hun_string(row[0]):
                word_form = '<' + hun_encode(row[0].lower()) + '>'
                tags = xpostag_set(row[3])
                freqs[(word_form, tags)] += 1
    return freqs

#> Encoding Hungarian multiletter sounds <#

hun_encodings = {
    'ccs': '11',
    'ddz': '22',
    'ggy': '33',
    'nny': '44',
    'ssz': '55',
    'tty': '66',
    'zzs': '77',
    'dzs': '8',
    'lly': 'jj',
    'cs': '1',
    'dz': '2',
    'gy': '3',
    'ny': '4',
    'sz': '5',
    'ty': '6',
    'zs': '7',
    'ly': 'j'
}

encode_pattern = re.compile(
    '|'.join(re.escape(k) for k in sorted(hun_encodings, key=len, reverse=True))
)

def hun_encode(word):
    return encode_pattern.sub(lambda x: hun_encodings[x[0]], word)

hun_decodings = {value: key for key, value in hun_encodings.items() if 'j' not in value}

decode_pattern = re.compile(
    '|'.join(re.escape(k) for k in sorted(hun_decodings, key=len, reverse=True))
)

def hun_decode(word):
    return decode_pattern.sub(lambda x: hun_decodings[x[0]], word)

#> Prettyprinting dictionaries <#

def sorted_items(dy, mapping=lambda x: x):
    mapped_items = ((mapping(key), value) for key, value in dy.items())
    return sorted(mapped_items, key=lambda x: x[1], reverse=True)

def dict_to_list_hun_decode(dy):
    return sorted_items(dy, mapping=lambda x: hun_decode(''.join(x)))

def custom_pp(list):
    pp([(hun_decode(''.join(x[0])), x[1]) for x in list])

#> Parsing paradigm cell tags in tagged SZTAKI corpus <#

def xpostag_set(xpostag, include_pos=False):
    """Convert xpostag into frozenset of cell tags.

    Arguments:
        - xpostag (string): e.g. '[/N][Poss.1Pl][Abl]'
        - include_pos (bool): whether to include first tag (general pos)

    Returns:
        - frozenset (of strings): e.g. {'Poss', '1Pl', 'Abl'}
    """
    return frozenset(re.findall(r'[^.\[\]]+', xpostag)[not include_pos:])

#> Old csv importer <#

def csv_to_wordfreqdict(filename):
    """Import filename as a dict of words (tuples of characters) with int values.
    """
    with open(filename, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        clean_word = lambda s: ('<',) + tuple(s.strip(punctuation).lower()) + ('>',)
        return {clean_word(row['key']): int(row['value']) for row in reader}

def sztaki_tsv_nouns_by_lemmas_import():
    """Import nouns from a SZTAKI cleaned corpus as dict, by lemmmas.

    Arguments:
        None, defaults to a particular corpus.

    Returns:
        freqs (Counter): dict of nouns and their frequencies
    """
    lemmas = defaultdict(Counter)
    sztaki_corpus_path = 'corpora/sztaki_corpus_2017_2018_0001_clean.tsv'
    with open(sztaki_corpus_path, newline='') as f:
        reader = csv.reader(
            (row for row in f if row.strip() and not row.startswith('#')),
            delimiter='\t'
        )
        next(reader, None) # skip first line
        is_hun_char = lambda x: x.lower() in ascii_lowercase + 'áéíóúöőüű'
        is_hun_string = lambda x: all(map(is_hun_char, x))
        for row in reader:
            if len(row) >= 4 and row[3].startswith('[/N]') and is_hun_string(row[0]):
                word_form = '<' + hun_encode(row[0].lower()) + '>'
                lemma = hun_encode(row[2].lower())
                tag = xpostag_set(row[3])
                lemmas[lemma][(word_form, tag)] += 1
    return lemmas

def sztaki_tsv_noun_tag_wordform_lemma_import():
    sztaki_corpus_path = 'corpora/sztaki_corpus_2017_2018_0001_clean.tsv'
    with open(sztaki_corpus_path, newline='') as f:
        reader = csv.reader(
            (row for row in f if row.strip() and not row.startswith('#')),
            delimiter='\t'
        )
        next(reader, None) # skip first line
        is_hun_char = lambda x: x.lower() in ascii_lowercase + 'áéíóúöőüű'
        is_hun_string = lambda x: all(map(is_hun_char, x))
        for row in reader:
            if len(row) >= 4 and row[3].startswith('[/N]') and is_hun_string(row[0]):
                word_form = hun_encode(row[0].lower())
                lemma = hun_encode(row[2].lower())
                tag = xpostag_set(row[3])
                yield (tag, word_form, lemma)

def sztaki_tsv_noun_tag_wordform_lemma_import_test():
    sztaki_corpus_path = 'corpora/sztaki_corpus_2017_2018_0002_clean.tsv'
    with open(sztaki_corpus_path, newline='') as f:
        reader = csv.reader(
            (row for row in f if row.strip() and not row.startswith('#')),
            delimiter='\t'
        )
        next(reader, None) # skip first line
        is_hun_char = lambda x: x.lower() in ascii_lowercase + 'áéíóúöőüű'
        is_hun_string = lambda x: all(map(is_hun_char, x))
        for row in reader:
            if len(row) >= 4 and row[3].startswith('[/N]') and is_hun_string(row[0]):
                word_form = hun_encode(row[0].lower())
                lemma = hun_encode(row[2].lower())
                tag = xpostag_set(row[3])
                yield (tag, word_form, lemma)