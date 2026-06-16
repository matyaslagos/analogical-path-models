"""Suffix-based baseline for morphological reinflection (SIGMORPHON 2016, task 1).

(Vibe coded with Claude Code.)

This is a faithful, minimal reimplementation of the "suffix-based baseline"
described in section 2 of:

    Ling Liu and Lingshuang Jack Mao (2016),
    "Morphological Reinflection with Conditional Random Fields and
    Unsupervised Features", Proceedings of SIGMORPHON 2016.

The idea is simple. For each training example `lemma -> form` (under some target
morphosyntactic description, MSD), we Levenshtein-align the two strings and read
off every "suffix rewrite rule": a mapping from a suffix of the source to the
corresponding suffix of the target. At test time, to inflect a lemma for a given
MSD, we apply the *longest* learned source suffix that matches the end of the
lemma. The bet is that, in suffixing languages, the end of a word predicts its
inflected form.

In task 1 the source is always the lemma, so rules are keyed by the target MSD.

The model is just nested dictionaries; everything here is plain functions.
"""

import os
from collections import Counter, defaultdict, namedtuple

# Start- and end-of-word markers. They let stems align fully despite prefixation
# and suffixation, and they anchor where a "suffix" begins and ends.
BEGIN, END = '<', '>'

LANGUAGES = [
    'arabic', 'finnish', 'georgian', 'german', 'hungarian',
    'maltese', 'navajo', 'russian', 'spanish', 'turkish',
]

# The largely suffixing languages, on which this method is expected to do well.
SUFFIXING_LANGUAGES = [
    'finnish', 'georgian', 'german', 'hungarian', 'russian', 'spanish', 'turkish',
]

#----------------#
# Data loading   #
#----------------#

def read_data(path):
    """Read a SIGMORPHON task-1 file as a list of (lemma, msd, form) triples.

    Each line is tab-separated: `lemma <TAB> msd <TAB> form`. In the
    `test-covered` files the inflected form is withheld, so `form` is None.
    """
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            lemma, msd = fields[0], fields[1]
            form = fields[2] if len(fields) > 2 else None
            triples.append((lemma, msd, form))
    return triples

#----------------#
# Alignment      #
#----------------#

def align(source, target):
    """Align two strings into a list of (src_char, tgt_char) columns.

    Either side of a column may be '' (the empty string), denoting a gap:
    `(c, '')` is a deletion (source char with no counterpart) and `('', c)` is
    an insertion (target char with no counterpart). Matching characters line up
    as `(c, c)`.

    This is an LCS-style alignment: differing characters are never paired up as
    a substitution; instead the shorter common subsequence is matched and
    everything else is treated as an insertion or a deletion. This matches the
    alignments shown in the paper (e.g. `rakko -> rakoitta` deletes a `k` and
    inserts `itta` rather than substituting), and it keeps the extracted suffix
    rules clean.

        align('<rakko>', '<rakoitta>') ==
            [('<','<'), ('r','r'), ('a','a'), ('k','k'),
             ('k',''),                       # the second k is deleted
             ('o','o'),
             ('','i'), ('','t'), ('','t'), ('','a'),   # "itta" is inserted
             ('>','>')]
    """
    n, m = len(source), len(target)
    # dp[i][j] = number of insertions + deletions needed to turn source[:i]
    # into target[:j], where matching characters are free and substitutions
    # are disallowed (so a mismatch costs one deletion plus one insertion).
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]              # free match
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],         # delete from source
                                   dp[i][j - 1])         # insert into target
    # Walk back from (n, m) to (0, 0), taking matches whenever possible. Build
    # the columns right-to-left, then reverse.
    columns = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1] == target[j - 1]:
            columns.append((source[i - 1], target[j - 1]))   # match
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            columns.append((source[i - 1], ''))              # deletion
            i -= 1
        else:
            columns.append(('', target[j - 1]))              # insertion
            j -= 1
    columns.reverse()
    return columns

#---------------------#
# Rule extraction     #
#---------------------#

def extract_rules(columns):
    """Read all suffix rewrite rules off an alignment.

    Each real source character "owns" a piece of the target: its own aligned
    character (empty if it was deleted) preceded by any insertions that sit
    directly before it. A rule then maps a source suffix to the concatenation
    of the target pieces owned by that suffix.

    For the `<rakko>` / `<rakoitta>` alignment above this yields exactly the
    rules listed in the paper:

        >       -> itta>
        o>      -> oitta>
        ko>     -> oitta>
        kko>    -> koitta>
        akko>   -> akoitta>
        rakko>  -> rakoitta>
        <rakko> -> <rakoitta>
    """
    # For each real source char, the target string it owns (in source order).
    owned = []
    pending_insertions = ''
    for src_char, tgt_char in columns:
        if src_char == '':                     # insertion: no source char yet
            pending_insertions += tgt_char     # ... attach it to the next one
        else:                                  # match, substitution or deletion
            owned.append((src_char, pending_insertions + tgt_char))
            pending_insertions = ''
    # Grow the suffix from the right, emitting one rule per source character.
    rules = []
    source_suffix, target_suffix = '', ''
    for src_char, target_piece in reversed(owned):
        source_suffix = src_char + source_suffix
        target_suffix = target_piece + target_suffix
        rules.append((source_suffix, target_suffix))
    return rules

#----------------#
# Training       #
#----------------#

def train(triples):
    """Build the rule table from (lemma, msd, form) triples.

    Returns a nested dict:
        msd -> source_suffix -> Counter(target_suffix -> token frequency)

    The token frequency is how many training examples produced that exact
    rewrite rule; it is used to break ties at inflection time.
    """
    model = defaultdict(lambda: defaultdict(Counter))
    for lemma, msd, form in triples:
        if form is None:
            continue
        columns = align(BEGIN + lemma + END, BEGIN + form + END)
        for source_suffix, target_suffix in extract_rules(columns):
            model[msd][source_suffix][target_suffix] += 1
    return model

def compute_reliabilities(triples, model):
    """Compute the reliability of every rule in `model` (Albright & Hayes style).

    A rule's reliability is: of all training pairs to which the rule *could be
    applied* (those whose marked lemma ends in the rule's source suffix), the
    fraction for which applying it (replacing that suffix with the target
    suffix) reproduces the correct form. For example, given the past-tense pairs
    walk->walked, jump->jumped and go->went, the rule `> -> ed>` applies to all
    three and is correct for two of them, so its reliability is 2/3.

    Note that "applying a rule" means plain suffix replacement -- exactly how
    `inflect` uses rules at test time -- so reliability measures a rule's actual
    success rate. It can differ from a rule's token frequency (how often it was
    extracted) whenever a word's stem changes outside the matched suffix.

    Returns a nested dict:
        msd -> source_suffix -> target_suffix -> reliability (float in [0, 1])
    """
    # scope[msd][S]    = # training pairs whose marked lemma ends in suffix S
    # hits[msd][S][T]  = # of those pairs for which replacing S with T is correct
    scope = defaultdict(Counter)
    hits = defaultdict(lambda: defaultdict(Counter))
    for lemma, msd, form in triples:
        if form is None:
            continue
        marked_lemma = BEGIN + lemma + END
        marked_form = BEGIN + form + END
        for length in range(1, len(marked_lemma) + 1):
            source_suffix = marked_lemma[len(marked_lemma) - length:]
            stem = marked_lemma[:len(marked_lemma) - length]
            scope[msd][source_suffix] += 1
            # The rule is correct here iff the form is the stem plus some suffix;
            # if so, that leftover suffix is the target suffix it would produce.
            if marked_form.startswith(stem):
                produced_suffix = marked_form[len(stem):]
                hits[msd][source_suffix][produced_suffix] += 1
    # Turn the counts into reliabilities, for exactly the rules we extracted.
    reliabilities = defaultdict(lambda: defaultdict(dict))
    for msd, source_rules in model.items():
        for source_suffix, target_counts in source_rules.items():
            applicable = scope[msd][source_suffix]
            for target_suffix in target_counts:
                correct = hits[msd][source_suffix][target_suffix]
                reliabilities[msd][source_suffix][target_suffix] = correct / applicable
    return reliabilities

#----------------#
# Inflection     #
#----------------#

def best_target_suffix(target_counts):
    """Choose the winning target suffix for one source suffix.

    Ties are broken by token frequency (more frequent rules win), and any
    remaining ties are broken alphabetically.
    """
    return min(target_counts, key=lambda suffix: (-target_counts[suffix], suffix))

def inflect(model, lemma, msd):
    """Inflect `lemma` for the target `msd` using the longest matching rule.

    If the MSD was never seen in training, fall back to returning the lemma
    unchanged (an identity guess).
    """
    rules = model.get(msd)
    if not rules:
        return lemma
    marked = BEGIN + lemma + END
    # Try the longest source suffix first, shortening until a rule matches. The
    # smallest possible rule has source suffix '>', which is shared by every
    # word, so a match is essentially always found for a seen MSD.
    for length in range(len(marked), 0, -1):
        source_suffix = marked[len(marked) - length:]
        if source_suffix in rules:
            target_suffix = best_target_suffix(rules[source_suffix])
            result = marked[:len(marked) - length] + target_suffix
            return result.removeprefix(BEGIN).removesuffix(END)
    return lemma

def best_target_suffix_reliable(target_counts, target_reliabilities):
    """Choose the winning target suffix using reliability as the primary key.

    Ties are then broken by token frequency, and any remaining ties
    alphabetically. (Missing reliabilities default to 0.)
    """
    def key(suffix):
        return (-target_reliabilities.get(suffix, 0.0), -target_counts[suffix], suffix)
    return min(target_counts, key=key)

def inflect_reliable(model, reliabilities, lemma, msd):
    """Like `inflect`, but pick among tied rules by reliability first.

    This is identical to the baseline `inflect` except for the tie-breaking
    rule, so the two share the same longest-matching-suffix strategy and rule
    inventory and can be compared directly.
    """
    rules = model.get(msd)
    if not rules:
        return lemma
    msd_reliabilities = reliabilities.get(msd, {})
    marked = BEGIN + lemma + END
    for length in range(len(marked), 0, -1):
        source_suffix = marked[len(marked) - length:]
        if source_suffix in rules:
            target_suffix = best_target_suffix_reliable(
                rules[source_suffix],
                msd_reliabilities.get(source_suffix, {}),
            )
            result = marked[:len(marked) - length] + target_suffix
            return result.removeprefix(BEGIN).removesuffix(END)
    return lemma

#--------------------#
# Indirect guessing  #
#--------------------#

# A sentinel "MSD" standing for the citation (lemma) form. Real MSDs are strings
# like 'pos=V,tense=PST', so this never collides with one.
LEMMA_CELL = '__lemma__'

def guess_form(rule_set, reliability_table, word):
    """Reliability-based guess from a single source `word` using one rule set.

    This is the same longest-matching-suffix strategy as `inflect_reliable`, but
    factored out so it can be applied to any source form (not only a lemma) and
    so it can report *how* reliable the guess was. Returns a triple
    (guessed form, reliability of the chosen rule, token frequency of that rule);
    if no rule matches, returns (word, 0.0, 0).
    """
    marked = BEGIN + word + END
    for length in range(len(marked), 0, -1):
        source_suffix = marked[len(marked) - length:]
        if source_suffix in rule_set:
            target_counts = rule_set[source_suffix]
            target_reliabilities = reliability_table.get(source_suffix, {})
            target_suffix = best_target_suffix_reliable(target_counts, target_reliabilities)
            result = marked[:len(marked) - length] + target_suffix
            form = result.removeprefix(BEGIN).removesuffix(END)
            return form, target_reliabilities.get(target_suffix, 0.0), target_counts[target_suffix]
    return word, 0.0, 0

def build_paradigms(triples):
    """Reconstruct each lemma's paradigm from training triples.

    Returns a dict `lemma -> {msd: form}`. Each paradigm also gets a special
    `LEMMA_CELL` entry holding the citation form (the lemma string itself), so
    that the lemma can serve as a source cell just like any inflected form.
    """
    paradigms = defaultdict(dict)
    for lemma, msd, form in triples:
        if form is None:
            continue
        paradigms[lemma].setdefault(msd, form)   # keep first if a cell repeats
    for lemma, cells in paradigms.items():
        cells.setdefault(LEMMA_CELL, lemma)
    return paradigms

def build_cross_training(paradigms):
    """Turn paradigms into cross-cell training triples for `train`.

    For every lemma and every ordered pair of its cells (a, b), emit a triple
    (form_a, (msd_a, msd_b), form_b). Keying by the MSD *pair* lets us reuse
    `train` and `compute_reliabilities` unchanged to learn, for each pair of
    cells, how the form of one cell is rewritten into the form of the other.

    Pairs whose target is the lemma cell are skipped: we never need to predict
    the citation form, only real inflected MSDs.
    """
    cross_triples = []
    for cells in paradigms.values():
        for source_msd, source_form in cells.items():
            for target_msd, target_form in cells.items():
                if target_msd == source_msd or target_msd == LEMMA_CELL:
                    continue
                cross_triples.append((source_form, (source_msd, target_msd), target_form))
    return cross_triples

def train_indirect(triples):
    """Build everything the indirect method needs from training triples.

    Returns (cross_model, cross_reliabilities, paradigms, sources_by_target),
    where `sources_by_target[m]` lists every source MSD from which a substitution
    into MSD `m` was observed (i.e. the cells we may pivot through to reach `m`).
    """
    paradigms = build_paradigms(triples)
    cross_triples = build_cross_training(paradigms)
    cross_model = train(cross_triples)
    cross_reliabilities = compute_reliabilities(cross_triples, cross_model)
    sources_by_target = defaultdict(list)
    for source_msd, target_msd in cross_model:
        sources_by_target[target_msd].append(source_msd)
    return cross_model, cross_reliabilities, paradigms, dict(sources_by_target)

# One pivot path's contribution to guessing a target form. `reliability` is the
# product of `pivot_reliability` (how reliably the pivot form was obtained) and
# `step_reliability` (how reliably the pivot predicts the target); `frequency` is
# the token frequency of the step-(b) rule.
PivotPath = namedtuple('PivotPath', [
    'form', 'reliability', 'frequency',
    'source_msd', 'pivot_form', 'pivot_reliability', 'step_reliability',
])

def indirect_candidates(cross_model, cross_reliabilities, paradigms,
                        sources_by_target, lemma, msd):
    """Collect one `PivotPath` per pivot path for guessing `lemma`'s `msd` form.

    For every source MSD `m'` from which `msd` can be reached:
      (a) obtain the lemma's `m'` form -- use it directly with reliability 1 if
          it is attested in training, otherwise guess it from the lemma with the
          reliability-based method (the `m' = LEMMA_CELL` path is the ordinary
          direct guess);
      (b) guess the `msd` form from that pivot form, again reliability-based.
    This is the shared raw material for the indirect methods (and for
    `explain_indirect`); each `PivotPath` records both the prediction and the
    word form that produced it.
    """
    candidates = []
    paradigm = paradigms.get(lemma, {})
    for source_msd in sources_by_target.get(msd, ()):
        # (a) pivot form of `lemma` for `source_msd`, with reliability r1.
        if source_msd == LEMMA_CELL or source_msd in paradigm:
            pivot_form = lemma if source_msd == LEMMA_CELL else paradigm[source_msd]
            r1 = 1.0
        else:
            direct_key = (LEMMA_CELL, source_msd)
            direct_rules = cross_model.get(direct_key)
            if not direct_rules:
                continue
            pivot_form, r1, _ = guess_form(
                direct_rules, cross_reliabilities.get(direct_key, {}), lemma)
        # (b) guess the target form from the pivot form, with reliability r2.
        key = (source_msd, msd)
        rule_set = cross_model.get(key)
        if not rule_set:
            continue
        form, r2, frequency = guess_form(
            rule_set, cross_reliabilities.get(key, {}), pivot_form)
        candidates.append(PivotPath(
            form=form, reliability=r1 * r2, frequency=frequency,
            source_msd=source_msd, pivot_form=pivot_form,
            pivot_reliability=r1, step_reliability=r2))
    return candidates

def inflect_indirect(cross_model, cross_reliabilities, paradigms, sources_by_target,
                     lemma, msd):
    """Indirect guessing that trusts the single most reliable pivot path.

    The candidate with the highest combined reliability wins, ties broken by
    frequency then alphabetically.
    """
    candidates = indirect_candidates(
        cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd)
    if not candidates:
        return lemma
    best = min(candidates, key=lambda c: (-c.reliability, -c.frequency, c.form))
    return best.form

def inflect_indirect_weighted(cross_model, cross_reliabilities, paradigms,
                              sources_by_target, lemma, msd):
    """Indirect guessing that pools reliability-weighted votes across pivot paths.

    Every pivot path votes for its candidate form, casting a number of votes
    equal to its token frequency scaled by its reliability, so unreliable paths
    contribute little and agreement only accumulates among trustworthy paths.
    The form with the greatest total weighted vote wins; ties are broken by total
    token frequency, then alphabetically.
    """
    candidates = indirect_candidates(
        cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd)
    if not candidates:
        return lemma
    weighted = defaultdict(float)        # form -> sum of frequency * reliability
    frequency_total = defaultdict(int)   # form -> total token frequency (tie-breaker)
    for path in candidates:
        weighted[path.form] += path.frequency * path.reliability
        frequency_total[path.form] += path.frequency
    return min(weighted, key=lambda form: (-weighted[form], -frequency_total[form], form))

#----------------#
# Inspection     #
#----------------#

def longest_matching_suffix(rule_set, word):
    """Return the longest suffix of marked `word` that has a rule, or None."""
    marked = BEGIN + word + END
    for length in range(len(marked), 0, -1):
        suffix = marked[len(marked) - length:]
        if suffix in rule_set:
            return suffix
    return None

def explain_direct(model, reliabilities, lemma, msd, train_triples=None):
    """Show the reasoning of the direct methods for one lemma and MSD.

    Returns a dict describing which suffix substitution produced the prediction.
    `source_suffix` is the longest suffix of the lemma that any rule matched, and
    `candidates` lists every target suffix learned for it (sorted by reliability,
    then frequency, then alphabetically), each with the word form it produces,
    its token `frequency` and its `reliability`. `baseline_choice` /
    `reliable_choice` are the actual predictions of `inflect` / `inflect_reliable`.

    If `train_triples` is given, each candidate also gets `contributing_forms` --
    the training `(lemma, form)` pairs that exhibit that substitution -- and the
    result reports `applicable`, the number of training pairs the matched suffix
    could apply to (so reliability = len(contributing_forms) / applicable).
    """
    result = {'lemma': lemma, 'msd': msd, 'source_suffix': None, 'candidates': [],
              'baseline_choice': inflect(model, lemma, msd),
              'reliable_choice': inflect_reliable(model, reliabilities, lemma, msd)}
    rule_set = model.get(msd)
    if not rule_set:
        return result
    source_suffix = longest_matching_suffix(rule_set, lemma)
    result['source_suffix'] = source_suffix
    target_reliabilities = reliabilities.get(msd, {}).get(source_suffix, {})
    marked = BEGIN + lemma + END
    stem = marked[:len(marked) - len(source_suffix)]

    contributing, applicable = {}, None
    if train_triples is not None:
        contributing, applicable = _contributing_forms(train_triples, msd, source_suffix)
        result['applicable'] = applicable

    for target_suffix, frequency in rule_set[source_suffix].items():
        form = (stem + target_suffix).removeprefix(BEGIN).removesuffix(END)
        candidate = {'form': form, 'target_suffix': target_suffix,
                     'frequency': frequency,
                     'reliability': target_reliabilities.get(target_suffix, 0.0)}
        if train_triples is not None:
            candidate['contributing_forms'] = contributing.get(target_suffix, [])
        result['candidates'].append(candidate)
    result['candidates'].sort(
        key=lambda c: (-c['reliability'], -c['frequency'], c['form']))
    return result

def _contributing_forms(train_triples, msd, source_suffix):
    """Find which training pairs a given source suffix's substitutions come from.

    Returns (contributing, applicable): `contributing` maps each target suffix to
    the list of training `(lemma, form)` pairs the substitution predicts correctly,
    and `applicable` is the total number of training pairs (under `msd`) whose
    lemma ends in `source_suffix` (the reliability denominator).
    """
    contributing = defaultdict(list)
    applicable = 0
    for lemma, m, form in train_triples:
        if m != msd or form is None:
            continue
        marked_lemma = BEGIN + lemma + END
        if not marked_lemma.endswith(source_suffix):
            continue
        applicable += 1
        stem = marked_lemma[:len(marked_lemma) - len(source_suffix)]
        marked_form = BEGIN + form + END
        if marked_form.startswith(stem):
            contributing[marked_form[len(stem):]].append((lemma, form))
    return contributing, applicable

def explain_indirect(cross_model, cross_reliabilities, paradigms, sources_by_target,
                     lemma, msd):
    """Show the reasoning of the indirect methods for one lemma and MSD.

    Returns a dict whose `candidates` lists every word form some pivot path
    proposes, sorted by weighted vote. For each candidate form it reports
    `best_reliability` (the score `inflect_indirect` ranks by), `weighted_vote`
    (the score `inflect_indirect_weighted` ranks by), `total_frequency`, and
    `paths`: the contributing pivot word forms, each with the cell it came from
    (`via_msd`), the pivot `reliability` (r1) and step `reliability` (r2), the
    combined `reliability`, its `frequency`, and its `weighted_vote`.
    `indirect_choice` / `weighted_choice` are the actual predictions.
    """
    result = {
        'lemma': lemma, 'msd': msd, 'candidates': [],
        'indirect_choice': inflect_indirect(
            cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd),
        'weighted_choice': inflect_indirect_weighted(
            cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd),
    }
    by_form = defaultdict(list)
    for path in indirect_candidates(
            cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd):
        by_form[path.form].append(path)
    for form, paths in by_form.items():
        contributions = sorted(
            ({'via_msd': 'LEMMA' if p.source_msd == LEMMA_CELL else p.source_msd,
              'pivot_form': p.pivot_form,
              'pivot_reliability': p.pivot_reliability,
              'step_reliability': p.step_reliability,
              'reliability': p.reliability,
              'frequency': p.frequency,
              'weighted_vote': p.frequency * p.reliability}
             for p in paths),
            key=lambda d: (-d['weighted_vote'], -d['reliability'], d['via_msd']))
        result['candidates'].append({
            'form': form,
            'best_reliability': max(p.reliability for p in paths),
            'weighted_vote': sum(p.frequency * p.reliability for p in paths),
            'total_frequency': sum(p.frequency for p in paths),
            'paths': contributions})
    result['candidates'].sort(
        key=lambda c: (-c['weighted_vote'], -c['best_reliability'], c['form']))
    return result

#----------------#
# Evaluation     #
#----------------#

def evaluate(predict, triples):
    """Score a prediction function on gold (lemma, msd, form) triples.

    `predict` is a function `(lemma, msd) -> guessed form`. Returns
    (n_correct, n_total, errors), where `errors` lists the mistakes as dicts.
    """
    correct = 0
    errors = []
    for lemma, msd, gold in triples:
        guess = predict(lemma, msd)
        if guess == gold:
            correct += 1
        else:
            errors.append({'lemma': lemma, 'msd': msd, 'gold': gold, 'guess': guess})
    return correct, len(triples), errors

def run_language(language, data_dir='sigmorphon_data', eval_split='dev'):
    """Train on a language's training set and evaluate all three methods.

    Returns a dict with the accuracy of (1) the frequency-based baseline,
    (2) the reliability-based variant, and (3) reliability-based indirect
    guessing, on the chosen split.
    """
    train_triples = read_data(os.path.join(data_dir, f'{language}-task1-train'))
    eval_triples = read_data(os.path.join(data_dir, f'{language}-task1-{eval_split}'))

    model = train(train_triples)
    reliabilities = compute_reliabilities(train_triples, model)
    cross_model, cross_reliabilities, paradigms, sources_by_target = \
        train_indirect(train_triples)

    methods = {
        'baseline': lambda lemma, msd: inflect(model, lemma, msd),
        'reliable': lambda lemma, msd: inflect_reliable(model, reliabilities, lemma, msd),
        'indirect': lambda lemma, msd: inflect_indirect(
            cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd),
        'weighted': lambda lemma, msd: inflect_indirect_weighted(
            cross_model, cross_reliabilities, paradigms, sources_by_target, lemma, msd),
    }
    result = {'language': language, 'split': eval_split, 'total': len(eval_triples)}
    for name, predict in methods.items():
        correct, total, _ = evaluate(predict, eval_triples)
        result[f'{name}_correct'] = correct
        result[f'{name}_accuracy'] = correct / total if total else 0.0
    return result

#----------------#
# Command line   #
#----------------#

if __name__ == '__main__':
    methods = ['baseline', 'reliable', 'indirect', 'weighted']
    print('Task 1 (dev set): four inflection methods\n')
    header = f'{"language":<12}' + ''.join(f'{name:>11}' for name in methods)
    print(header)
    print('-' * len(header))
    for language in LANGUAGES:
        result = run_language(language)
        suffixing = '*' if language in SUFFIXING_LANGUAGES else ' '
        row = f'{language + suffixing:<12}'
        row += ''.join(f'{result[f"{name}_accuracy"] * 100:>10.2f}%' for name in methods)
        print(row)
    print('\n(* = largely suffixing language)')
