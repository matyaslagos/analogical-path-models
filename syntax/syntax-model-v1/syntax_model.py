#!/usr/bin/env python3
from collections import defaultdict
from itertools import product
from operator import itemgetter
from heapq import nlargest

#-----------------#
# Setup functions #
#-----------------#

# Import some text as corpus
def txt_to_list(filename):
    """Import a txt list of sentences as a list of tuples of words.

    Argument:
        filename (string): e.g. 'corpus.txt', the name of a txt file with one sentence
        per line

    Returns:
        list of tuples of strings: each sentence is an endmarked tuple of strings,
        e.g. ('<', 'this', 'is', 'good', '>')
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [('<',) + tuple(line.strip().split()) + ('>',) for line in lines]

#-----------------------------------------#
# FreqNode and FreqTrie class definitions #
#-----------------------------------------#

class FreqNode:
    def __init__(self):
        self.children = {}
        self.freq = 0

    def _increment_or_make_branch(self, sequence, freq=1):
        """Increment the frequency of sequence or make a new branch for it.
        """
        current_node = self
        for token in sequence:
            current_node = current_node._get_or_make_child(token)
            current_node.freq += freq

    def _get_or_make_child(self, token):
        """Return the child called token or make a new child called token.
        """
        if token not in self.children:
            self.children[token] = FreqNode()
        return self.children[token]

class FreqTrie:
    def __init__(self):
        self.fw_root = FreqNode()
        self.bw_root = FreqNode()
    
    # Insert a sequence
    def _insert(self, sequence, freq=1):
        """Record distributions of prefixes and suffixes of sequence.

        Arguments:
            sequence (iterable of strings): e.g. ('<', 'this', 'is', 'good', '>')
            freq (int): how many occurrences of sequence should be recorded

        Effect:
            For each prefix--suffix split of sequence, record the occurrences of
            prefix and suffix. (Prefix is reversed to make shared-neighbor search more
            efficient.)
        """
        # Add token frequency mass of sequence to root nodes (to record corpus size)
        token_freq_mass = len(sequence) * freq
        self.fw_root.freq += token_freq_mass
        self.bw_root.freq += token_freq_mass
        # Record each suffix in fw_trie and each reversed prefix in bw_trie
        prefix_suffix_pairs = (
            (sequence[:i], sequence[i:])
            for i in range(len(sequence) + 1)
        )
        for prefix, suffix in prefix_suffix_pairs:
            self.fw_root._increment_or_make_branch(suffix, freq)
            self.bw_root._increment_or_make_branch(reversed(prefix), freq)
    
    # Set up model with training corpus
    def setup(self, training_corpus_path='corpora/norvig_corpus.txt'):
        corpus = txt_to_list(training_corpus_path)
        for sequence in corpus:
            self._insert(sequence)
    
    # Get node of sequence
    def sequence_node(self, sequence, direction='fw'):
        """Return the node that represents sequence.

        Arguments:
            sequence (tuple of strings): of the form ('this', 'is')
            direction (string): 'fw' or 'bw', indicating whether we are looking
                for forward neighbors or backward neighbors

        Returns:
            FreqNode representing sequence.
        """
        # If looking for right neighbors, start in root of forward trie
        if direction == 'fw':
            current_node = self.fw_root
        # If looking for left neighbors, start in root of backward trie and reverse sequence
        else:
            current_node = self.bw_root
            sequence = reversed(sequence)
        # Go to node of sequence
        for token in sequence:
            try:
                current_node = current_node.children[token]
            except KeyError:
                return None
        return current_node

    # Get frequency of sequence
    def freq(self, sequence=''):
        """Return the frequency of sequence.
        """
        seq_node = self.sequence_node(sequence)
        return seq_node.freq if seq_node else 0
    
    # Get right neighbors (with frequencies) of sequence
    def right_neighbors(self, sequence, max_length=float('inf'),
                        min_length=0, only_completions=False):
        """Return generator of (right_neighbor, joint_freq) pairs for sequence.
        """
        return self._neighbors(sequence, 'fw',
                               max_length, min_length, only_completions)
    
    # Get left neighbors (with frequencies) of sequence
    def left_neighbors(self, sequence, max_length=float('inf'),
                       min_length=0, only_completions=False):
        """Return generator of (left_neighbor, joint_freq) pairs for sequence.
        """
        return self._neighbors(sequence, 'bw',
                               max_length, min_length, only_completions)
    
    # Get neighbors of sequence (with frequencies) in direction
    def _neighbors(self, sequence, direction='fw',
                   max_length=float('inf'), min_length=0, only_completions=False):
        """Return generator of (neighbor, joint_freq) pairs for sequence in direction.
        """
        seq_node = self.sequence_node(sequence, direction)
        if not seq_node:
            return iter(())
        return self._neighbors_aux(seq_node, direction, max_length, min_length,
                                   only_completions, path=[])
    
    # Auxiliary recursive function for _neighbors()
    def _neighbors_aux(self, seq_node, direction,
                       max_length, min_length, only_completions, path):
        """Yield each neighbor of sequence with their joint frequency.
        """
        if len(path) >= max_length:
            return
        for child in seq_node.children:
            new_path = path + [child] if direction == 'fw' else [child] + path
            child_node = seq_node.children[child]
            freq = child_node.freq
            if len(new_path) >= min_length:
                if (not only_completions) or (child in {'<','>'}):
                    yield (tuple(new_path), freq)
            yield from self._neighbors_aux(child_node, direction,
                                           max_length, min_length,
                                           only_completions, new_path)
    
    # Get shared right neighbors (with frequencies) of sequence_1 and sequence_2
    def shared_right_neighbors(self, sequence_1, sequence_2, max_length=float('inf'),
                               min_length=0, only_completions=False):
        """Return generator of (shared_neighbor, joint_freq_1, joint_freq_2) pairs.
        """
        return self._shared_neighbors(sequence_1, sequence_2, 'fw',
                                      max_length, min_length, only_completions)
    
    # Get shared left neighbors (with frequencies) of sequence_1 and sequence_2
    def shared_left_neighbors(self, sequence_1, sequence_2, max_length=float('inf'),
                              min_length=0, only_completions=False):
        """Return generator of (shared_neighbor, joint_freq_1, joint_freq_2) pairs.
        """
        return self._shared_neighbors(sequence_1, sequence_2, 'bw',
                                      max_length, min_length, only_completions)
    
    # Get shared neighbors (with frequencies) of sequence_1 and sequence_2 in direction
    def _shared_neighbors(self, sequence_1, sequence_2, direction='fw',
                          max_length=float('inf'), min_length=0, only_completions=False):
        """Return generator of shared neighbors of sequence_1 and sequence_2.

        Arguments:
            sequence_1 (tuple of strings): e.g. ('is', 'good')
            sequence_2 (tuple of strings): e.g. ('was', 'here')

        Returns:
            generator of (neighbor, freq_1, freq_2) tuples:
                if direction is 'bw' and the tuple (('this',), 23, 10) is yielded:
                - 'this' occurred before 'is good' 23 times, and
                - 'this' occurred before 'was here' 10 times.
        """
        seq_node_1 = self.sequence_node(sequence_1, direction)
        seq_node_2 = self.sequence_node(sequence_2, direction)
        if not seq_node_1 or not seq_node_2:
            return iter(())
        return self._shared_neighbors_aux(seq_node_1, seq_node_2, direction,
                                          max_length, min_length, only_completions, path=[])
    
    # Auxiliary function for _shared_neighbors()
    def _shared_neighbors_aux(self, seq_node_1, seq_node_2, direction,
                              max_length, min_length, only_completions, path):
        """Yield shared neighbors of seq_node_1 and seq_node_2.
        """
        if len(path) >= max_length:
            return
        for child in seq_node_1.children:
            if child in seq_node_2.children:
                new_path = path + [child] if direction == 'fw' else [child] + path
                child_node_1 = seq_node_1.children[child]
                child_node_2 = seq_node_2.children[child]
                freq_1 = child_node_1.freq
                freq_2 = child_node_2.freq
                if len(new_path) >= min_length:
                    if (not only_completions) or (child in {'<','>'}):
                        yield (tuple(new_path), freq_1, freq_2)
                yield from self._shared_neighbors_aux(child_node_1, child_node_2,
                                                      direction, max_length, min_length,
                                                      only_completions, new_path)
    # Get right analogies (with scores) of sequence
    def right_analogies(self, sequence, max_length=float('inf')):
        """Return dict of analogies of sequence based on right contexts.
        """
        return self._analogies(sequence, 'fw', max_length)

    # Get left analogies (with scores) of sequence
    def left_analogies(self, sequence, max_length=float('inf')):
        """Return dict of analogies of sequence based on left contexts.
        """
        return self._analogies(sequence, 'bw', max_length)
    
    # Get analogies (with scores) of sequence in direction
    def _analogies(self, sequence, direction, max_length=float('inf')):
        """Return analogies of target sequence in the given direction.
        """
        target_freq = self.freq(sequence)
        source_scores = defaultdict(float)
        # For each context, find sources that can be substituted by target, i.e.
        # such that we can go (source -> context –> target) with high probability
        contexts = self._neighbors(sequence, direction)
        for context, context_target_freq in contexts:
            context_freq = self.freq(context)
            # Probability of going from context to target
            context_target_prob = context_target_freq / context_freq
            other_direction = 'bw' if direction == 'fw' else 'fw'
            sources = self._neighbors(context, other_direction, max_length)
            for source, context_source_freq in sources:
                source_freq = self.freq(source)
                # Probability of going from source to context
                context_source_prob = context_source_freq / source_freq
                source_score = combine_path_scores(context_target_prob, context_source_prob)
                source_scores[source] += source_score
        return source_scores

#-----------------------------#
# Analogical parser functions #
#-----------------------------#

def combine_path_scores(source_to_context_prob, context_to_target_prob):
    """Combine the conditional probabilities of an analogical path.
    """
    return min(source_to_context_prob, context_to_target_prob)

def combine_side_scores(left_subst_score, right_subst_score):
    """Combine the left- and right-substitutabilities of a phrase by another phrase.
    """
    return min(left_subst_score, right_subst_score)

def combine_split_scores(prefix_subst_score, suffix_subst_score):
    """Combine the substitutabilities of phrases p1, p2 in a split (p1, p2).
    
    Suppose we are analysing the split ('my cat', 'was chasing a mouse'). Then let
        - prefix_subst_score be the degree to which 'the dog' is substitutable
          by 'my cat', and
        - suffix_subst_score be the degree to which 'is sleeping' is substitutable
          by 'was chasing a mouse'.
    Then this function returns the degree to which ('the dog', 'is sleeping') is
    substitutable by ('my cat', 'was chasing a mouse').
    """
    return min(prefix_subst_score, suffix_subst_score)

def print_freqs(model, analogy_list):
    best_analogies = [(' '.join(x[0]), model.freq(x[0])) for x in analogy_list[:10]]
    max_length = max(max(len(x[0]) for x in best_analogies), len('analogy'))
    print('')
    print('analogy' + (max_length - len('analogy')) * ' ' + '  ', 'frequency')
    print('-' * max_length + '---' + '-' * len('frequency'))
    for analogy, freq in best_analogies:
        print(analogy + (max_length - len(analogy)) * ' ' + '  ', freq)
    print('')

def bilateral_analogies(model: FreqTrie, sequence: tuple[str, ...]):
    left_anl_scores = model.left_analogies(sequence, max_length=len(sequence))
    right_anl_scores = model.right_analogies(sequence, max_length=len(sequence))
    bilateral_anls = left_anl_scores.keys() & right_anl_scores.keys()
    bilateral_anl_scores = {}
    for anl in bilateral_anls:
        bilateral_anl_scores[anl] = min(left_anl_scores[anl], right_anl_scores[anl])
    return nlargest(50, bilateral_anl_scores.items(), key=itemgetter(1))

def bigram_analogies(model: FreqTrie, bigram: tuple[str, str]):
    s1, s2 = ((word,) for word in bigram)
    s1_anls = bilateral_analogies(model, s1)
    s2_anls = bilateral_analogies(model, s2)
    anls = {}
    for s1_anl, s1_score in s1_anls:
        for s2_anl, s2_score in s2_anls:
            n = model.freq(s1_anl + s2_anl)
            if n:
                anls[s1_anl + s2_anl] = (s1_score * s2_score) * n
    return nlargest(50, anls.items(), key=itemgetter(1))

def bigram_to_unigrams(model: FreqTrie, bigram: tuple[str, str]):
    bigrams_to_mix = bigram_analogies(model, bigram)
    return mix_and_reduce(model, bigrams_to_mix, 1)

def mix_and_reduce(model, weighted_sequences, analogy_length):
    # Sum the frequencies of sequences in left contexts and in right contexts
    # to get the frequency of the mixed distribution in each context
    left_contexts = defaultdict(float)
    right_contexts = defaultdict(float)
    for sequence, _ in weighted_sequences:
        for left_context, freq in model.left_neighbors(sequence):
            left_contexts[left_context] += freq # weight is unnecessary
        for right_context, freq in model.right_neighbors(sequence):
            right_contexts[right_context] += freq
    # Find left analogies for mixed distribution
    left_anl_contexts = defaultdict(lambda: defaultdict(float))
    left_anls = defaultdict(float)
    for left_context, context_target_freq in left_contexts.items():
        context_freq = model.freq(left_context)
        sources = model.right_neighbors(left_context, max_length=analogy_length)
        for source, context_source_freq in sources:
            source_freq = model.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = combine_path_scores(source_to_context_prob, context_to_target_prob)
            left_anls[source] += score
            left_anl_contexts[source][left_context] += score
    # Find right analogies for mixed distribution
    right_anl_contexts = defaultdict(lambda: defaultdict(float))
    right_anls = defaultdict(float)
    for right_context, context_target_freq in right_contexts.items():
        context_freq = model.freq(right_context)
        sources = model.left_neighbors(right_context, max_length=analogy_length)
        for source, context_source_freq in sources:
            source_freq = model.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = combine_path_scores(source_to_context_prob, context_to_target_prob)
            right_anls[source] += score
            right_anl_contexts[source][right_context] += score
    # Compute bilateral substitutability of analogies by the mixed distribution,
    # and record which contexts contributed the most
    anls = {}
    context_dict = {}
    for anl in left_anls.keys() & right_anls.keys():
        anls[anl] = combine_side_scores(left_anls[anl], right_anls[anl])
        best_left_contexts = nlargest(10, left_anl_contexts[anl].items(), key=itemgetter(1))
        best_right_contexts = nlargest(10, right_anl_contexts[anl].items(), key=itemgetter(1))
        context_dict[anl] = {'left': best_left_contexts, 'right': best_right_contexts}
    return nlargest(10, anls.items(), key=itemgetter(1))

def recursive_analogies(model, sequence, lookup_dict=None):
    # Initialise dynamic lookup dict
    if lookup_dict is None:
        lookup_dict = {}
    # Dynamic lookup
    if sequence in lookup_dict:
        return lookup_dict[sequence]
    # Base case
    elif len(sequence) == 1:
        anls = [(sequence, 1)]
        lookup_dict[sequence] = anls
        return anls
    # Recursive case
    else:
        binary_splits = ((sequence[:i], sequence[i:]) for i in range(1, len(sequence)))
        anl_seq_scores = defaultdict(float)
        for prefix, suffix in binary_splits:
            # Recursive calls on constituents
            rec_prefixes = recursive_analogies(model, prefix, lookup_dict)
            rec_suffixes = recursive_analogies(model, suffix, lookup_dict)
            # Find analogical sequences for prefix and suffix analogies
            anl_sequences = split_analogies(model, prefix, suffix, rec_prefixes, rec_suffixes)
            for anl_sequence, score in anl_sequences:
                anl_seq_scores[anl_sequence] += score
        best_anls = nlargest(10, anl_seq_scores.items(), key=itemgetter(1))
        lookup_dict[sequence] = best_anls
        return best_anls

def split_analogies(model, prefix, suffix, rec_prefixes, rec_suffixes):
    prefix_anls = defaultdict(float)
    for rec_prefix, weight in rec_prefixes[:10]:
        anls = bilateral_analogies(model, rec_prefix)
        for anl, score in anls:
            prefix_anls[anl] += score * weight
    sorted_prefix_anls = nlargest(10, prefix_anls.items(), key=itemgetter(1))
    suffix_anls = defaultdict(float)
    for rec_suffix, weight in rec_suffixes[:10]:
        anls = bilateral_analogies(model, rec_suffix)
        for anl, score in anls:
            suffix_anls[anl] += score * weight
    sorted_suffix_anls = nlargest(10, suffix_anls.items(), key=itemgetter(1))
    anls = defaultdict(float)
    for prefix_info, suffix_info in product(sorted_prefix_anls, sorted_suffix_anls):
        prefix_anl, prefix_score = prefix_info
        suffix_anl, suffix_score = suffix_info
        if model.freq(prefix_anl + suffix_anl):
            score = combine_split_scores(prefix_score, suffix_score)
            anls[prefix_anl + suffix_anl] += score
    return nlargest(10, anls.items(), key=itemgetter(1))
