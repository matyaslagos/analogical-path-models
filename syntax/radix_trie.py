#!/usr/bin/env python3
from collections import deque, defaultdict
from typing import Optional, Tuple, List, Dict, Iterator, Literal, Union
from pprint import pp
from random import shuffle
from heapq import nlargest
from operator import itemgetter

Direction = Literal["fw", "bw"]

# ------------------------- #
# Radix trie implementation #
# ------------------------- # 

class RadixNode:
    def __init__(
            self,
            sequence_label: Tuple[str, ...] = (),
            sequence_freq: int = 0,
            sequence_children: Optional[Dict[str, "RadixNode"]] = None
        ):
        """Node of a radix trie, containing compressed edge in label.
        """
        self.label = sequence_label
        self.freq = sequence_freq
        self.children = {} if sequence_children is None else sequence_children
        self.prob_of_neighbor = None
        self.prob_of_mix = None

    def _insert(self, sequence: Tuple[str, ...], freq: int = 1) -> None:
        """Insert sequence and increment freq at node.
        """
        if not sequence:
            return
        label = self.label
        # Find index of last agreeing token and update node accordingly
        split_point = -1
        for seq_token, label_token in zip(sequence, label):
            split_point += 1
            # Case 1: sequence and label disagree at some token. Split node at disagreeing
            # token and rearrange children
            if seq_token != label_token:
                # Get remaining suffixes
                rem_sequence, rem_label = sequence[split_point:], label[split_point:]
                # Relabel node to common prefix
                self.label = label[:split_point]
                # Make child node for remaining label, copy original label's children
                rem_label_child_node = RadixNode(rem_label, self.freq, self.children.copy())
                # Make child node for remaining sequence
                rem_sequence_child_node = RadixNode(rem_sequence, freq)
                # Assign new children and increment freq
                self.children = {
                        rem_sequence[0]: rem_sequence_child_node,
                        rem_label[0]: rem_label_child_node
                    }
                self.freq += freq
                return
        # Sequence and label don't disagree. Sequence is endmarked, so they are either
        # identical or label is proper prefix of sequence. Increment label's freq, then
        # check which case holds and handle it
        self.freq += freq
        rem_sequence, rem_label = sequence[split_point + 1:], label[split_point + 1:]
        # Case 2: sequence is prefix of label => they are identical. Do nothing.
        if not rem_sequence:
            return
        # Case 3: label is proper prefix of sequence. Check if node has child headed by
        # head of remaining sequence, and insert or create new node accordingly
        self._insert_or_make_child(rem_sequence, freq)

    def _insert_or_make_child(self, sequence, freq):
        sequence_head = sequence[0] 
        if sequence_head in self.children:
            self.children[sequence_head]._insert(sequence, freq)
        else:
            self.children[sequence_head] = RadixNode(sequence, freq)

    def _find(self, sequence) -> Optional[Tuple["RadixNode", int]]:
        label = self.label
        # Match sequence with label as long as possible
        position = -1
        for seq_token, label_token in zip(sequence, label):
            if seq_token != label_token:
                return None
            position += 1
        rem_sequence, rem_label = sequence[position + 1:], label[position + 1:]
        # Sequence ends inside label => return found node and position
        if not rem_sequence:
            return (self, position)
        # Sequence doesn't end inside label => recurse to child if possible
        try:
            return self.children[rem_sequence[0]]._find(rem_sequence)
        except KeyError:
            return None

    def _continuations(self, position, direction="fw", max_length=float("Inf"), path=None):
        if path is None:
            path = [] if direction == "fw" else deque()
        label = self.label[position:]
        for token in label:
            if direction == "fw":
                path.append(token)
            else:
                path.appendleft(token)
            yield tuple(path.copy()), self.freq
            if len(path) == max_length:
                return
        for child in self.children.values():
            yield from child._continuations(0, direction, max_length, path.copy())

    def _shared_continuations(
            self, self_position,
            other, other_position,
            direction="fw", max_length=float("Inf"), path=None
        ):
        if path is None:
            path = [] if direction == "fw" else deque()
        self_label, other_label = self.label[self_position:], other.label[other_position:]
        for self_token, other_token in zip(self_label, other_label):
            if self_token != other_token:
                return
            self_position, other_position = self_position + 1, other_position + 1
            if direction == "fw":
                path.append(self_token)
            else:
                path.appendleft(self_token)
            yield tuple(path.copy()), self.freq, other.freq
            if len(path) == max_length:
                return
        # Self's label hasn't been fully traversed
        if self_position < len(self.label):
            self_next = self.label[self_position]
            if self_next in other.children:
                new_node_pairs = [(self, other.children[self_next])]
                new_self_position, new_other_position = self_position, 0
        # Other's label hasn't been fully traversed
        elif other_position < len(other.label):
            other_next = other.label[other_position]
            if other_next in self.children:
                new_node_pairs = [(self.children[other_next], other)]
                new_self_position, new_other_position = 0, other_position
        # Both labels have been fully traversed
        else:
            shared_children = self.children.keys() & other.children.keys()
            if shared_children:
                new_node_pairs = [(self.children[child], other.children[child])
                                  for child in shared_children]
                new_self_position, new_other_position = 0, 0
        # For each shared continuing node, recursively yield continuations
        for new_self_node, new_other_node in new_node_pairs:
            yield from new_self_node._shared_continuations(
                    new_self_position, new_other_node, new_other_position,
                    direction, max_length, path.copy()
                )


class RadixTrie:
    def __init__(self):
        self.fw_root = RadixNode()
        self.bw_root = RadixNode()

    def setup(self, corpus: List[Tuple[str, ...]]) -> None:
        for sequence in corpus:
            self._insert(sequence)

    def _insert(self, sequence: Tuple[str, ...], freq: int = 1) -> None:
        reversed_sequence = tuple(reversed(sequence))
        for i in range(len(sequence) + 1):
            self.fw_root._insert(sequence[i:], freq)
            self.bw_root._insert(reversed_sequence[i:], freq)

    def _find(self, sequence, direction: Direction = "fw"):
        if direction == "fw":
            root, goal_sequence = self.fw_root, sequence
        else:
            root, goal_sequence = self.bw_root, tuple(reversed(sequence))
        return root._find(goal_sequence)

    def freq(self, sequence, direction: Direction = "fw"):
        try:
            return self._find(sequence, direction)[0].freq
        except:
            return 0

    def _neighbors(self, sequence, direction: Direction, max_length=float("Inf")):
        node, position = self._find(sequence, direction)
        return node._continuations(position + 1, direction, max_length)

    def right_neighbors(self, sequence, max_length=float("Inf")):
        return self._neighbors(sequence, "fw", max_length)

    def left_neighbors(self, sequence, max_length=float("Inf")):
        return self._neighbors(sequence, "bw", max_length)

    def _shared_neighbors(self, sequence1, sequence2,
                          direction: Direction, max_length=float("Inf")):
        node1, position1 = self._find(sequence1, direction)
        node2, position2 = self._find(sequence2, direction)
        return node1._shared_continuations(
                position1 + 1, node2, position2 + 1, direction, max_length
            )

    def shared_right_neighbors(self, sequence1, sequence2, max_length=float("Inf")):
        return self._shared_neighbors(sequence1, sequence2, "fw", max_length)

    def shared_left_neighbors(self, sequence1, sequence2, max_length=float("Inf")):
        return self._shared_neighbors(sequence1, sequence2, "bw", max_length)


# ------------------------------------------------------------- #
# Computing conditional probabilities for mixtures of sequences #
# ------------------------------------------------------------- #

# Import corpus
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

def analogies_of_mix(trie, seqs_with_scores, max_length=float("inf")):
    analogies = defaultdict(float)
    for seq, seq_score in seqs_with_scores:
        single_anls = analogies_of_seq(trie, seq, max_length=len(seq))
        for anl, anl_score in single_anls:
            analogies[anl] += anl_score
    return sorted(analogies.items(), key=lambda x: x[1], reverse=True)[:100]

def analogies_of_seq(trie, seq, max_length=float("inf")):
    right_anls = right_analogies_of_seq(trie, seq, max_length)
    left_anls = left_analogies_of_seq(trie, seq, max_length)
    anls = {}
    for anl, right_anl_score in right_anls.items():
        if anl in left_anls:
            left_anl_score = left_anls[anl]
            anls[anl] = min(right_anl_score, left_anl_score)
    return anls.items()

def right_analogies_of_seq(trie, seq, max_length=float("inf")):
    anls = defaultdict(float)
    for right_neighbor, right_neighbor_target_freq in trie.right_neighbors(seq, max_length=1):
        right_neighbor_freq = trie.freq(right_neighbor)
        for source, right_neighbor_source_freq in trie.left_neighbors(right_neighbor, max_length=len(seq)):
            source_freq = trie.freq(source)
            source_to_neighbor_prob = right_neighbor_source_freq / source_freq
            neighbor_to_target_prob = right_neighbor_target_freq / right_neighbor_freq
            anls[source] += min(source_to_neighbor_prob, neighbor_to_target_prob)
    return dict(anls)

def left_analogies_of_seq(trie, seq, max_length=float("inf")):
    anls = defaultdict(float)
    for left_neighbor, left_neighbor_target_freq in trie.left_neighbors(seq, max_length=1):
        left_neighbor_freq = trie.freq(left_neighbor)
        for source, left_neighbor_source_freq in trie.right_neighbors(left_neighbor, max_length=len(seq)):
            source_freq = trie.freq(source)
            source_to_neighbor_prob = left_neighbor_source_freq / source_freq
            neighbor_to_target_prob = left_neighbor_target_freq / left_neighbor_freq
            anls[source] += min(source_to_neighbor_prob, neighbor_to_target_prob)
    return dict(anls)

def recursive_analogies(trie, sequence, lookup_dict=None):
    # Initialize lookup dicts
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
            # Recursive calls
            rec_prefixes = recursive_analogies(trie, prefix, lookup_dict, mix_anls_cache)
            rec_suffixes = recursive_analogies(trie, suffix, lookup_dict, mix_anls_cache)
            # Find analogies for mixes
            prefix_anls = analogies_of_mix(trie, rec_prefixes)
            suffix_anls = analogies_of_mix(trie, rec_suffixes)
            # Find attested concatenations of analogies
            for prefix_anl, prefix_score in prefix_anls:
                for suffix_anl, suffix_score in suffix_anls:
                    anl_sequence = prefix_anl + suffix_anl
                    if trie.freq(anl_sequence):
                        anl_seq_scores[anl_sequence] += min(prefix_score, suffix_score)
        best_anls = nlargest(50, anl_seq_scores.items(), key=itemgetter(1))
        lookup_dict[sequence] = best_anls
        return best_anls
