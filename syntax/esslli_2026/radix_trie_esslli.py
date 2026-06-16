#!/usr/bin/env python3
from collections import deque, defaultdict
from typing import Optional, Tuple, List, Dict, Iterator, Literal, Union
from pprint import pp
from random import shuffle
import pickle
import numpy as np
import similarity_metrics as sm

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
            yield path, self.freq
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
            yield path, self.freq, other.freq
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
        return self._find(sequence, direction)[0].freq

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

# Get set of words that share at least one left & right context with mixture
def similar_word_candidates(self, sequences):
    candidates = {"fw": set(), "bw": set()}
    for direction, other_direction in [("fw", "bw"), ("bw", "fw")]:
        for sequence in sequences:
            for context, _ in self._neighbors(sequence, direction, max_length=1):
                for candidate, _ in self._neighbors(tuple(context),
                                                    other_direction,
                                                    max_length=1):
                    candidates[direction].add(tuple(candidate))
    return candidates["fw"] & candidates["bw"]

# Get dict of form {w: P(A | w)} for neighbors w of list A of sequences
def cond_probs_of_mix(self, sequences, direction):
    """Return dict of cond probs of mixture of sequences given its neighbors.
    """
    cond_probs = defaultdict(float)
    for sequence in sequences:
        for neighbor, joint_freq in self._neighbors(sequence, direction, max_length=1):
            neighbor_freq = self.freq(tuple(neighbor))
            cond_prob = joint_freq / neighbor_freq
            cond_probs[tuple(neighbor)] += cond_prob
    return dict(cond_probs)

# Get dict of form {w: P(w | A)} for neighbors w of list A of sequences
def cond_probs_of_neighbors(self, sequences, direction):
    """Return dict of cond probs of neighbors given mixture of sequences.
    """
    mix_freq = sum(self.freq(sequence) for sequence in sequences)
    cond_probs = defaultdict(float)
    for sequence in sequences:
        for neighbor, joint_freq in self._neighbors(sequence, direction, max_length=1):
            cond_prob = joint_freq / mix_freq
            cond_probs[tuple(neighbor)] += cond_prob
    return dict(cond_probs)

def vectorize(dict1, dict2):
    # Compute common and non-common keys
    common_keys = list(dict1.keys() & dict2.keys())
    dict1_keys = list(dict1.keys() - dict2.keys())
    dict2_keys = list(dict2.keys() - dict1.keys())
    # Shared parts
    common_vec1 = [dict1[k] for k in common_keys]
    common_vec2 = [dict2[k] for k in common_keys]
    # In dict1 but not in dict2 (vec2's values should be 0 at these coordinates)
    dict1_vec1 = [dict1[k] for k in dict1_keys]
    dict1_vec2 = [0] * len(dict1_keys)
    # In dict2 but not in dict1 (vec1's values should be 0 at these coordinates)
    dict2_vec1 = [0] * len(dict2_keys)
    dict2_vec2 = [dict2[k] for k in dict2_keys]
    
    # Create numpy vectors
    vec1 = np.array(common_vec1 + dict1_vec1 + dict2_vec1)
    vec2 = np.array(common_vec2 + dict1_vec2 + dict2_vec2)

    return vec1, vec2

def compute_similarity(vec1, vec2, metric):
    if metric == "jacc":
        return sm.jaccard_coefficient(vec1, vec2)
    elif metric == "min_conf":
        return sm.min_confusion_probability(vec1, vec2)
    elif metric == "l1":
        return sm.l1_norm(vec1, vec2)
    elif metric == "cosine":
        return sm.cosine_similarity(vec1, vec2)
    elif metric == "js_div":
        return sm.jensen_shannon_divergence(vec1, vec2)
    elif metric == "skew_div":
        return sm.skew_divergence(vec1, vec2)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

def main():
    pass
