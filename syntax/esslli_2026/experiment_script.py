#!/usr/bin/env python3
# Module imports
import pickle
import radix_trie_esslli as rt
import similarity_metrics as sm
from pprint import pp
from random import shuffle, sample
from collections import defaultdict

sim_metrics = ["jacc", "min_conf", "l1", "cosine", "js_div", "skew_div"]
def sim_metric_code(metric):
    match metric:
        case "l1":
            return "s1"
        case "cosine":
            return "s2"
        case "min_conf":
            return "s3"
        case "jacc":
            return "s4"
        case "js_div":
            return "s5"
        case "skew_div":
            return "s6"

np_sample_sizes = [5, 10, 25, 50, 100]

corpus = rt.txt_to_list("corpora/lassy_corpus.txt")
distr = rt.RadixTrie()
distr.setup(corpus)

# Get list of pronouns and list of proper nouns
pronouns = pickle.load(open("corpora/lassy_pronouns.pkl", "rb"))
proper_nouns = pickle.load(open("corpora/lassy_proper_nouns.pkl", "rb"))

all_noun_phrases = pickle.load(open("corpora/lassy_real_noun_phrases.pkl", "rb"))

def experiment_run(always_shuffle=False):

    # Get sample of noun phrases
    noun_phrases = all_noun_phrases.copy()
    if not always_shuffle:
        shuffle(noun_phrases)
        noun_phrase_samples = {n: noun_phrases[:n] for n in np_sample_sizes}
    else:
        noun_phrase_samples = {n: sample(noun_phrases, n) for n in np_sample_sizes}
    found_words_by_sample_size = {}
    # Test for incresing sample sizes
    for n in np_sample_sizes:
        noun_phrases_sample = noun_phrase_samples[n] # list of tuples of strings

        # (a) Get dicts with items of form (w, P(noun_phrases_sample | w)) for neighbor words w of noun_phrases_sample
        # (used later for confusion similarities)
        np_mix_fw = rt.cond_probs_of_mix(distr, noun_phrases_sample, "fw") # right neighbors
        np_mix_bw = rt.cond_probs_of_mix(distr, noun_phrases_sample, "bw") # left neighbors

        # (b) Get dicts with items of form (w, P(w | noun_phrases_sample)) for neighbor words w of noun_phrases_sample
        # (used later for non-confusion similarities)
        np_neighbors_fw = rt.cond_probs_of_neighbors(distr, noun_phrases_sample, "fw") # right neighbors
        np_neighbors_bw = rt.cond_probs_of_neighbors(distr, noun_phrases_sample, "bw") # left neighbors

        # Get set of words that share at least one left and one right context with noun_phrases_sample
        candidates = rt.similar_word_candidates(distr, noun_phrases_sample)


        # For each candidate, get dicts with items of form (w, P(w | candidate)) for neighbor words w of candidate,
        # convert these dicts and the dicts of the noun phrase sample into vectors, and compute their similarities
        similarities = {metric: {} for metric in sim_metrics}
        for candidate in candidates:
            # Compute conditional probabilities of neighbors of candidate
            candidate_neighbors_fw = rt.cond_probs_of_neighbors(distr, [candidate], "fw") # right neighbors
            candidate_neighbors_bw = rt.cond_probs_of_neighbors(distr, [candidate], "bw") # left neighbors

            # Compute vectors for non-confusion similarity metrics
            np_fw_vector, candidate_fw_vector = rt.vectorize(np_neighbors_fw, candidate_neighbors_fw)
            np_bw_vector, candidate_bw_vector = rt.vectorize(np_neighbors_bw, candidate_neighbors_bw)

            # Compute similarities for all metrics
            for metric in sim_metrics:
                if metric in ["jacc", "l1", "cosine", "js_div", "skew_div"]:
                    # Compute similarity from vectors
                    sim_fw = rt.compute_similarity(np_fw_vector, candidate_fw_vector, metric)
                    sim_bw = rt.compute_similarity(np_bw_vector, candidate_bw_vector, metric)
                    sim = min(sim_fw, sim_bw)
                    similarities[metric][candidate] = sim

                elif metric in ["min_conf", "conf"]:
                    # Compute vectors for confusion similarity metrics
                    np_conf_fw_vector, candidate_fw_vector = rt.vectorize(np_mix_fw, candidate_neighbors_fw)
                    np_conf_bw_vector, candidate_bw_vector = rt.vectorize(np_mix_bw, candidate_neighbors_bw)

                    # Compute confusion score from vectors
                    sim_fw = rt.compute_similarity(np_conf_fw_vector, candidate_fw_vector, metric)
                    sim_bw = rt.compute_similarity(np_conf_bw_vector, candidate_bw_vector, metric)
                    sim = min(sim_fw, sim_bw)
                    similarities[metric][candidate] = sim
        found_words_by_metric = {}
        for metric, sim_dict in similarities.items():
            if metric in ["l1", "js_div", "skew_div"]:
                sorted_words = sorted(sim_dict.items(), key=lambda x: x[1])[:20]
            else:
                sorted_words = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)[:20]
            found_pronouns = {}
            wrong_words = {}
            for tup in sorted_words:
                tup_freq = distr.freq(tup[0])
                if "".join(tup[0]) in pronouns:
                    found_pronouns[tup[0]] = (tup[1], tup_freq)
                else:
                    wrong_words[tup[0]] = (tup[1], tup_freq)
            found_words_by_metric[sim_metric_code(metric)] = {"good": found_pronouns, "bad": wrong_words}
        found_words_by_sample_size[n] = found_words_by_metric
    return found_words_by_sample_size

def avgs_with_variance(results):
    avg_results = {metric:
                   {sample_size:
                    sum(results[i][sample_size][metric] for i in results) / len(results)
                    for sample_size in np_sample_sizes}
                   for metric in sim_metrics}
    variances = {metric:
                 {sample_size:
                  sum((results[i][sample_size][metric] - avg_results[metric][sample_size]) ** 2
                      for i in results) / (len(results) - 1)
                  for sample_size in np_sample_sizes}
                 for metric in sim_metrics}
    return avg_results, variances

def main(n, always_shuffle=False):
    found_words_by_run = {}
    run_number = 1
    while run_number <= n:
        try:
            found_words_by_run[run_number] = experiment_run(always_shuffle)
            print(f"Finished run number {run_number}")
            run_number += 1
        except Exception as e:
            print(f"Run failed with exception {e}")
    averaged_results = {}
    pickle.dump(found_words_by_run, open(f"lassy_full_results_02_19.pkl", "wb"))
    """
    pickle.dump(wrong_words_per_run, open("lassy_wrong_words_02_19.pkl", "wb"))
    for result, result_type in [(correct_pct, "correct pct"), (pronoun_pct, "pronoun pct"), (proper_noun_pct, "proper noun pct")]:
        averaged_result, result_variances = avgs_with_variance(result)
        averaged_results[result_type] = averaged_result
        pickle.dump(averaged_result, open(f"lassy_results_{result_type}_02_19.pkl", "wb"))
    """
    return found_words_by_run
