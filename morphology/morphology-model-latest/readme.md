# Readme

This directory contains code and papers for an exemplar-based model of morphological inflection.
- `morphology_model.py` is an implementation of an exemplar-based inference method that infers novel inflected word forms based on attested word forms. The method is described in `papers/imm_abstract`.
- `corpora` contains Hungarian inflected word data.
- `papers` contains the relevant papers. `imm_abstract` describes the idea behind `morphology_model.py`, and `Liu and Mao (2016)…` describes a similar, exemplar-based inference method in their section 2 (“a suffix-based baseline”).
- `suffix_baseline.py` is an implementation of several exemplar-based inference methods, vibe coded with Claude Code