# Efficient State Computation of Process Ongoing Cases

![build](https://github.com/AutomatedProcessImprovement/process-running-state/actions/workflows/build.yaml/badge.svg)
![version](https://img.shields.io/github/v/tag/AutomatedProcessImprovement/process-running-state)

Approach to, given a process model in BPMN formal, compute the state of ongoing cases in constant time.
The approach consists of, in design time, given a maximum size _n_, create an index that associates each
_n_-gram -- i.e., execution of _n_ consecutive activities -- with the state(s) they lead to in the process model.
Then, at runtime, the state of an ongoing process case can be computed in constant time by searching for the last _n_
executed activities in the index.
For example, for an ongoing case `A-B-F-T-W-S-G-T-D`, after building the 5-gram index, the state would be computed
by searching in the index with the sequence `[W, S, G, T, D]`.

This approach has been submitted as a publication to ICPM 2024 under the title "Efficient Online Computation of Business
Process State From Trace Prefixes via N-Gram Indexing", by David Chapela-Campa and Marlon Dumas.

## Requirements

- Python v3.9.5+
- PIP v23.0+
- Python dependencies: all packages listed in [
  _pyproject.toml_](https://github.com/AutomatedProcessImprovement/process-running-state/blob/main/pyproject.toml)

## Basic Usage

Given a process model in BPMN format, the code to build an _n_-gram index and compute the state given an _n_-gram prefix
is:

```Python
from pathlib import Path

from process_running_state.n_gram_index import NGramIndex
from process_running_state.utils import read_bpmn_model

# Read BPMN model
bpmn_model_path = Path("./inputs/synthetic/synthetic_and_k5.bpmn")
bpmn_model = read_bpmn_model(bpmn_model_path)
# Compute reachability graph
reachability_graph = bpmn_model.get_reachability_graph()
# Build n-gram index
n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=5)
n_gram_index.build()
# Compute the state of an ongoing case
n_gram = ["B", "E", "F", "C", "G"]
ongoing_state = n_gram_index.get_best_marking_state_for(n_gram)
# Compute the state of an ongoing case with less than N recorded events
n_gram = [NGramIndex.TRACE_START, "A", "B", "F"]
ongoing_state = n_gram_index.get_best_marking_state_for(n_gram)
```
