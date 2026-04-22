# Graph Dataset Topology Report

## Original Dataset

- Total rows: 1580
- Unique DAG IDs: 73
- Unique topology hashes: 12
- Avg query length: 122.5
- Node count distribution: {4: 280, 3: 700, 2: 480, 1: 40, 5: 60, 7: 20}

### Original Topology Families

- chain_medium: 560
- chain_short: 480
- fanin: 280
- fanout: 80
- chain_long: 60
- diamond: 40
- single_node: 40
- fork_join: 20
- wide_fanout: 20

## Augmented Dataset

- Total rows: 480
- Unique DAG IDs: 120
### Augmented Topology Families

- multi_branch_independent: 72
- double_diamond: 68
- y_shape: 48
- w_shape: 44
- inverted_y: 40
- asymmetric_fork_join: 40
- deep_asymmetric_merge: 40
- diamond: 32
- hourglass: 32
- repeated_tool: 32
- wide_fanout_deep: 24
- long_chain_branched: 8

## Hard Negatives

- Total: 305
- add_edge: 53
- extra_node: 73
- remove_edge: 53
- swap_edges: 53
- swap_tools: 73

## Topology-Held-Out Splits

### Train

- Rows: 1796
- Unique DAGs: 143
- Families: chain_long, chain_medium, chain_short, deep_asymmetric_merge, double_diamond, fanin, fanout, hourglass, long_chain_branched, repeated_tool, single_node, w_shape, wide_fanout_deep, y_shape

### Dev

- Rows: 72
- Unique DAGs: 18
- Families: multi_branch_independent

### Test

- Rows: 192
- Unique DAGs: 32
- Families: asymmetric_fork_join, diamond, fork_join, inverted_y, wide_fanout

### Train-Test Topology Overlap: 0 families


## Topology Diversity

- Before: 12 unique topologies
- After (base + augmented): 132 unique topologies
