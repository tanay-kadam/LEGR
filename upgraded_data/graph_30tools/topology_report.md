# Graph Dataset Topology Report

## Original Dataset

- Total rows: 1580
- Unique DAG IDs: 77
- Unique topology hashes: 12
- Avg query length: 121.7
- Node count distribution: {3: 800, 2: 420, 4: 280, 7: 20, 1: 40, 5: 20}

### Original Topology Families

- chain_medium: 600
- chain_short: 420
- fanout: 200
- fanin: 200
- chain_long: 40
- diamond: 40
- single_node: 40
- wide_fanout: 40

## Augmented Dataset

- Total rows: 480
- Unique DAG IDs: 120
### Augmented Topology Families

- y_shape: 64
- double_diamond: 56
- w_shape: 48
- hourglass: 48
- diamond: 44
- repeated_tool: 44
- deep_asymmetric_merge: 40
- asymmetric_fork_join: 36
- multi_branch_independent: 32
- wide_fanout_deep: 32
- inverted_y: 28
- long_chain_branched: 8

## Hard Negatives

- Total: 322
- add_edge: 56
- extra_node: 77
- remove_edge: 56
- swap_edges: 56
- swap_tools: 77

## Topology-Held-Out Splits

### Train

- Rows: 1828
- Unique DAGs: 155
- Families: asymmetric_fork_join, chain_long, chain_medium, chain_short, deep_asymmetric_merge, double_diamond, fanin, fanout, long_chain_branched, repeated_tool, single_node, w_shape, wide_fanout_deep, y_shape

### Dev

- Rows: 32
- Unique DAGs: 8
- Families: multi_branch_independent

### Test

- Rows: 200
- Unique DAGs: 34
- Families: diamond, hourglass, inverted_y, wide_fanout

### Train-Test Topology Overlap: 0 families


## Topology Diversity

- Before: 12 unique topologies
- After (base + augmented): 132 unique topologies
