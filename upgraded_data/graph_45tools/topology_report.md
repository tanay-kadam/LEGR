# Graph Dataset Topology Report

## Original Dataset

- Total rows: 1560
- Unique DAG IDs: 78
- Unique topology hashes: 12
- Avg query length: 126.6
- Node count distribution: {3: 600, 4: 420, 1: 40, 2: 480, 7: 20}

### Original Topology Families

- chain_medium: 640
- chain_short: 480
- fanin: 140
- fanout: 120
- diamond: 60
- single_node: 40
- wide_fanout: 40
- wide_fanin: 20
- chain_long: 20

## Augmented Dataset

- Total rows: 480
- Unique DAG IDs: 120
### Augmented Topology Families

- multi_branch_independent: 52
- inverted_y: 52
- wide_fanout_deep: 44
- repeated_tool: 44
- diamond: 44
- y_shape: 40
- asymmetric_fork_join: 40
- hourglass: 36
- w_shape: 36
- deep_asymmetric_merge: 32
- long_chain_branched: 32
- double_diamond: 28

## Hard Negatives

- Total: 312
- add_edge: 52
- extra_node: 78
- remove_edge: 52
- swap_edges: 52
- swap_tools: 78

## Topology-Held-Out Splits

### Train

- Rows: 1740
- Unique DAGs: 147
- Families: chain_long, chain_medium, chain_short, deep_asymmetric_merge, double_diamond, fanin, fanout, hourglass, inverted_y, long_chain_branched, repeated_tool, single_node, w_shape, y_shape

### Dev

- Rows: 52
- Unique DAGs: 13
- Families: multi_branch_independent

### Test

- Rows: 248
- Unique DAGs: 38
- Families: asymmetric_fork_join, diamond, wide_fanin, wide_fanout, wide_fanout_deep

### Train-Test Topology Overlap: 0 families


## Topology Diversity

- Before: 12 unique topologies
- After (base + augmented): 132 unique topologies
