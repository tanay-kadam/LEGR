# Campaign V4 V3 action-versus-semantic geometry

**Evidence classification:** NO_ACTION_ADVANTAGE

The same 322 frozen V3 DAG embeddings and the same t-SNE coordinates are used in both panels. All quantitative diagnostics are computed in the original 256-dimensional cosine space.

```json
{
  "action": {
    "counts": {
      "Data Retrieval": 131,
      "Mixed": 69,
      "Orchestration": 6,
      "State Modification": 116
    },
    "silhouette_cosine": 0.014306539669632912,
    "silhouette_95ci": [
      0.0013087932631606237,
      0.0277075020596385
    ],
    "knn_k": 5,
    "knn_macro_f1": 0.8212152951322533,
    "knn_macro_f1_95ci": [
      0.6556890494448527,
      0.9103142579625955
    ],
    "knn_neighborhood_purity": 0.7857142857142857,
    "knn_neighborhood_purity_95ci": [
      0.7627329192546584,
      0.8093322981366461
    ]
  },
  "semantic": {
    "counts": {
      "Account & Subscription": 58,
      "Data & Access Management": 51,
      "Mixed": 126,
      "Service & Incident Operations": 87
    },
    "silhouette_cosine": 0.02982461266219616,
    "silhouette_95ci": [
      0.02195068448781967,
      0.038439434859901665
    ],
    "knn_k": 5,
    "knn_macro_f1": 0.88457216639448,
    "knn_macro_f1_95ci": [
      0.84356364670125,
      0.9199088521130305
    ],
    "knn_neighborhood_purity": 0.7167701863354037,
    "knn_neighborhood_purity_95ci": [
      0.6931521739130434,
      0.7434782608695651
    ]
  },
  "differences": {
    "silhouette_action_minus_semantic": -0.015518072992563248,
    "knn_macro_f1_action_minus_semantic": -0.06335687126222667,
    "purity_action_minus_semantic": 0.06894409937888202
  },
  "permutation_test": {
    "permutations": 1000,
    "alternative": "action_minus_semantic > null difference from independently permuted labels",
    "silhouette_difference_p_one_sided": 0.35364635364635366,
    "purity_difference_p_one_sided": 0.30569430569430567
  }
}
```
