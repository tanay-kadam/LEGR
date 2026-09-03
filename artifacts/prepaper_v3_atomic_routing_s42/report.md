# Frozen V3 cross-dataset atomic routing

The Campaign V4 V3 checkpoints are frozen. Each query is ranked directly against 15 one-node, zero-edge tool graphs; neither routing taxonomy is used by V3. This is a cross-dataset transfer result, not an unseen-topology claim and not causal evidence for Functional Categorization.

```json
{
  "v3_no_ged_15t": {
    "architecture": "V3",
    "objective": "InfoNCE",
    "conditions": {
      "Standard": {
        "n": 1005,
        "correct": 527,
        "accuracy": 0.5243781094527363,
        "accuracy_pct": 52.43781094527363,
        "recall@1": 0.5243781094527363,
        "recall@3": 0.7333333333333333,
        "recall@5": 0.8696517412935323,
        "mrr@5": 0.6462686567164179,
        "mean_rank": 2.691542288557214,
        "macro_f1": 0.495250564828018
      },
      "Lexical": {
        "n": 1005,
        "correct": 288,
        "accuracy": 0.2865671641791045,
        "accuracy_pct": 28.65671641791045,
        "recall@1": 0.2865671641791045,
        "recall@3": 0.5014925373134328,
        "recall@5": 0.6467661691542289,
        "mrr@5": 0.41290215588723056,
        "mean_rank": 4.566169154228856,
        "macro_f1": 0.2606398799363923
      },
      "Confusable": {
        "n": 450,
        "correct": 181,
        "accuracy": 0.4022222222222222,
        "accuracy_pct": 40.22222222222222,
        "recall@1": 0.4022222222222222,
        "recall@3": 0.6644444444444444,
        "recall@5": 0.8266666666666667,
        "mrr@5": 0.5478148148148149,
        "mean_rank": 3.097777777777778,
        "macro_f1": 0.3375396852368372
      },
      "Paraphrase": {
        "n": 1255,
        "correct": 643,
        "accuracy": 0.5123505976095617,
        "accuracy_pct": 51.23505976095617,
        "recall@1": 0.5123505976095617,
        "recall@3": 0.6948207171314741,
        "recall@5": 0.8350597609561753,
        "mrr@5": 0.6237715803452855,
        "mean_rank": 2.954581673306773,
        "macro_f1": 0.49324048021399614
      }
    },
    "micro_aggregate": {
      "n": 3715,
      "correct": 1639,
      "accuracy": 0.4411843876177658,
      "accuracy_pct": 44.11843876177658
    }
  },
  "v3_ged_15t": {
    "architecture": "V3",
    "objective": "InfoNCE+GED",
    "conditions": {
      "Standard": {
        "n": 1005,
        "correct": 534,
        "accuracy": 0.5313432835820896,
        "accuracy_pct": 53.13432835820896,
        "recall@1": 0.5313432835820896,
        "recall@3": 0.7522388059701492,
        "recall@5": 0.8656716417910447,
        "mrr@5": 0.6523548922056385,
        "mean_rank": 2.691542288557214,
        "macro_f1": 0.5020090061880581
      },
      "Lexical": {
        "n": 1005,
        "correct": 275,
        "accuracy": 0.2736318407960199,
        "accuracy_pct": 27.363184079601986,
        "recall@1": 0.2736318407960199,
        "recall@3": 0.5184079601990049,
        "recall@5": 0.6358208955223881,
        "mrr@5": 0.4039635157545605,
        "mean_rank": 4.655721393034826,
        "macro_f1": 0.24872395122380214
      },
      "Confusable": {
        "n": 450,
        "correct": 180,
        "accuracy": 0.4,
        "accuracy_pct": 40.0,
        "recall@1": 0.4,
        "recall@3": 0.7066666666666667,
        "recall@5": 0.8377777777777777,
        "mrr@5": 0.5622962962962963,
        "mean_rank": 2.991111111111111,
        "macro_f1": 0.33975311377417433
      },
      "Paraphrase": {
        "n": 1255,
        "correct": 666,
        "accuracy": 0.5306772908366534,
        "accuracy_pct": 53.06772908366534,
        "recall@1": 0.5306772908366534,
        "recall@3": 0.7099601593625497,
        "recall@5": 0.8254980079681274,
        "mrr@5": 0.6356972111553785,
        "mean_rank": 2.9235059760956177,
        "macro_f1": 0.5103082975511455
      }
    },
    "micro_aggregate": {
      "n": 3715,
      "correct": 1655,
      "accuracy": 0.44549125168236875,
      "accuracy_pct": 44.54912516823688
    }
  }
}
```
