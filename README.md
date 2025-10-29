# Heart Disease Decision Tree Model Card

**Model version:** v1  
**Training date:** 2025-10-28  

### Best Params
{'dt__max_depth': 5, 'dt__min_samples_leaf': 5}

### Metrics (Test Set)
{
  "roc_auc": 0.8189736738915427,
  "pr_auc": 0.8344524395789452,
  "precision": 0.7303788066353057,
  "recall": 0.9026927784577723,
  "f1": 0.8074449158341317,
  "specificity": 0.5291828793774317,
  "confusion_matrix": {
    "tn": 1224,
    "fp": 1089,
    "fn": 318,
    "tp": 2950
  }
}

### Notes
Educational demo only. Not medical advice.

Dependencies:
lucide-react: npm install lucide-react@latest
recharts: npm install recharts