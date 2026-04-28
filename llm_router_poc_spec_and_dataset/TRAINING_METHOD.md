# Training Method Notes

Install:

```bash
pip install pandas scikit-learn joblib
```

Recommended baseline:

1. Load `prompt_classifier_dataset_1000.csv`.
2. Split train/test 80/20.
3. Use `TfidfVectorizer` on prompt text.
4. Train:
   - LogisticRegression for task_type
   - GradientBoostingRegressor for difficulty
   - GradientBoostingRegressor for required_quality
   - LogisticRegression for risk_level
   - RandomForestRegressor for expected_output_tokens
   - LogisticRegression for latency_sensitivity
5. At runtime, combine predictions into one JSON object.
