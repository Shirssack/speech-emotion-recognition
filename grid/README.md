# Grid Search Results Directory

This directory stores hyperparameter tuning results from grid search optimization.

## Files Generated

When you run `grid_search.py`, it creates JSON files with results:
- `grid_search_results_YYYYMMDD_HHMMSS.json` - Timestamped results

## Result Format

Each file contains:
- Model configurations tested
- Training accuracy
- Test accuracy
- Best hyperparameters found
- Execution time

## Usage

Run grid search:
```bash
python grid_search.py --model MLP --fast
```

View results:
```bash
cat grid/*.json | jq '.'
```
