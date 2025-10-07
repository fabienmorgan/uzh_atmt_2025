# Weights & Biases Integration

This project now includes Weights & Biases (wandb) integration for tracking training progress and metrics.

## Setup

1. **Install wandb** (if not already installed):

   ```bash
   pip install wandb
   ```

2. **Get your API key**:

   - Visit https://wandb.ai/authorize
   - Copy your API key

3. **Configure your API key**:

   - Edit the `wandb_config.txt` file in the project root
   - Replace `YOUR_WANDB_API_KEY` with your actual API key
   - Optionally set your project name and entity (username/team)

   Example `wandb_config.txt`:

   ```
   WANDB_API_KEY=your_actual_api_key_here
   WANDB_PROJECT=my-translation-project
   WANDB_ENTITY=my_username
   ```

## Usage

Add the `--use-wandb` flag to your training command:

```bash
python train.py --use-wandb [other arguments...]
```

### Additional wandb arguments:

- `--wandb-project PROJECT_NAME`: Set project name (overrides config file)
- `--wandb-entity ENTITY_NAME`: Set entity name (overrides config file)
- `--wandb-run-name RUN_NAME`: Set a specific run name

Example:

```bash
python train.py --use-wandb --wandb-project "czech-english-translation" --wandb-run-name "transformer-baseline"
```

## What gets logged

The integration logs the following metrics:

### Training metrics (per epoch):

- `train/loss`: Training loss
- `train/lr`: Learning rate
- `train/num_tokens`: Number of tokens processed
- `train/batch_size`: Batch size
- `train/grad_norm`: Gradient norm
- `train/clip`: Gradient clipping occurrences
- `train/epoch_time`: Time to complete epoch

### Validation metrics (per epoch):

- `valid/perplexity`: Validation perplexity
- `valid/loss`: Validation loss
- `valid/bleu`: BLEU score on validation set

### Test metrics (final):

- `test/final_bleu`: Final BLEU score on test set
- `test/best_valid_perplexity`: Best validation perplexity achieved
- `test/examples`: Table with example translations

### Model configuration:

- All command-line arguments
- Model architecture details
- Total number of parameters
- Random seed used

## Security

- The `wandb_config.txt` file is ignored by git (added to `.gitignore`)
- Your API key will not be committed to the repository
- You can safely push your code without exposing credentials

## Viewing Results

After training starts, wandb will print a URL to view your training progress in real-time. You can also access your runs at https://wandb.ai/[your-username]/[project-name].
