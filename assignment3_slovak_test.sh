#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=2:0:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment3_slovak.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Test Slovak-English translation using Czech-English model
# The idea is to test how well a Czech-English model performs on Slovak-English
# since Czech and Slovak are closely related languages

# Create output directory for results
mkdir -p slovak-english/results

# TRANSLATE Slovak to English using Czech-English model
python translate.py \
    --cuda \
    --input slovak-english/sk.txt \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints_run_2/checkpoint_best.pt \
    --output slovak-english/results/translated_output.txt \
    --max-len 300 \
    --bleu \
    --reference slovak-english/en.txt