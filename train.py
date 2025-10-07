import os
import random
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
import sentencepiece as spm
from collections import OrderedDict
import sacrebleu

import torch
import torch.nn as nn

# Wandb import with optional fallback
try:
    import wandb
except ImportError:
    wandb = None
    logging.warning("wandb not installed. Install with: pip install wandb")

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq import models, utils
from seq2seq.decode import decode
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

SEED = random.randint(1, 1_000_000_000)


def load_wandb_config():
    """Load wandb configuration from wandb_config.txt file."""
    config = {}
    config_file = os.path.join(os.path.dirname(__file__), 'wandb_config.txt')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value and value != 'YOUR_WANDB_API_KEY':
                        config[key] = value
    
    return config


def get_device_info():
    """Get device information for wandb logging."""
    device_info = {}
    
    if torch.cuda.is_available():
        device_info["device_type"] = "cuda"
        device_info["device_count"] = torch.cuda.device_count()
        device_info["current_device"] = torch.cuda.current_device()
        
        # Get GPU model name
        try:
            device_info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        except:
            device_info["gpu_name"] = "unknown"
        
        # Get GPU memory info
        try:
            memory_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            device_info["gpu_memory_gb"] = round(memory_total / (1024**3), 1)
        except:
            device_info["gpu_memory_gb"] = "unknown"
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info["device_type"] = "mps"
        device_info["gpu_name"] = "Apple Silicon"
    else:
        device_info["device_type"] = "cpu"
        
    return device_info


def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='indomain/preprocessed_data/', help='path to data directory')
    parser.add_argument('--source-lang', default='fr', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', help='path to target sentencepiece tokenizer', required=True)
    parser.add_argument('--max-tokens', default=None, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--train-on-tiny', action='store_true', help='train model on a tiny dataset')
    
    # # Add model arguments
    parser.add_argument('--arch', default='transformer', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=10000, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--max-length', default=300, type=int, help='maximum output sequence length during testing')
    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints_asg4', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')
    parser.add_argument('--ignore-checkpoints', action='store_true', help='don\'t load any previous checkpoint')
    
    # Add wandb arguments
    parser.add_argument('--use-wandb', action='store_true', default=True, help='enable Weights & Biases logging (default: True)')
    parser.add_argument('--no-wandb', action='store_false', dest='use_wandb', help='disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='seq2seq-translation', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity (username or team name)')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb-model-type', type=str, choices=['toy_example', 'assignment1', 'custom'], 
                        default=None, help='explicitly set model type for wandb organization')
    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args



def main(args):
    """ Main training function. Trains the translation model over the course of several epochs, including dynamic
    learning rate adjustment and gradient clipping. """
    
    # Training configuration
    log_interval = 100  # Steps between wandb logging (also used as rolling window size)
    
    # Record start time and log timestamp
    training_start_time = time.time()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time))
    logging.info('=' * 60)
    logging.info(f'Training started at: {start_timestamp}')
    logging.info('=' * 60)
    logging.info('Commencing training!')
    torch.manual_seed(SEED)

    utils.init_logging(args)
    
    # Initialize wandb if requested
    wandb_available = False
    if args.use_wandb and wandb is not None:
        logging.info("Wandb logging requested - loading configuration...")
        
        # Load wandb configuration
        wandb_config = load_wandb_config()
        # Only log non-sensitive config keys
        safe_keys = [key for key in wandb_config.keys() if 'API_KEY' not in key and 'TOKEN' not in key.upper()]
        logging.info(f"Loaded wandb config keys: {safe_keys}")
        
        # Set environment variables if provided
        if 'WANDB_API_KEY' in wandb_config:
            os.environ['WANDB_API_KEY'] = wandb_config['WANDB_API_KEY']
            logging.info("✓ API key loaded from config file")
        else:
            logging.warning("⚠ No API key found in config file - make sure it's set in wandb_config.txt")
        
        # Use config values or command line arguments
        project = wandb_config.get('WANDB_PROJECT', args.wandb_project)
        entity = wandb_config.get('WANDB_ENTITY', args.wandb_entity)
        
        # Use model type from command line argument, default to "custom"
        model_type = args.wandb_model_type or "custom"
        
        # Get device information
        device_info = get_device_info()
        
        # Create enhanced tags
        tags = [
            f"arch_{args.arch}", 
            f"lr_{args.lr}",
            f"model_type_{model_type}",
            f"lang_pair_{args.source_lang}-{args.target_lang}",
            f"device_{device_info['device_type']}"
        ]
        
        # Add dataset size tag
        if args.train_on_tiny:
            tags.append("dataset_tiny")
        else:
            tags.append("dataset_full")
        
        # Initialize wandb first to get the auto-generated ID
        logging.info("Initializing wandb...")
        logging.info(f"Project: {project}")
        logging.info(f"Entity: {entity}")
        logging.info(f"Model type: {model_type}")
        logging.info(f"Tags: {tags}")
        
        try:
            # Initialize wandb without a custom name first to get the auto-generated ID
            wandb.init(
                project=project,
                entity=entity,
                config=vars(args),
                tags=tags,
                notes=f"Training {args.arch} model ({model_type}) with {args.source_lang}->{args.target_lang}",
                group=model_type  # This groups runs by model type in the wandb UI
            )
            
            # Now create a descriptive run name using the wandb ID
            base_run_name = args.wandb_run_name
            if base_run_name is None:
                base_run_name = f"{model_type}_{args.arch}_{args.source_lang}-{args.target_lang}"
                if args.train_on_tiny:
                    base_run_name += "_tiny"
            
            # Add the wandb ID to make it unique
            unique_run_name = f"{base_run_name}_{wandb.run.id}"
            
            # Update the run name
            wandb.run.name = unique_run_name
            
            # Log additional config
            wandb.config.update({
                "seed": SEED,
                "model_type": model_type,
                "dataset_type": "tiny" if args.train_on_tiny else "full",
                "log_interval": log_interval,
                **device_info  # Add all device information
            })
            # Note: total_params will be added after model creation
            
            logging.info("✓ Wandb initialized successfully!")
            logging.info(f"✓ Wandb run URL: {wandb.run.url}")
            logging.info(f"✓ Wandb run ID: {wandb.run.id}")
            logging.info(f"✓ Wandb run name: {unique_run_name}")
            logging.info(f"✓ Wandb project: {wandb.run.project}")
            if wandb.run.entity:
                logging.info(f"✓ Wandb entity: {wandb.run.entity}")
            
            wandb_available = True
                
        except Exception as e:
            logging.error(f"✗ Failed to initialize wandb: {str(e)}")
            logging.error("Continuing without wandb logging...")
            wandb_available = False
    elif args.use_wandb and wandb is None:
        logging.warning("✗ Wandb requested but not installed. Install with: pip install wandb")
        logging.warning("✗ Training will continue WITHOUT wandb logging")
    elif args.use_wandb and not wandb_available:
        logging.warning("✗ Wandb was requested but initialization failed")
        logging.warning("✗ Training will continue WITHOUT wandb logging")
    elif not args.use_wandb:
        logging.info("ℹ Wandb logging disabled (use --use-wandb to enable)")
    
    # Log final wandb status
    if wandb_available:
        logging.info("✓ Wandb logging is ACTIVE")
    else:
        logging.info("✗ Wandb logging is INACTIVE")

    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.source_lang)),
            tgt_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.target_lang)),
            src_model=args.src_tokenizer, tgt_model=args.tgt_tokenizer)

    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    train_dataset = load_data(split='train') if not args.train_on_tiny else load_data(split='tiny_train')
    valid_dataset = load_data(split='valid')

    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info('Built a model with {:d} parameters'.format(total_params))
    
    # Update wandb config with total parameters
    if wandb_available:
        try:
            wandb.config.update({"total_params": total_params}, allow_val_change=True)
            logging.info("✓ Model parameters logged to wandb")
        except Exception as e:
            logging.warning(f"✗ Failed to log model parameters to wandb: {str(e)}")
    elif args.use_wandb:
        logging.warning("✗ Cannot log model parameters - wandb not available")
    
    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_id(), reduction='sum')

    # Move model to GPU if available
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    device = torch.device("cuda" if args.cuda else "cpu")
    
    
    # Instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    state_dict = None
    if not args.ignore_checkpoints:
        # Load last checkpoint if one exists
        state_dict = utils.load_checkpoint(args, model, optimizer)  # lr_scheduler
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1
    
    # Track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    make_batch = utils.make_batch_input(device=device, pad=src_tokenizer.pad_id(), max_seq_len=args.max_seq_len)
    
    # Log wandb status before training starts
    if args.use_wandb and wandb_available:
        logging.info("✓ Starting training with wandb logging enabled")
    elif args.use_wandb and not wandb_available:
        logging.warning("⚠ Starting training - wandb was requested but is not working")
    
    # Initialize global step counter and rolling windows
    global_step = 0  # Track global step across all epochs
    recent_losses = []  # Rolling window persists across epochs
    recent_grad_norms = []  # Rolling window persists across epochs
    
    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = \
            torch.utils.data.DataLoader(train_dataset, num_workers=1, collate_fn=train_dataset.collater,
                                        batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, 1,
                                                                   0, shuffle=True, seed=SEED))
        model.train()
        stats = OrderedDict()
        stats['loss'] = 0
        stats['lr'] = 0
        stats['num_tokens'] = 0
        stats['batch_size'] = 0
        stats['grad_norm'] = 0
        stats['clip'] = 0
        
        # Display progress
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False,
                            # update progressbar every 2 seconds
                            mininterval=2.0)

        # Iterate over the training set
        start_time = time.perf_counter()
        for i, sample in enumerate(progress_bar):
            global_step += 1
            
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            model.train()
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch(x=sample['src_tokens'],
                                                                           y=sample['tgt_tokens'])
            
            output = model(src, src_pad_mask, trg_in, trg_pad_mask).to(device)

            loss = \
                criterion(output.view(-1, output.size(-1)), trg_out) / len(sample['src_lengths'])

            if torch.isnan(loss).any():
                logging.warning('Loss is NAN!')
                print(src_tokenizer.Decode(sample['src_tokens'].tolist()[0]), '---', tgt_tokenizer.Decode(sample['tgt_tokens'].tolist()[0]))
                # print()
                # import pdb;pdb.set_trace()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # Gradient norm - monitors training stability
            optimizer.step()
            optimizer.zero_grad()

            # Update statistics for progress bar (matching original logic)
            total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])
            
            # Calculate weighted loss for consistency (as used in original)
            weighted_step_loss = total_loss * len(sample['src_lengths']) / sample['num_tokens']
            step_grad_norm = float(grad_norm)
            
            # Update stats exactly like original
            stats['loss'] += weighted_step_loss
            stats['lr'] += optimizer.param_groups[0]['lr'] 
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            
            # Rolling averages for wandb - use SAME weighted loss for consistency
            if not (np.isnan(weighted_step_loss) or np.isinf(weighted_step_loss)):
                recent_losses.append(weighted_step_loss)
            
            if not (np.isnan(step_grad_norm) or np.isinf(step_grad_norm)):
                recent_grad_norms.append(step_grad_norm)
            
            # Maintain window size - lightweight
            if len(recent_losses) > log_interval:
                recent_losses.pop(0)
            if len(recent_grad_norms) > log_interval:
                recent_grad_norms.pop(0)
            
            # Simple rolling averages - fast calculation
            rolling_avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else weighted_step_loss
            rolling_avg_grad_norm = sum(recent_grad_norms) / len(recent_grad_norms) if recent_grad_norms else step_grad_norm
            rolling_avg_perplexity = np.exp(rolling_avg_loss)
            
            # Step-level wandb logging (rolling averages only)
            if wandb_available and global_step % log_interval == 0:
                try:
                    step_metrics = {
                        "train/rolling_avg_loss": rolling_avg_loss,
                        "train/rolling_avg_perplexity": rolling_avg_perplexity,
                        "train/rolling_avg_grad_norm": rolling_avg_grad_norm
                    }
                    wandb.log(step_metrics, step=global_step)
                except Exception as e:
                    logging.warning(f"✗ Failed to log step metrics to wandb at step {global_step}: {str(e)}")
            
            # Update progress bar (matching original format but with rolling averages)
            progress_bar.set_postfix({
                'loss': '{:.4g}'.format(stats['loss'] / (i + 1)),
                'lr': '{:.4g}'.format(stats['lr'] / (i + 1)),
                'tokens': '{:.4g}'.format(stats['num_tokens'] / (i + 1)),
                'grad': '{:.4g}'.format(stats['grad_norm'] / (i + 1)),
                f'avg_{log_interval}': '{:.3f}'.format(rolling_avg_loss)},
                refresh=False)
        # measure time to complete epoch (training only)
        epoch_time = time.perf_counter()- start_time
        
        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))
        logging.info(f'Time to complete epoch {epoch:03d} (training only): {epoch_time:.2f} seconds')

        # Calculate validation loss
        valid_perplexity, valid_bleu = validate(args, model, criterion, valid_dataset, epoch, batch_fn=make_batch, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        model.train()
        
        # Calculate epoch-level training perplexity (matching original)
        epoch_train_loss = stats['loss'] / len(progress_bar)
        epoch_train_perplexity = np.exp(epoch_train_loss)
        
        # Log epoch-level metrics to wandb
        if wandb_available:
            try:
                epoch_metrics = {
                    "train/epoch_loss": epoch_train_loss,
                    "train/epoch_perplexity": epoch_train_perplexity,
                    "train/epoch_grad_norm": stats['grad_norm'] / len(progress_bar),
                    "valid/perplexity": valid_perplexity,
                    "valid/loss": np.log(valid_perplexity),
                    "epoch": epoch  # Keep epoch for x-axis grouping
                }
                
                # Add BLEU score if available
                if valid_bleu is not None:
                    epoch_metrics["valid/bleu"] = valid_bleu
                    
                wandb.log(epoch_metrics, step=global_step)
                logging.debug(f"Logged epoch metrics to wandb for epoch {epoch}")
            except Exception as e:
                logging.warning(f"✗ Failed to log epoch metrics to wandb for epoch {epoch}: {str(e)}")
                logging.warning("✗ Continuing training without wandb logging for this epoch")
        
        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_perplexity)  # lr_scheduler

        # Check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break

    # Final evaluation on the test set
    test_dataset = load_data(split='test')
    logging.info('Loading the best model for final evaluation on the test set')
    utils.load_checkpoint(args, model, optimizer)

    # Evaluate the model on the test set
    bleu_score, all_hypotheses, all_references = evaluate(
        args,
        model,
        test_dataset,
        batch_fn=make_batch,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    logging.info('Final Test Set Results: BLEU {:.2f}'.format(bleu_score))
    
    # Calculate and log total training time BEFORE wandb logging
    training_end_time = time.time()
    end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_end_time))
    total_duration = training_end_time - training_start_time
    
    # Log final test results to wandb
    if wandb_available:
        try:
            logging.info("Logging final results to wandb...")
            final_metrics = {
                "final/test_bleu": bleu_score,  # Final BLEU score for plotting
                "final/best_valid_perplexity": best_validate,
                "training/total_duration_seconds": total_duration,
                "training/total_duration_hours": total_duration / 3600,
                "training/start_time": training_start_time,  # Use unix timestamp
                "training/end_time": training_end_time      # Use unix timestamp
            }
            wandb.log(final_metrics)
            
            # Add timestamp strings and other summary info to wandb summary
            wandb.summary.update({
                "start_timestamp_str": start_timestamp,
                "end_timestamp_str": end_timestamp,
                "total_epochs_trained": epoch + 1,
                "early_stopped": bad_epochs >= args.patience,
                "final_test_bleu": bleu_score,
                "best_validation_perplexity": best_validate
            })
            
            logging.info("✓ Final metrics logged to wandb")
            
            # Create a summary table with some example translations
            if len(all_hypotheses) > 0:
                logging.info("Creating example translations table...")
                examples = []
                for i in range(min(10, len(all_hypotheses))):
                    examples.append([i, all_references[i], all_hypotheses[i]])
                
                wandb.log({
                    "test/examples": wandb.Table(
                        columns=["Index", "Reference", "Hypothesis"],
                        data=examples
                    )
                })
                logging.info("✓ Example translations table logged to wandb")
            
            logging.info("Finishing wandb run...")
            wandb.finish()
            logging.info("✓ Wandb run finished successfully")
            
        except Exception as e:
            logging.error(f"✗ Error logging final results to wandb: {str(e)}")
            logging.error("✗ Final wandb logging failed - results may not appear in wandb dashboard")
            try:
                wandb.finish()
                logging.info("✓ Wandb session closed despite logging errors")
            except:
                logging.warning("✗ Failed to properly close wandb session")
    
    # Format duration in a human-readable way (variables already calculated above)
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    logging.info('=' * 60)
    logging.info(f'Training completed at: {end_timestamp}')
    logging.info(f'Total training time: {hours:02d}h:{minutes:02d}m:{seconds:02d}s ({total_duration:.2f} seconds)')
    logging.info('=' * 60)


def validate(args, model, criterion, valid_dataset, epoch,
             batch_fn: callable,
             src_tokenizer: spm.SentencePieceProcessor,
             tgt_tokenizer: spm.SentencePieceProcessor):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
                                    batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                               shuffle=False, seed=SEED))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0

    device = torch.device('cuda' if args.cuda else 'cpu')

    all_references = []  # list of reference strings
    all_hypotheses = []  # list of hypothesis strings

    progress_bar = tqdm(valid_loader, desc='| Validating Epoch {:03d}'.format(epoch), leave=False, disable=False, 
                        # update progressbar every 2 seconds
                        mininterval=2.0)
    # Iterate over the validation set
    for i, sample in enumerate(progress_bar):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            src_tokens, trg_in, trg_out, src_pad_mask, trg_pad_mask = batch_fn(x=sample['src_tokens'],
                                                                               y=sample['tgt_tokens'])
            # Compute loss (with teacher forcing)
            output = model(src_tokens, src_pad_mask, trg_in, trg_pad_mask).to(device)
            loss = criterion(output.view(-1, output.size(-1)), trg_out.view(-1))
            
            # Decoding for BLEU (no teacher forcing)
            predicted_tokens = decode(model=model,
                                      src_tokens=src_tokens,
                                      src_pad_mask=src_pad_mask,
                                      max_out_len=args.max_length,
                                      tgt_tokenizer=tgt_tokenizer,
                                      args=args,
                                      device=device)

        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])

        # Collect references and hypotheses for BLEU
        if tgt_tokenizer is not None:
            for ref_tgt, hyp_src in zip(sample['tgt_tokens'], predicted_tokens):
                ref_sentence = tgt_tokenizer.Decode(ref_tgt.tolist())
                hyp_sentence = tgt_tokenizer.Decode(hyp_src)

                all_references.append(ref_sentence)
                all_hypotheses.append(hyp_sentence)

    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']

    # Compute BLEU with sacrebleu
    bleu_score = None
    if src_tokenizer is not None and tgt_tokenizer is not None and len(all_hypotheses) > 0:
        bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
        bleu_score = bleu.score

    # Logging
    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity) +
        ('' if bleu_score is None else ' | BLEU {:.3f}'.format(bleu_score))
    )

    return perplexity, bleu_score
    

def evaluate(args, model, test_dataset,
    batch_fn: callable,
    src_tokenizer: spm.SentencePieceProcessor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    decode_kwargs: dict = None,
):
    """Evaluates the model on a test set using sacrebleu.
       decode_fn: function that generates translations.
       decode_kwargs: dict of extra parameters for decode_fn.
    """
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=1,
        collate_fn=test_dataset.collater,
        # batch_size != 1 may mess things up with decoding
        batch_sampler=BatchSampler(test_dataset, args.max_tokens, batch_size=1, 
                                   num_shards=1, shard_id=0, shuffle=False, seed=SEED),
    )

    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")

    all_references = []
    all_hypotheses = []
    decode_kwargs = decode_kwargs or {}

    progress_bar = tqdm(test_loader, desc='| Evaluating', leave=False, disable=False,
                        # update progressbar every 2 seconds
                        mininterval=2.0)   
    # Iterate over test set
    for i, sample in enumerate(progress_bar):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue

        with torch.no_grad():
            src_tokens, tgt_in, tgt_out, src_pad_mask, _ = batch_fn(
                x=sample["src_tokens"], y=sample["tgt_tokens"]
            )

            #-----------------------------------------
            # Decode without teacher forcing
            prediction = decode(model=model,
                                      src_tokens=src_tokens,
                                      src_pad_mask=src_pad_mask,
                                      max_out_len=args.max_length,
                                      tgt_tokenizer=tgt_tokenizer,
                                      args=args,
                                      device=device)
            #----------------------------------------

        # Collect hypotheses and references
        for ref, hyp in zip(sample["tgt_tokens"], prediction):
        # the for-loop is technically redundant since batch_size=1, but kept for clarity
            ref_sentence = tgt_tokenizer.Decode(ref.tolist())
            hyp_sentence = tgt_tokenizer.Decode(hyp)

            all_references.append(ref_sentence)
            all_hypotheses.append(hyp_sentence)

    # Compute BLEU with sacrebleu
    bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
    bleu_score = bleu.score

    logging.info("Test set results: BLEU {:.3f}".format(bleu_score))

    return bleu_score, all_hypotheses, all_references



if __name__ == '__main__':
    args = get_args()
    args.seed = SEED
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # Set up logging to file
    logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
    main(args)
