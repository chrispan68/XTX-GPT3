# Built-in imports
import argparse
import random
import logging
import pickle
from os.path import join as pjoin
from os.path import exists as pexists
from os import makedirs
import json

# Third party imports
import jericho
import torch
import numpy as np
import wandb
import pandas as pd

# Custom imports
from agents import (
    DrrnAgent,
    DrrnInvDynAgent,
    DrrnGraphInvDynAgent
)

from trainers import (
    DrrnTrainer,
    DrrnInvDynTrainer,
    DrrnGraphInvDynTrainer,
    DrrnInvDynEvaluator
)

from transformers import GPT2LMHeadModel, GPT2Config

import definitions.defs as defs
from utils.env import JerichoEnv
from utils.vec_env import VecEnv
from utils import logger
from utils.memory import State, Transition


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger().setLevel(logging.CRITICAL)
# torch.autograd.set_detect_anomaly(True)


def configure_logger(args):
    """
    Setup various logging channels (wandb, text files, etc.).
    """
    if not pexists(args.output_dir):
        makedirs(args.output_dir)
        
    log_dir = args.output_dir
    wandb_on = args.wandb

    type_strs = ["json", "stdout"]
    if wandb_on and log_dir != "logs":
        type_strs += ["wandb"]
    tb = logger.Logger(
        log_dir,
        [
            logger.make_output_format(type_str, log_dir, args=args)
            for type_str in type_strs
        ],
    )

    logger.configure("{}-{}".format(log_dir, wandb.run.id),
                     format_strs=["log", "stdout"], off=args.logging_off)
    log = logger.log

    return tb, log


def parse_args():
    parser = argparse.ArgumentParser()

    # General Settings
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--rom_path', default='./games/detective.z5')
    parser.add_argument('--wandb', default=1, type=int)
    parser.add_argument('--save_path', default='princeton-nlp/text-games/')
    parser.add_argument('--bottleneck_cache_path', default='bottleneck_state_cache')
    parser.add_argument('--logging_off', default=0, type=int)
    parser.add_argument('--weight_file', default=None, type=str)
    parser.add_argument('--memory_file', default=None, type=str)
    parser.add_argument('--traj_file', default=None, type=str)
    parser.add_argument('--run_id', default=None, type=str)
    parser.add_argument('--project_name', default='xtx', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--jericho_add_wt', default='add_wt', type=str)

    # Environment settings
    parser.add_argument('--check_valid_actions_changed', default=0, type=int)

    # Training Settings
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--dynamic_episode_length', default=0, type=int)
    parser.add_argument('--episode_ext_type', default='steady_50', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--jericho_seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--q_update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=5000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--target_update_freq', default=100, type=int)
    parser.add_argument('--dump_traj_freq', default=5000, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--memory_size', default=500000, type=int)
    parser.add_argument('--memory_alpha', default=.4, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--no_invalid_act_detect', default=0, type=int)
    parser.add_argument('--filter_invalid_acts', default=1, type=int)
    parser.add_argument('--start_from_reward', default=0, type=int)
    parser.add_argument('--start_from_wt', default=0, type=int)
    parser.add_argument('--filter_drop_acts', default=0, type=int)
    parser.add_argument('--cache_bottlenecks', default=0, type=int)
    parser.add_argument('--num_steps_bottleneck', default=1000, type=int)
    
    # Action Model Settings
    parser.add_argument('--max_acts', default=5, type=int)
    parser.add_argument('--tf_embedding_dim', default=128, type=int)
    parser.add_argument('--tf_hidden_dim', default=128, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--feedforward_dim', default=512, type=int)
    parser.add_argument('--tf_num_layers', default=3, type=int)
    parser.add_argument('--ngram', default=3, type=int)
    parser.add_argument('--traj_k', default=1, type=int)
    parser.add_argument('--action_model_update_freq', default=1e9, type=int)
    parser.add_argument('--smooth_alpha', default=0.00001, type=float)
    parser.add_argument('--cut_beta_at_threshold', default=0, type=int)
    parser.add_argument('--action_model_type', default='ngram', type=str)
    parser.add_argument('--tf_num_epochs', default=50, type=int)
    parser.add_argument(
        '--turn_action_model_off_after_falling', default=0, type=int)
    parser.add_argument('--traj_dropout_prob', default=0, type=float)
    parser.add_argument('--init_bin_prob', default=0.1, type=float)
    parser.add_argument('--num_bins', default=0, type=int)
    parser.add_argument('--binning_prob_update_freq', default=1e9, type=int)
    parser.add_argument('--random_action_dropout', default=0, type=int)
    parser.add_argument('--use_multi_ngram', default=0, type=int)
    parser.add_argument('--use_action_model', default=0, type=int)
    parser.add_argument('--sample_action_argmax', default=0, type=int)
    parser.add_argument('--il_max_context', default=512, type=int)
    parser.add_argument('--il_k', default=5, type=int)
    parser.add_argument('--il_batch_size', default=64, type=int)
    parser.add_argument('--il_lr', default=1e-3, type=float)
    parser.add_argument('--il_max_num_epochs', default=200, type=int)
    parser.add_argument('--il_num_eval_runs', default=3, type=int)
    parser.add_argument('--il_eval_freq', default=300, type=int)
    parser.add_argument('--il_vocab_size', default=2000, type=int)
    parser.add_argument('--il_temp', default=1., type=float)
    parser.add_argument('--use_il', default=0, type=int)
    parser.add_argument('--il_len_scale', default=1.0, type=float)
    parser.add_argument('--use_il_graph_sampler', default=0, type=int)
    parser.add_argument('--use_il_buffer_sampler', default=1, type=int)
    parser.add_argument('--il_top_p', default=0.9, type=float)
    parser.add_argument('--il_use_dropout', default=0, type=int)
    parser.add_argument('--il_use_only_dropout', default=0, type=int)

    # DRRN Model Settings
    parser.add_argument('--drrn_embedding_dim', default=128, type=int)
    parser.add_argument('--drrn_hidden_dim', default=128, type=int)
    parser.add_argument('--use_drrn_inv_look', default=1, type=int)
    parser.add_argument('--use_counts', default=0, type=int)
    parser.add_argument('--reset_counts_every_epoch', default=0, type=int)
    parser.add_argument('--sample_uniform', default=0, type=int)
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--rotating_temp', default=0, type=int)
    parser.add_argument('--augment_state_with_score', default=0, type=int)

    # Graph Model Settings
    parser.add_argument('--graph_num_explore_steps', default=50, type=int)
    parser.add_argument('--graph_rescore_freq', default=500, type=int)
    parser.add_argument('--graph_merge_freq', default=500, type=int)
    parser.add_argument('--graph_hash', default='inv_loc_ob', type=str)
    parser.add_argument('--graph_score_temp', default=1, type=float)
    parser.add_argument('--graph_q_temp', default=1, type=float)
    parser.add_argument('--graph_alpha', default=0.5, type=float)
    parser.add_argument('--log_top_blue_acts_freq', default=100, type=int)

    # Offline Q Learning settings
    parser.add_argument('--offline_q_steps', default=1000, type=int)
    parser.add_argument('--offline_q_transfer_freq', default=100, type=int)
    parser.add_argument('--offline_q_eval_runs', default=10, type=int)

    # Inv-Dyn Settings
    parser.add_argument('--type_inv', default='decode')
    parser.add_argument('--type_for', default='ce')
    parser.add_argument('--w_inv', default=0, type=float)
    parser.add_argument('--w_for', default=0, type=float)
    parser.add_argument('--w_act', default=0, type=float)
    parser.add_argument('--r_for', default=0, type=float)

    parser.add_argument('--nor', default=0, type=int, help='no game reward')
    parser.add_argument('--randr', default=0, type=int,
                        help='random game reward by objects and locations within episode')
    parser.add_argument('--perturb', default=0, type=int,
                        help='perturb state and action')

    parser.add_argument('--hash_rep', default=0, type=int,
                        help='hash for representation')
    parser.add_argument('--act_obs', default=0, type=int,
                        help='action set as state representation')
    parser.add_argument('--fix_rep', default=0, type=int,
                        help='fix representation')

    # Additional Model Settings
    parser.add_argument('--model_name', default='xtx', type=str)
    parser.add_argument('--beta', default=0.3, type=float)
    parser.add_argument('--beta_trainable', default=0, type=int)
    parser.add_argument(
        '--eps',
        default=0,
        type=int,
        help='0: ~ softmax act_value; 1: eps-greedy-exploration',
    )
    parser.add_argument(
        '--eps_type',
        default='uniform',
        type=str,
        help='uniform (-1): uniform exploration; softmax_lm (0): ~ softmax lm_value; uniform_lm_topk (>0): ~ uniform(top k w.r.t. lm_value)',
    )
    parser.add_argument(
        '--alpha',
        default=0,
        type=float,
        help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now',
    )
    parser.add_argument('--sample_argmax',
                        default=0,
                        type=int,
                        help='whether to replace sampling with argmax')
    
    # bottleneck evaluation args
    parser.add_argument('--bottleneck_directory', type=str, required=True)
    parser.add_argument('--gpt3_acts_filename', type=str, required=True)
    parser.add_argument('--aggregate_results_filename',type=str, default='eval_gpt3/results.csv')
    parser.add_argument('--gpt3_steps',type=int,required=True)
    return parser.parse_args()


def main():
    assert jericho.__version__.startswith(
        "3"), "This code is designed to be run with Jericho version >= 3.0.0."

    args = parse_args()
    print(args)
    print("device", device)
    print(args.model_name)

    # Set seed across imports
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Start logger
    tb, log = configure_logger(args)

    if args.debug:
        import pdb
        pdb.set_trace()

    # Setup envs
    starting_state = pickle.load(open(pjoin(args.bottleneck_directory, 'env_state.pickle'), 'rb'))
    starting_env = jericho.FrotzEnv(args.rom_path, seed=args.jericho_seed)
    starting_env.set_state(starting_state)
    starting_score = starting_env.get_score()
    gpt3acts = json.load(open(args.gpt3_acts_filename, 'r'))
    if 'Evaluation' in gpt3acts[0]:
        prompting='multiaction'
    else:
        prompting='oneaction'
    
    for step in gpt3acts[:args.gpt3_steps]:
        starting_env.step(step['Best Action'])
    xtx_starting_score = starting_env.get_score()
    starting_state = starting_env.get_state()
    
    cache = dict()
    eval_env = JerichoEnv(args.rom_path,
                          args.env_step_limit,
                          get_valid=True,
                          seed=args.jericho_seed,
                          args=args,
                          cache=cache,
                          start_from_reward=args.start_from_reward,
                          start_from_wt=args.start_from_wt,
                          log=log,
                          starting_state=starting_state)
    envs = [
        JerichoEnv(args.rom_path,
                   args.env_step_limit,
                   get_valid=True,
                   cache=cache,
                   args=args,
                   seed=args.jericho_seed,
                   start_from_reward=args.start_from_reward,
                   start_from_wt=args.start_from_wt,
                   log=log,
                   starting_state=starting_state) for _ in range(args.num_envs)
    ]

    assert(args.model_name == defs.INV_DY) # Only supports INV_DYN for now.

    if args.model_name == defs.INV_DY:
        assert args.r_for > 0, "r_for needs to be ON when using inverse dynamics."
        assert args.use_action_model == 0, "'use_action_model' needs to be OFF."

        envs = VecEnv(args.num_envs, eval_env)

        agent = DrrnInvDynAgent(args, None, tb, log, envs)
        trainer = DrrnInvDynEvaluator(tb, log, agent, envs, eval_env, args)

    else:
        raise Exception("Unknown model type!")

    agent.load_from_dir(args.bottleneck_directory)
    
    try:
        scores = trainer.train()
        column_names = ['game', 'prompting_type', 'bottleneck_dir', 'num_steps_gpt3', 'starting_score', 'xtx_starting_score', 'final_score', 'gpt3win', 'gpt3xtxwin']
        data = []
        for score in scores:
            data.append([starting_env.bindings['name'], prompting, args.bottleneck_directory, args.gpt3_steps, starting_score, xtx_starting_score, score, 1 if (xtx_starting_score > starting_score) else 0, 1 if (score > starting_score) else 0])
        data = pd.DataFrame(columns=column_names, data=data)
        if pexists(args.aggregate_results_filename):
            df_existing = pd.read_csv(args.aggregate_results_filename, index_col=0)
            data = df_existing.append(data, ignore_index=True)
        data.to_csv(args.aggregate_results_filename)
            

    finally:
        for ps in envs.ps:
            ps.terminate()


if __name__ == "__main__":
    main()
