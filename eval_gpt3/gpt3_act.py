from os.path import join as pjoin
import jericho
import pickle 
from gpt3 import *
import json
import argparse

openai_api_key = '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'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bottleneck_directory', type=str, required=True)
    parser.add_argument('--rom_path', type=str, required=True)
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpt3_steps',type=int,default=10)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--engine',type=str, default='text-davinci-002')
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.load(open(args.config_dir, 'r'))
    config['Examples'] = []
    for filename in config['Example Filenames']:
        config['Examples'].append(json.load(open(filename, 'r')))
    prompt_type = 'multiaction' if 'Evaluation' in config['Features'] else 'oneaction'
    filename = pjoin(args.bottleneck_directory, 'gpt3acts_' + prompt_type + '.json')

    if os.path.exists(filename):
        print('File Already Exists. Stopping...')
        return
    
    env = jericho.FrotzEnv(args.rom_path, seed=args.seed)
    starting_state = pickle.load(open(pjoin(args.bottleneck_directory, 'env_state.pickle'), 'rb'))
    env.set_state(starting_state)
    
    gpt3_helper = GPTHelper(engine=args.engine, config=config, openai_api_key=openai_api_key, args=args)
    acts = gpt3_helper.get_actions(env, max_steps=args.gpt3_steps,print_progress=True)
    json.dump(acts, open(filename, 'w'), indent=4)
    

if __name__ == "__main__":
    main()
