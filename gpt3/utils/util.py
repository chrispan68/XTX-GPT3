import openai
import json
from jericho import *

def post_process(completion, mode):
    if mode == 'action-evaluation':
        lines = [x for x in completion.splitlines() if (not x.strip() == '')]
        act_evaluation_dict = {}
        for line in lines:
            try:
                act = line[:line.index(':')]
                evaluation = line[line.index(':') + 2:]
                act_evaluation_dict[act] = evaluation
            except:
                pass
        return act_evaluation_dict
    else:
        try:
            lines = [x for x in completion.splitlines() if (not x.strip() == '')]
            return lines[0].strip()
        except:
            return 'wait'
        
def complete(prompt, mode, engine, temperature, presence_penalty):
    if engine == 'stupid':
        if mode == 'explain':
            completion = ' We should probably go north right now.'
        elif mode == 'act':
            completion = ' go north'
        elif mode == 'action-evaluation':
            completion = '\ngo north: Good.\ngo south: Good.\ngo east: Good.\ngo west: Good.'
    elif engine == 'human':
        print(prompt, end='')
        completion = ''
        for line in stdin:
            completion += line
            break
    else: 
        if mode == 'explain':
            max_tokens = 100
        elif mode == 'act':
            max_tokens = 10
        elif mode == 'action-evaluation':
            max_tokens = 500
        raw = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, temperature=temperature, presence_penalty=presence_penalty)
        completion = raw['choices'][0]['text']
    return post_process(completion, mode)

def make_config(config_file):
    f = open(config_file, 'r')
    config = json.load(f)
    examples = []
    for filename in config['Example Filenames']:
        f = open(filename, 'r')
        examples.append(json.load(f))
    config['Examples'] = examples
    return config