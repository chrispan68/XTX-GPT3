from jericho import *
from re import sub
import string
import random
import copy

class JerichoEnvWrapper():
    def __init__(
        self,
        env: FrotzEnv,
        features_list: list,
        instruction: str,
        in_context_examples: list
    ):
        assert len(in_context_examples) > 0 

        self.context = []
        self.env = env.copy() # make sure we don't make updates to the input environment
        self.features = features_list
        self.output_features = list(in_context_examples[0].keys())
        self.prompt = self._get_prompt(instruction, in_context_examples)

        step_context = {}
        step_context['Look'] = self._get_look()
        step_context['Observation'] = step_context['Look'] # Not ideal, but this was the best workaround I could come up with.
        step_context['Inventory'] = self._get_inventory()
        step_context['Possible Actions'] = self._get_val_actions()
        step_context['Situation'] = self._get_situation(step_context['Observation'], step_context['Inventory'], step_context['Look'])
        self.context.append(step_context)
    
    def _contains(self, situation, obs):
        obs_words = obs.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).split()
        cnt = 0
        for word in obs_words:
            if word in situation:
                cnt += 1
        return ((cnt / len(obs_words)) >= 0.8)

    def _get_val_actions(self):
        val_actions_list = self.env.get_valid_actions()
        return ', '.join(val_actions_list) + '.'
    
    def _get_situation(self, obs, inv, look):
        situation = ''
        if len(self.context) >= 1:
            movement_actions = ['north', 'east', 'south', 'west', 'up', 'down', 'southeast', 'southwest', 'northeast', 'northwest']
            if self.context[-1]['Best Action'] in movement_actions:
                situation += 'You go ' + self.context[-1]['Best Action'] + '. '
            else:
                situation += 'You ' + self.context[-1]['Best Action'] + '. '
        
        situation += obs + ' '
        if not self._contains(situation, look):
            situation += look + ' '
        situation += inv
        return situation.strip()
    
    def _get_inventory(self):
        state = self.env.get_state()
        inv, _, _, _ = self.env.step('inventory')
        inv = inv.replace('\n', ' ').replace('   ', ' ').strip()
        if len(inv) > 0 and inv[-1] != '.':
            inv += '.'
        self.env.set_state(state)
        return inv
    
    def _get_look(self):
        state = self.env.get_state()
        look, _, _, _ = self.env.step('look')
        self.env.set_state(state)
        return look.replace('\n', ' ').strip()

    def _get_prompt(self, instruction, in_context_examples):
        prompt = instruction + '\n\n'
        for example in in_context_examples:
            prompt += self._step_to_string(example)
        return prompt

    def _sample_action(self):
        good = []
        neutral = []
        bad = []
        for act, evaluation in self.context[-1]['Evaluation'].items():
            if 'good' in evaluation.lower():
                good.append(act)
            elif 'bad' in evaluation.lower():
                bad.append(act)
            else:
                neutral.append(act)

        if len(good) > 0:
            best = good
        elif len(neutral) > 0:
            best = neutral
        else:
            best = bad
        
        return random.sample(best, 1)[0]

    def step(self, action=None, explanation=None, action_evaluation=None):
        if explanation:
            self.context[-1]['Explanation'] = explanation
            return
        if action:
            self.context[-1]['Best Action'] = action
            obs, reward, done, info = self.env.step(self.context[-1]['Best Action'])
        if action_evaluation:
            self.context[-1]['Evaluation'] = action_evaluation
            self.context[-1]['Best Action'] = self._sample_action()
            obs, reward, done, info = self.env.step(self.context[-1]['Best Action'])
        if done:
            step_context = {}
            step_context['Observation'] = obs.replace('\n', ' ').strip()
            self.context.append(step_context)
            return True, {key: self.context[-2][key] for key in self.output_features}
        step_context = {}
        step_context['Observation'] = obs.replace('\n', ' ').strip()
        step_context['Inventory'] = self._get_inventory()
        step_context['Look'] = self._get_look()
        step_context['Possible Actions'] = self._get_val_actions()
        step_context['Situation'] = self._get_situation(step_context['Observation'], step_context['Inventory'], step_context['Look'])
        self.context.append(step_context)
        return False, {key: self.context[-2][key] for key in self.output_features}

    def _step_to_string(self, step_context, log=False):
        output = ''
        if log:
            features = self.output_features
        else:
            features = self.features
        for field in self.features:
            if field in step_context:
                if field == 'Evaluation':
                    output += field + ':\n'
                    for act, evaluation in step_context[field].items():
                        output += act + ': ' + evaluation + '\n'
                else:
                    output += field + ': '
                    output += step_context[field] + '\n'
            else:
                output += field + ':'
                return output
        return output + '\n'

    def to_string(self, log=False):
        if not log:
            output = self.prompt
            step_context = self.context[-1]
            output += self._step_to_string(step_context)
            return output
        else:
            output = ''
            for step_context in self.context:
                output += self._step_to_string(step_context, log)
            return output