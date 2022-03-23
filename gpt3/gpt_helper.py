from jericho import *
import openai
from gpt3.utils.jericho_env_wrapper import JerichoEnvWrapper
from gpt3.utils.util import complete
from typing import Dict, Union
from tqdm import tqdm

class GPTHelper():
    """
    Initializes the GPT-Helper
    engine: Openai engine to use, Ex: text-davinci-001
    openai_api_key: Access key for the Openai API, Ex: sk-AAJJIOSSDIUx93kajsidja9
    config: Dictionary containing a features list, instructions, and a list of in-context examples
    args: Arguments including the presence_penalty and temperature
    """
    def __init__(
        self, 
        engine: str,
        config: dict,
        openai_api_key: str=None, 
        args: Dict[str, Union[str, int, float]]=None
    ):
        # Check that features list is either one-action or multi-action
        assert ('Best Action' in config['Features']) ^ ('Evaluation' in config['Features'])

        openai.api_key = openai_api_key
        self.engine = engine
        self.features_list = config['Features']
        self.instruction = config['Instruction']
        self.in_context_examples = config['Examples']
        self.presence_penalty = args.presence_penalty if (args and 'presence_penalty' in args) else 0.0
        self.temperature = args.temperature if (args and 'temperature' in args) else 0.7

    """
    Runs the GPT model for a certain number of steps
    env: FrotzEnvironment that supports valid actions, game score etc.
    max_steps: How many steps to run for. Output length can be less than max_steps if the gpt model dies. 
    
    Returns a list containining GPT3's actions, in the same format as the in context examples
    """
    def get_actions(
        self,
        env: FrotzEnv,
        max_steps: int,
        print_progress: bool=False
    ):
        env_wrapper = JerichoEnvWrapper(env, self.features_list, self.instruction, self.in_context_examples)
        output = []
        for step_cnt in tqdm(range(max_steps)) if print_progress else range(max_steps):
            if 'Explanation' in env_wrapper.features:
                explanation = complete(env_wrapper.to_string(), mode='explain', engine=self.engine, temperature=self.temperature, presence_penalty=self.presence_penalty)
                env_wrapper.step(explanation=explanation)
                done = False
            if 'Evaluation' in env_wrapper.features:
                action_evaluation = complete(env_wrapper.to_string(), mode='action-evaluation', engine=self.engine, temperature=self.temperature, presence_penalty=self.presence_penalty)
                done, output_step = env_wrapper.step(action_evaluation=action_evaluation)
                output.append(output_step)
            if 'Best Action' in env_wrapper.features:
                action = complete(env_wrapper.to_string(), mode='act', engine=self.engine, temperature=self.temperature, presence_penalty=self.presence_penalty)
                done, output_step = env_wrapper.step(action=action)
                output.append(output_step)
            if done:
                break
        return output

