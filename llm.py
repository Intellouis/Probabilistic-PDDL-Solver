import numpy as np
from TextWorld import TextWorld

from openai import OpenAI

api_key = "API_KEY"         # replace with your own API key
client = OpenAI(api_key=api_key)

class History:
    def __init__(self, init_state_description):
        self.description = f"{init_state_description} \n"

    def insert(self, action, state):
        self.description += f"Action: {action} \n"
        self.description += f"State: {state} \n"

    def get_history(self):
        return self.description
    

def llm(messages, **kwargs) -> str:
    # print(f"[Debug Info] messages = {messages}")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are an agent and you will complete a task."},
            {"role": "user", "content": f"{messages}"}
        ]
    )
    content = response.choices[0].message.content
    return content

def prompt(history, goal_description):
    pre_prompt = "You should complete a task of block stacking. \n"
    pre_prompt += "I will explain the PDDL language first: \n"
    pre_prompt += "There are 8 blocks, numbered from 1 to 8. \n"
    pre_prompt += "We use 2 predicates to describe a state: On (i, j) and Clear (i). \n"
    pre_prompt += "On (i, j) means block i is above block j (1 <= i, j <= 8). This means two cases: \n"
    pre_prompt += "1. Block i is just right on the top of block j (block is next to block j); \n"
    pre_prompt += "2. There exists some block k, such that block i is on the top of block k, and block k is on the top of block j. \n"
    pre_prompt += "Clear (i) means there are no other blocks on the top of block i (1 <= i <= 8). \n"
    pre_prompt += "Given a state described by several predicates, you should output the next action. \n"
    pre_prompt += "The action is in the form of 'put i on j', where 1 <= i, j <= 8. \n"
    pre_prompt += "The action means putting block i on the top of block j. \n"
    pre_prompt += "After execution, the next state will be notified to you. Please note that if you take an invalid action, the state will not change. \n"
    pre_prompt += "The history of actions and states will be shown to you: \n"
    pre_prompt += history
    pre_prompt += "\n The goal state is: \n"
    pre_prompt += goal_description

    post_prompt = "\n \n Please give your next action in exactly one sentence, in the format of 'put <i> on <j>': \n"
    post_prompt = "Please give your next action in exactly one sentence, in the format of 'put <i> on <j>': \n"
    post_prompt = "Please give your next action in exactly one sentence, in the format of 'put <i> on <j>': \n"
    return pre_prompt, post_prompt

def post_process(action):
    if "put " in action:
        idx = action.index("put ")
        return action[idx:idx+10]
    elif "Put " in action:
        idx = action.index("Put ")
        return "p" + action[idx+1:idx+10]
    else:
        return action


if __name__ == "__main__":
    env = TextWorld()
    tasks = np.load("./tasks.npy")
    for episode in range(2000):
        print(f"task {episode}")
        env.set_task(tasks[episode])
        obs, goal, done, reward, info = env.reset(next_task=tasks[episode])
        step = 0
        init_description = env.state_to_natual_language()
        print(f"State: {init_description}")
        print(f"Goal: {env.vector_to_natural_language(goal)}")
        history = History(init_description)
        while not done and step < 30:
            step += 1
            pre_prompt, post_prompt = prompt(history=history.get_history(), goal_description=env.vector_to_natural_language(goal))
            action = llm(pre_prompt + post_prompt)
            action = post_process(action)
            print(f"---------- step {step} ----------")
            print(f"Action: {action}")
            next_obs, goal, done, reward, info = env.step(action)
            current_state_description = env.state_to_natual_language()
            print(f"State: {current_state_description}")
            print(f"Goal: {env.vector_to_natural_language(goal)}")
            history.insert(action=action, state=current_state_description)
            obs = next_obs
        if done:
            print(f">>>>>>>>>>>>>>>>>>>>> done!")
    env.close()