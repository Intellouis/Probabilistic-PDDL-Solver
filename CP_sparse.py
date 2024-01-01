"""
This continuous planner works under sparse coding condition.

Sparse coding condition:
In this condition, "on(A, B)" and "on(B, C)" != "on(A, C)". 
In other words, "on(A, C)" means "A is on C", not "A is above C".
Thus for any symbolic states, there are only 8 ground atoms that are ture.

The symbolic planner (baseline) also works under sparse coding condition.
"""


import math
import queue
import copy
import numpy as np
from queue import Queue
import time
# from TextWorld import TextWorld

NUM_BLOCKS = 8


def ij2k(i, j):
    """
    Mapping block index to state index.

    input:
    - i, j: index of block

    output:
    - k: corresponding index of ground atom in 64-dimensional vector
    """
    idx = -1
    for m in range(NUM_BLOCKS):
        for n in range(NUM_BLOCKS):
            if m == n:
                continue
            idx += 1
            if m == i and n == j:
                return idx
    assert False, "should not reach here"

def k2ij(k):
    """
    Mapping state index to obj index.

    input:
    - k: index of ground atom in 64-dimensional vector

    output:
    - i, j: corresponding index of block
    """
    idx = -1
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            if i == j:
                continue
            idx += 1
            if idx == k:
                return i, j
    assert False, "should not reach here"

def distance(sa, sb):
    """
    Calcute similarity (Euclidean distance) of current state and goal state.

    input:
    - sa, sb: 64-dimensional vector representing symbolic states

    output:
    - ret: Euclidean distance of both vectors
    """
    ret = 0
    for i in range(0,64):
        ret += (sa[i]-sb[i])*(sa[i]-sb[i])
    ret = math.sqrt(ret)
    
    return ret

def L1(sa, sb):
    """
    Calcute L1 distance of current state and goal state.

    input:
    - sa, sb: 64-dimensional vector representing symbolic states

    output:
    - ret: Euclidean distance of both vectors
    """
    ret = np.sum(np.abs(sa - sb))

    return ret

def effect(action, index, s_c=None, s_g=None):
    """
    Determine the type (positive, negative, irrelevant) of ground atom after action is executed.

    input:
    - action: action name (e.g., put 1 on 2)
    - index: index of ground atom in 64-dimensional vector

    -------------------------Is these required?-------------------------
    - s_c: 64-dimensional vector representing current symbolic states
    - s_g: 64-dimensional vector representing goal symbolic states
    --------------------------------------------------------------------

    output:
    - label: 1 means positive, -1 means negative, and 0 means irrelevant
    """
    obj_A = int(action[4]) - 1
    obj_B = int(action[9]) - 1
    # print(f"[Debug Info] s_cur: {s_cur}, sg: {sg}")
    
    # clear(x)
    if index in range(56, 64):
        x = index - 56 # 0 ~ 7
        if x == obj_B:
            return -1
        else:
            return 0

    # on(x, y)
    else:
        x, y = k2ij(index)
        if x == obj_A and y == obj_B:
            return 1
        # elif x == obj_A:
        #     return -1
        else:
            return 0

    assert False, "should not reach here"
    

# action checking function, need to import from environment
# def action_is_available(action, state):
#     return True

def precondition(action):
    """
    Determine the precondition of action

    input:
    - action: action name (e.g., put 1 on 2)
    
    output:
    - g1, g2: corresponding index of block
    """
    g1 = int(action[4]) - 1
    g2 = int(action[9]) - 1
    return g1, g2

def pi_g(action, s_c):
    """
    Compute Pi P_{z(s)}(g) representing the action's applicability.

    input:
    - action: action name (e.g., put 1 on 2)
    - s_c: 64-dimensional vector representing current symbolic states

    output:
    - p_a: Pi P_{z(s)}(g)
    """
    g1, g2 = precondition(action)
    p_a = s_c[56+g1] * s_c[56+g2]
    return p_a

actions = []
for i in range(1,9):
    for j in range(1,9):
        if i == j:
            continue
        string = "put " + str(i) + " on " + str(j)
        actions.append(string)
#['put 1 on 2', 'put 1 on 3', ..., 'put 2 on 1',...,'put 8 on 7']
        

def state_update(action, s_c):
    """
    Distribution of ground atoms shifts after applying action.

    input:
    - action: action name (e.g., put 1 on 2)
    - s_c: 64-dimensional vector representing current symbolic states

    output:
    - s_new: New distribution of ground atoms after applying action
    """
    s_new = np.zeros((64, ))
    p_a = pi_g(action, s_c)
    for index in range(0,64):
        label = effect(action, index)
        if label == 1:  # positive effect
            s_new[index] = p_a + (1 - p_a) * s_c[index]
            s_new[index] = max(0, s_new[index])
            s_new[index] = min(1, s_new[index])
        elif label == -1: # negative effect
            s_new[index] = s_c[index] - p_a
            s_new[index] = max(0, s_new[index])
            s_new[index] = min(1, s_new[index])
        else: # non effect
            s_new[index] = s_c[index]
            # s_new[index] = max(0, s_new[index])
            # s_new[index] = min(1, s_new[index])
        
    return s_new


# continous planner function for SGN
# input - two state vector, each one is a 64-dimentional vector from SGN, every element is a propability of the state
#         0-55 are On(1, 2), On(1, 3), ... ,On(8, 7); 56-63 are Clear(1),..., Clear(8)
# output - a list of actions like "pick_1_on_2"

def ranking(s_p, s_c, s_g, action, gamma=1.):
    """
    Ranking current states with distance and applicability.

    input:
    - s_p: 64-dimensional vector representing previous symbolic states (before applying action)
    - s_c: 64-dimensional vector representing current symbolic states (after applying action)
    - s_g: 64-dimensional vector representing goal symbolic states
    - action: action name (e.g., put 1 on 2)
    - gamma: scaling number

    output:
    - score: used for ranking
    """

    d = distance(s_c, s_g)
    p_a = pi_g(action, s_p)
    score = d - gamma * p_a

    return score

class State:
    def __init__(self, s_p, s_c, actions, d_c):
        self.s_p = s_p
        self.s_c = s_c
        self.actions = actions
        self.layer_num = len(actions)
        # self.d_p = d_p
        self.d_c = d_c

def clip(states):
    """
    Transform continuous vector into discrete vector via clipping.
    """
    states = np.where(states >= 0.5, 1., 0.)
    return states

def continous_planner(s_0, s_g, if_clip=False):
    """Continuous Planner"""

    if if_clip:
        s_0 = clip(s_0)
        s_g = clip(s_g)
        max_l = L1(s_0, s_g)
    else:
        max_l = L1(clip(s_0), clip(s_g))

    Pi = [] # action list to return
    max_d = distance(s_0, s_g)
    layer_num = int(max_l / 2)
    clip_num = 5 # max number of children
    # layer_num = 6 # layer number of search
    root = State(s_0, s_0, [], max_d) # (s_p, s_c, action_history[], d_c)
    # finished = False # searching finished
    # mmax = 1e9
    # threshold = max_d / layer_num

    queue = Queue()
    queue.put(root)
    start_time = time.time()
    while not queue.empty():
        
        # if queue.qsize() == clip_num ** layer_num:
        #     break

        # if mmax <= threshold:
        #     break

        state = queue.get()
        layer = state.layer_num
        if layer >= layer_num:
            break

        new_state_list = []
        for action in actions:
            # add action to action list
            s_new = state_update(action, state.s_c)
            d_c = distance(s_new, s_g)
            if d_c > max_d or d_c > state.d_c:
                continue
            plan = copy.deepcopy(state.actions)
            plan.append(action)
            state_new = State(s_p=state.s_c, s_c=s_new, actions=plan, d_c=d_c)
            new_state_list.append(state_new)
        # print(len(new_state_list))
        new_state_list = sorted(new_state_list, key=lambda x: ranking(s_p=x.s_p, s_c=x.s_c, s_g=s_g, action=x.actions[-1]))
        new_state_list = new_state_list[:clip_num]
        
        # min_dis = 1e9
        for new_state in new_state_list:
            queue.put(new_state)
            # min_dis = min(min_dis, new_state.d_c)

        # mmax = min(mmax, min_dis)


    t = time.time() - start_time
    print(f"Planning Time: {t}s") 

    length = queue.qsize()
    # print(length)
    state_list = [queue.get() for _ in range(length)]
    choose_state = sorted(state_list, key=lambda x: distance(x.s_c, s_g))[0]
    Pi = choose_state.actions
    print(f"Pi: {Pi}")

    return Pi

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
        while not done and step < 30:
            step += 1
            action = continous_planner(obs, goal)
            print(f"---------- step {step} ----------")
            print(f"Action: {action}")
            next_obs, goal, done, reward, info = env.step(action)
            current_state_description = env.state_to_natual_language()
            print(f"State: {current_state_description}")
            print(f"Goal: {env.vector_to_natural_language(goal)}")
            obs = next_obs
        if done:
            print(f">>>>>>>>>>>>>>>>>>>>> done!")
    env.close()