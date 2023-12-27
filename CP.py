# waiting for importing from env:
# action_is_available()
# effect()


import math
import queue
import copy
import numpy as np
from TextWorld import TextWorld

NUM_BLOCKS = 8

def ij2k(i, j):
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
    ret = 0
    for i in range(0,64):
        ret += (sa[i]-sb[i])*(sa[i]-sb[i])
    
    return math.sqrt(ret)

def effect(action, index, s_cur, sg):
    obj_A = int(action[4])
    obj_B = int(action[9])
    # print(f"[Debug Info] s_cur: {s_cur}, sg: {sg}")
    if s_cur[0][index] == sg[index]:
        return 0
    if index >= 0 and index <= 55:
        tgt_A, tgt_B = k2ij(index)
        tgt_A, tgt_B = tgt_A + 1, tgt_B + 1
        if obj_A != tgt_A:
            return 0
        if s_cur[0][index] == 0 and sg[index] == 1:
            if obj_B == tgt_B:
                return 1
            elif obj_B != tgt_B and s_cur[0][ij2k(obj_B-1, tgt_B-1)] == 1:        
                # obj_B is somehow above tgt_B, indirectly
                return 1
            else:
                return -1
        elif s_cur[0][index] == 1 and sg[index] == 0:
            if obj_B == tgt_B:
                return -1
            elif obj_B != tgt_B and s_cur[0][ij2k(obj_B-1, tgt_B-1)] == 1:        
                # obj_B is somehow above tgt_B, indirectly
                return -1
            else:
                return 1
    elif index >= 56 and index <= 63:
        tgt_C = index - 55
        if s_cur[0][index] == 0 and sg[index] == 1:
            if obj_B != tgt_C and s_cur[0][ij2k(obj_B-1, tgt_C-1)] == 1:
                # obj_B is somehow above tgt_C, indirectly
                return -1
            else:
                return 1
        elif s_cur[0][index] == 1 and sg[index] == 0:
            if obj_B != tgt_C and s_cur[0][ij2k(obj_B-1, tgt_C-1)] == 1:
                # obj_B is somehow above tgt_C, indirectly
                return 1
            else:
                return 0
    assert False, "should not reach here"
    

# action checking function, need to import from environment
def action_is_available(action, state):
    return True




def precondition(action):
    g1 = int(action[4])
    g2 = int(action[9])
    return g1, g2

def pi_g(action, s_cur): # get Pi P_{z(s)}(g)
    g1,g2 = precondition(action)
    p_a = s_cur[55+g1] * s_cur[55+g2]
    return p_a





#init actions list including all actions
actions = []
for i in range(1,9):
    for j in range(1,9):
        if i == j:
            continue
        string = "put " + str(i) + " on " + str(j)
        actions.append(string)
#['put_1_on_2', 'put_1_on_3', ..., 'put_2_on_1',...,'put_8_on_7']
        

# get new state s_new = (state_new, action_history[])
def state_update(action, s_cur, sg):
    new = [0] * 64
    for index in range(0,64):
        if effect(action, index, s_cur, sg) == 1:  # positive effect
            new[index] = pi_g(action, s_cur[0]) +(1-pi_g(action, s_cur[0])*s_cur[0][index])
        elif effect(action, index, s_cur, sg) == -1: # negative effect
            new[index] = s_cur[0][index] - pi_g(action, s_cur[0])
        else: # non effect
            new[index] = s_cur[0][index]
        
    history = copy.deepcopy(s_cur[1])
    history.append(action)
    return (new, history)





# continous planner function for SGN
# input - two state vector, each one is a 64-dimentional vector from SGN, every element is a propability of the state
#         0-55 are On(1, 2), On(1, 3), ... ,On(8, 7); 56-63 are Clear(1),..., Clear(8)
# output - a list of actions like "pick_1_on_2"
        
def continous_planner(s0, sg):

    Pi = [] # action list to return
    layer_num = 15 # layer number of search
    states = [] # states list of current layer
    states.append((s0,[])) # (s_cur, action_history[])
    finished = False # searching finished


    for i in range(0,layer_num):

        if finished:
            break

        # cutting
        states = sorted(states, key=lambda x: distance(x[0],sg))
        if len(states) >= 3:
            states = copy.deepcopy(states[:3])
        length = len(states)
        
        for k in range(0,length):

            s_cur = states.pop(k)

            if s_cur == sg:
                finished = True
                break

            action_list = [] 
            for action in actions:
                if action_is_available(action,s_cur[0]):
                    action_list.append(action)
                    p_a = pi_g(action, s_cur[0])
                    s_new = state_update(action, s_cur, sg)
                    states.append(s_new)


    states = sorted(states, key=lambda x: distance(x[0],sg))
    Pi = states[0][1]

    print(f"Pi: {Pi}")
    return Pi[0]


        
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
    

