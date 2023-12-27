# waiting for importing from env:
# action_is_available()
# effect()



import math
import queue
import copy



def distance(sa, sb):
    ret = 0
    for i in range(0,64):
        ret += (sa[i]-sb[i])*(sa[i]-sb[i])
    
    return math.sqrt(ret)

def effect():
    return 0
    

# action checking function, need to import from environment
def action_is_available(action, state):
    return True




def precondition(action):
    g1 = int(action[5])
    g2 = int(action[10])
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
        string = "pick_" + str(i) + "_on_" + str(j)
        actions.append(string)
#['pick_1_on_2', 'pick_1_on_3', ..., 'pick_2_on_1',...,'pick_8_on_7']
        

# get new state s_new = (state_new, action_history[])
def state_update(action, s_cur):
    new = [0] * 64
    for index in range(0,64):
        if effect(action, index) == 1:  # positive effect
            new[index] = pi_g(action, s_cur[0]) +(1-pi_g(action, s_cur[0])*s_cur[0][index])
        elif effect(action, index) == -1: # negative effect
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
        if len(states >= 3):
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
                    s_new = state_update(action, s_cur)
                    states.append(s_new)


    states = sorted(states, cmp=lambda x,y:cmp(x[0]-distance(x[0],sg),y[0]-distance(y[0],sg)))
    Pi = states[0][1]


    return Pi


        


