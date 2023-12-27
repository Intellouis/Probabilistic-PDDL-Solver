# BFS algorithm planner
# Input: init_states and goal_states (64, )
# Output: List of Actions

import torch
import time
import itertools
from pddl_parser.planner import Planner

class PDDL_Generator():
    """
    Generate problem.pddl give symbolic states
    Configuration: 8 blocks, predicates: on(x, y) and clear(x)
    - states: 64-dimensional vector (discrete or continuous).
               Dimension 0~55 represent represent on(i, j)
               Dimension 56~63 clear(A~H).
    - filepath: where the generated file problem.pddl is saved
    - is_discrete: Clip states if False
    """
    def __init__(self):
        self.predicates = ['clear', 'on', 'onTable']
        self.objects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    def generate(self, init_states, goal_states, filepath, is_discrete=True):
        """
        Generate problem.pddl file given 64-dimension vectors
        """
        filename = filepath.split('/')[-1].split('.')[0]    # folder/problem.pddl
        init_atoms = "(:init"
        goal_atoms = "(:goal (and"
        problem = ""
        permutations = list(itertools.permutations(self.objects, 2))

        if not is_discrete:
            init_states = self.clip(init_states)
            goal_states = self.clip(goal_states)
        
        for i in range(56, 64):
            if init_states[i] == 1:
                init_atoms += f" ({self.predicates[0]} {self.objects[i-56]})"
                init_atoms += f" ({self.predicates[2]} {self.objects[i-56]})"
            if goal_states[i] == 1:
                # goal_atoms += f" ({self.predicates[0]} {self.objects[i-56]})"
                goal_atoms += ""

        for i in range(0, 56):
            if init_states[i] == 1:
                init_atoms += f" ({self.predicates[1]} {permutations[i][0]} {permutations[i][1]})"
            if goal_states[i] == 1:
                goal_atoms += f" ({self.predicates[1]} {permutations[i][0]} {permutations[i][1]})"

        problem += f"(define (problem {filename})\n\t(:domain blockstacking)\n\t(:objects A B C D E F G H)\n\t"   
        problem += init_atoms + ')\n\t' + goal_atoms +')))\n'
        # print(problem)
        
        with open(filepath, 'w') as f:
            f.write(problem)
            f.close()

    def clip(self, states):
        """
        Transform continuous states into discrete states
        """
        zeros = torch.zeros_like(states)
        ones = torch.ones_like(states)
        states = torch.where(states < 0.5, states, ones)
        states = torch.where(states == 1, states, zeros)
        return states


class SymbolicPlanner():
    """ 
    BFS planner with 64-dimensional symbolic states, output list of actions
    """
    def __init__(self):
        self.generator = PDDL_Generator()
        self.planner = Planner()

    def action_parsing(self, plan):
        ind = {'a':1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,}
        actions = []
        for act in plan:
            action = f"pick {ind[act.parameters[0]]} on {ind[act.parameters[1]]}"
            actions.append(action)
        return actions

    def plan(self, init_states, goal_states, domain, problem, if_discrete):
        """
        - init_states/goal_states: 64-dimensional vector
        - domain: path of domain file
        - problem: path of problem file (will be generated later)
        - if_discrete: True if the vector is discrete
        """
        self.generator.generate(init_states, goal_states, problem, if_discrete)
        planner = Planner()
        plan = planner.solve(domain, problem)
        if plan is not None:
            print('The plan was found.')
            actions = self.action_parsing(plan)
            # print(actions)
            return actions
        else:
            print('No plan was found.')
            return None
        
if __name__ == "__main__":
    """ Test """
    a = torch.zeros((64,))
    b = torch.zeros((64,))
    a[56:] = 1
    b[0] = 1
    b[8] = 1
    b[16] = 1
    b[56] = 1
    b[59:] = 1
    # g = PDDL_Generator()
    # g.generate(a, b, './problem1.pddl', False)
    # p = Planner()
    # plan = p.solve('./blockstacking.pddl', './problem1.pddl')
    sp = SymbolicPlanner()
    actions = sp.plan(a, b, './blockstacking.pddl', './problem1.pddl', False)
    print(actions)
    
    