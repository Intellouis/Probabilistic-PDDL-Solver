# Towards a Unified Probabilistic PDDL Solver in the Block Stacking Domain

This is the repository for PKU CoRe 2023 Fall Course Project: Probabilistic PDDL Solver (Track of Abstract Reasoning).

## Environment

Following [NTP Vat Release](https://github.com/StanfordVL/NTP-vat-release/tree/master), we implement a lightweight environment of block stacking.

### Environment Introduction and Details
- Object space: We have 8 blocks, numbered 1 to 8. These blocks are totally same except for numbers.
- State space: We provide two versions of state description - both the precise locations of blocks (the coordinate space) and the PDDL-language descriptions. These coordinate-based state representations form the observation space of SGN, while those language-based descriptions form the observation space of LLM-based agents. We also provide the ground truth embeddings of each state during interactions and rollouts, which are used to determine whether a task is finished (whether the goal is reached).
- Embedding space: We use a 64-dimensional vector to represent a state, as details can be figured out in our paper.
- Action space: We use "put $i$ on $j$" to represent an action, where $i \in [1, 2, ..., 8]$ is the source block, and $j \in [1, 2, ..., 8]$ is the target place. In the domain of block stacking, we only allow agents placing a block right on the top of another block. This representation is consistent with the "grasp" and "place" in [Continuous Relaxation of Symbolic Planner for One-Shot Imitation Learning](https://arxiv.org/abs/1908.06769).
- Goal space: We also use a 64-dimensional vector to represent the goal state, in which 8 elements are 1 exactly. In each step of a trajectory, the goal vector is used to determine whether the goal has been reached (whether the task is finished).
- Tasks: We provide 2000 different tasks, which are obtained from [Neural Task Programming: Learning to Generalize Across Hierarchical Tasks](https://arxiv.org/abs/1710.01813). Each task has a unique goal, forming a diverse set of test cases.


## PDDL Solvers

We implement both the SGN-based PDDL Solver and the LLM-agent (baseline). The LLM-agent is based on GPT-4-11.6-preview.







## Definitions and standards
### Terms
* Planning problem: $(S_0,S_G,O)$
  * $O = \{o\}$
  * $o = (name(o), precondition(o),effect(o))$
  * precondition(o): The conditions that need to be satisfied in ordered to apply the operator.
* $Z(s)$ : The distribution over symbolic states, maps a symbolic states $s$ to its corresponding probability.
* $P_{Z(s)}(g)$ : The probability that ground atom $g$ is true given the distribution $Z(s)$ for all $g$
* atom: e.g. $on(A, B)$, $clear(C,D)$


### Standards
* Symbolic state vector $s$: A 64-dimentional vector, which represent:
  * 0-55 are $On(1, 2), On(1, 3), ... ,On(8, 7)$; 56-63 are $Clear(1),..., Clear(8)$


### Functions
* Continous planner $CP(L)$
  * input: a list of list $L$:$[s_0,s_1,...,s_t]$. Each vector $s_i$ denote: $[P_{Z(s_i)}(g_0),P_{Z(s_i)}(g_1),...,P_{Z(s_i)}(g_{63})]$
  * output: An action list $\Pi$.
* Pick and Place
* 
