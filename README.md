# core

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