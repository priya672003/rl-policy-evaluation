# POLICY EVALUATION

## AIM

To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States :
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.

### Actions :
The agent can take two actions:

R: Move right.
L: Move left.

### Transition Probabilities :
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards :
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation :

![image](https://github.com/priya672003/rl-policy-evaluation/assets/81132849/122d7291-0301-4d01-9eb0-cd84555dfd4e)



## POLICY EVALUATION FUNCTION

Formula :

![image](https://github.com/priya672003/rl-policy-evaluation/assets/81132849/35beeb8a-5d86-4742-b9cb-56664375e0fd)

## PROGRAM :

```python3

def policy_evaluation(pi, P, gamma=0.9, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V
```
## OUTPUT:
Policy 1:

![image](https://github.com/priya672003/rl-policy-evaluation/assets/81132849/12e6ba51-79d6-4d54-969d-feee0cd7efa8)

Policy 2:

![image](https://github.com/priya672003/rl-policy-evaluation/assets/81132849/4c905ec3-01ba-4230-a538-12faceaee607)

Comparison:


![image](https://github.com/priya672003/rl-policy-evaluation/assets/81132849/46249b0a-f222-4dc0-8d36-dea074515fe4)

Conclusion:


![image](https://github.com/priya672003/rl-policy-evaluation/assets/81132849/ef8baba7-2700-4234-af26-33fe55e938fd)

## RESULT:

Thus, a Python program is developed to evaluate the given policy.
