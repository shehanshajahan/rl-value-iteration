# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## VALUE ITERATION ALGORITHM
```
Step 1: Set the value of each state to 0 (initial guess).
Step 2: Look at all the actions you can take from that state (like moving up, down, left, or right).
Step 3: Calculate the expected value of each action (i.e., how good that action is based on its possible results).
Step 4: Pick the action that gives the highest value and update the value of the state with that number.
Step 5: Keep updating the values for all states until the difference between the old and new values is very small.
Step 6: Once the values have stabilized, go through each state again and pick the action that leads to the highest value. This gives you the optimal action (policy) for each state.
```

## VALUE ITERATION FUNCTION
### Name: Shehan Shajahan
### Register Number: 212223240154
#  Frozen Lake environment
```py
envdesc  = ['SFFF','FHFH','FFFH', 'HFFG']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 14 
P = env.env.P
```
# Value iteration function
```py
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward+gamma*V[next_state]*(not done))
        if np.max(np.abs(V-np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi
```

## OUTPUT:
### Optimal policy
<img width="516" height="139" alt="image" src="https://github.com/user-attachments/assets/67dbf982-226c-4192-aec1-e7fac17596b5" />

### Optimal value function
<img width="520" height="103" alt="image" src="https://github.com/user-attachments/assets/fb1da7c0-8531-4a3f-9925-afcded3131c5" />

### Success rate for the optimal policy
<img width="666" height="28" alt="image" src="https://github.com/user-attachments/assets/ec626905-ded3-4136-857e-b5027267e789" />

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.



