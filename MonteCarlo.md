# Monte Carlo for Reinforcement Learning

## 1. What is Monte Carlo for RL?

- Any method that solves a problem by generating suitable random numbers, and observe that fraction of numbers obey some properties, is classified as Monte Carlo method.
- Monte Carlo is a method that does not have assume knowledge of environment/model, and are based on averaging complete returns.
- The MC agent works with environment -> gain experience -> then works with that environment instead.

## 2. Monte Carlo prediction
- Prediction is the problem of estimating value of states. The higher value the better -> Hence it is better to be more in that state.
- One way of estimate is: (All returns after visit to a state)/(Times of visit). As time of visit increases, average return will be close to true value of state. There are 2 ways to average return:
  - **First visit**: Average all retwards **after first visit** to state
  - **Every visit**: Average all rewards **after all visit** to state
    ### Example:
    Consider MDP with prob to non-terminal state $p$, and prob to terminal state $1-p$. Reward is +1 and $\gamma$ = 1. 
    Suppose we observe 1 episode that lasts 10 steps, with return of 10. 
    What are first-visit and every-visit value of nonterminal state?

    - First-visit estimator: $V(s) = 10$
    - Every-visit estimator: $V(s) = 10+9+8+7+6+5+4+3+2+1/10 = 5.5$
- Monte-Carlo does not **bootstrapping** - Means making use of estimates to make further estimates.

## 3. Monte Carlo control
- Problem: Estimating optimal policies in cases when model information is not available. We must estimate value of each action in state to be used to suggest a policy.
- However, there could be action-state pairs that may _never_ visited. 
  - Solution: *Assumption of exploring starts* - States that all state-action pairs have non-zero prob of being selected. This allows us to Generalized Policy Iteration (GPI). 

In PD, the formula for policy improvement is as below. However, this uses transition probability, which is not known beforehand in this model-free learning.
$$\pi'(s)\quad=\quad\arg\max_{a}\sum_{s'.r}p(s',r | s,a)\Big[r+\gamma v_{\pi}(s')\Big]$$

Policy improvement in this case is done by making policy greedy with respect to current value. We have action-value function, no model is needed to construct greedy policy
$$\pi(s)\doteq\arg\max_aq(s,a)$$

However, greedy action may lack of exploration. 2 solutions for this:

### 3.1. Monte Carlo with exploring starts
All state action pairs have non-zero probability of being the starting pair. This will ensure each episode will take agent to new states, hence, more exploration.

Initialize, for all $s\in\mathcal{S},a\in\mathcal{A}(s)$:
- $Q(s,a)\leftarrow$arbitrary
- $\pi(s)\leftarrow$arbitrary
- $Returns(s,a)\leftarrow$empty list

Repeat forever:

(a) Generate an episode using exploring starts and $\pi$ <br>
(b) For each pair $s,a$ appearing in the episode:<br>
- $R\leftarrow$return following the first occurrence of $s,a$
- Append $R$ to $Returns(s,a)$
- $Q(s,a)\leftarrow$average$(Returns(s,a))$ <br>

(c) For each $s$ in the episode:
$\pi(s)\leftarrow\arg\max_aQ(s,a)$

### 3.2. Monte Carlo with epsilon-soft
Use epsilon to randomly explore. 

## 4. GPI
- Where value function is repeatedly altered to more closely approximate the value function for current policy. 
- Policy evaluation is done by experiencing many episodes. 

## 5. On-policy and off-policy learning
- **On-policy**: Evaluate or improve policy that is used to make decisions. But actually, learn action values for a near-optimal policy that still explores (Not on optimal policy). 
- **Off-policy**: Use 2 policies. One is learned and becomes optimal policy (*target policy*). One is more exploratory, used to generate behavior (behavior policy).

### 5.1. Off-policy: Importance sampling.
- Goal: Technique for estimating expected values under one distribution given samples from another. 
- Usage: Apply to off-policy learning by weighting returns according to **importance-sampling ratio** -  relative probability of trajectories occurring under target and behavior policy.
- 2 types of importance sampling:
  - Ordinary importance sampling: IS with simple average.
  - Weighted importance sampling: IS with weighted average.

## 6. Model-based vs Model-free learning

- Dynamic Programming is used to solve problems where underlying model of environment is known beforehand. We had all state transition probabilities to continue.
- However, in real life situations, the transition probability is not known beforehand. Example: If train a bot to learn how to play chess, the environment may have $10^50$ states, with large number of possible actions. Hence the model of this environment is impossible to design.

## 7. Monte Carlo Reinforcement Learning
- MC learns directly from episodes of experience without any prior knowledge of MDP transitions. 
- Random component is return or rewards
- The value is mean return of all sample trajectories for each state.

## 8. Monte Carlo Policy Evaluation
Value function is expected return $$v_\pi(s) \doteq \mathbb{E}_{\pi}[G_t \mid S_t=s]$$

Estimate expected value by adding up samples, dividing by total number of samples $$\bar{V}_{\pi}(s)=\frac{1}{N}\sum_{i=1}^{N}G_{i,s}$$

with *i* is Episode index, and *s* is State index

For every episode we play, there's sequence of states and rewards. From these rewards, we can calculate the return by sum up all future rewards.

### 8.1. First-visit Monte Carlo
- Average returns only for first time *s* is visited in an episode
- Step-by-step:
  
1. Initialize policy, state-value function
2. Start by generating an episode, according to current policy
   1. Keep track of states encountered in that episode
3. Select a state in 2.1.
   1. Add to a list of return, which received after first occurence of this state
   2. Average over all returns
   3. Set value of state as the average above
4. Repeat step 3
5. Repeat 2-4 until satisfied

### 9.1. Every-visit Monte Carlo
Average return for every time s is visited in an episode

1. Generate episode (follow current policy).
2. For each state visited:
  1. Find all returns from that state onwards (in current episode).
  2. Average those returns.
  3. Update state-value with the average.
3. (Optional) Improve policy based on updated state values.
4. Repeat 1-3.

### 10. Incremental Mean
It is more convenient to convert mean into incremental update, so mean can be updated with each episode. Learnt from multi-armed bandit problem

We update $v(s)$ incrementally after episodes. For each state $S_t$, return $G_t$:
$$\begin{aligned}
&N(S_{t}) \leftarrow N(S_t)+1 \\
&V(S_{t}) \leftarrow V(S_t)+\frac1{N(S_t)}\left(G_t-V(S_t)\right) 
\end{aligned}$$

For non-stationary problems. We can track running mean.
$$V(S_{t})\leftarrow V(S_{t})+\alpha (G_{t}-V(S_{t}))$$