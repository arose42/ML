import numpy as np
import gym
from gym import wrappers
import time
import sys
import random
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration, QLearning, PolicyIteration
import mdptoolbox.example


def frozen_lake():
    env = gym.make('FrozenLake8x8-v0')
    env = env.unwrapped
    env_desc = env.desc
    time_arr = []
    gamma_arr = []
    iters_arr = []
    scores_l = []

    problem_rewards = []
    for eps in [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]:
        # start = time.time()
        rewards, iterations = qlearner_2(env, 0.99, eps)
        problem_rewards.append(np.mean(rewards))

    plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.95], problem_rewards, label='Q-Learning')
    plt.xlabel('Epsilon')
    plt.title('Frozen lake - Q-Learning - Rewards/Epsilon')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.legend()
    plt.show()

    for i in range(0, 10):
        start = time.time()
        policy, value_func, k, y = policy_iteration(env, gamma=(i+0.5)/10)
        time_arr.append(time.time() - start)
        gamma_arr.append((i + 0.5) / 10)
        scores_l.append(np.mean(value_func))
        iters_arr.append(k)
        #plot_visualization('Frozen Lake Policy Map - Gamma - '  + str((i+0.5)/10), policy.reshape(4,4), env_desc)


        #print("Gamma", (i+0.5)/10)
        #print(policy.reshape(4,4))

    time_arr_v = []
    gamma_arr_v = []
    iters_arr_v = []
    scores_l_v = []

    for i in range(0, 10):
        start = time.time()
        val_func, iteration = value_iteration(env, gamma=(i+0.5)/10)
        time_arr_v.append(time.time() - start)
        gamma_arr_v.append((i + 0.5) / 10)
        scores_l_v.append(np.mean(val_func))
        iters_arr_v.append(iteration)
        #print("Gamma", (i + 0.5) / 10)
        #print(val_func.reshape(4, 4))

    time_arr_q = []
    gamma_arr_q = []
    iters_arr_q = []
    scores_l_q = []



    for i in range(0, 10):
        start = time.time()
        rewards, iterations = qlearner(env, (i+0.5)/10)
        time_arr_q.append(time.time() - start)
        gamma_arr_q.append((i + 0.5) / 10)
        scores_l_q.append(np.mean(rewards))
        iters_arr_q.append(iterations)


    plt.plot(gamma_arr, time_arr, label='Policy Iteration')
    plt.plot(gamma_arr_v, time_arr_v, label='Value Iteration')
    plt.plot(gamma_arr_q, time_arr_q, label='Q-Learning')
    plt.xlabel('Gammas')
    plt.title('Frozen lake - Computation Time - Policy Iteration vs Value Iteration vs Q-Learning')
    plt.ylabel('Computation Time')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(gamma_arr, scores_l, label='Policy Iteration')
    plt.plot(gamma_arr_v, scores_l_v, label='Value Iteration')
    plt.plot(gamma_arr_q, scores_l_q, label='Q-Learning')
    plt.xlabel('Gammas')
    plt.title('Frozen Lake - Average Rewards - Policy Iteration vs Value Iteration vs Q-Learning')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(gamma_arr, iters_arr, label="Policy Iteration")
    plt.plot(gamma_arr_v, iters_arr_v, label="Value Iteration")
    #plt.plot(gamma_arr_q, iters_arr_q, label="Q-Learning")
    plt.xlabel('Gammas')
    plt.title('Iterations to Converge - Policy Iteration vs Value Iteration')
    plt.ylabel('Iterations')
    plt.grid()
    plt.legend()
    plt.show()

def forest_experiment():
    P, R = mdptoolbox.example.forest(S=1250, r1=500, r2=250)
    value = []
    policy = []
    iters = []
    time_ = []
    gamma = []

    rewards_p = []
    rewards_v = []
    time_p = []
    time_v = []
    iters_p = []
    iters_v = []
    rewards_q = []
    time_q = []
    iters_q = []

    mean_discrep = []

    env2 = gym.make('FrozenLake-v0')

    q_table = []
    value_q = []
    policy_q = []
    iters_q = []
    time_q_arr = []
    gamma_q = []
    q_vals = []
    q_rewards = []
    mean_discrep = []

    for i in range(0, 10):
        start = time.time()
        q_policy = mdptoolbox.mdp.QLearning(P, R, 0.8)
        time_q = time.time() - start
        q_policy.run()
        q_rewards.append(np.mean(q_policy.V))
        value_q.append(np.mean(q_policy.V))
        policy_q.append(q_policy.policy)
        gamma_q.append((i + 0.5) / 10)
        q_vals.append(q_policy.Q)
        mean_discrep.append(q_policy.mean_discrepancy)
        # iters_q.append(q_policy.n_iters)
        time_q_arr.append(time_q)


    plt.plot(gamma_q, mean_discrep, label='Q-Learning')
    plt.xlabel('Gammas')
    plt.title('Q-Learning Mean Discrepancy')
    plt.ylabel('Mean Discrepancy')
    plt.grid()
    plt.show()

    for size in [1250]:

        P, R = mdptoolbox.example.forest(S=size)
        forest_policy_p = PolicyIteration(P, R, 0.99)
        forest_policy_v = ValueIteration(P, R, 0.99)
        forest_policy_q = QLearning(P, R, 0.1)
        forest_policy_p.run()
        forest_policy_v.run()
        forest_policy_q.run()
        rewards_p.append(np.mean(forest_policy_p.V))
        rewards_v.append(np.mean(forest_policy_v.V))
        rewards_q.append(np.mean(forest_policy_q.V))
        time_p.append(forest_policy_p.time)
        time_v.append(forest_policy_v.time)
        #time_q.append(forest_policy_q.time)
        iters_p.append(forest_policy_p.iter)
        iters_v.append(forest_policy_v.iter)
        #iters_q.append(forest_policy_q.iter)


    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], rewards_p, label='Policy Iteration')
    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], rewards_v, label='Value Iteration')
    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], rewards_q, label='Q-Learning')
    #plt.grid()
    #plt.xlabel('State Size')
    #plt.title('Forest Management - Rewards vs State Size')
    #plt.ylabel('Average Rewards')
    #plt.legend()
    #plt.show()

    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], time_p, label='Policy Iteration')
    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], time_v, label='Value Iteration')
    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], time_q, label='Q-Learning')
    #plt.grid()
    #plt.xlabel('State Size')
    #plt.title('Forest Management - Computation Time vs State Size')
    #plt.ylabel('Computation Time')
    #plt.legend()
    #plt.show()

    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], iters_p, label='Policy Iteration')
    #plt.plot([1250, 1500, 1750, 2000, 2250, 2500], iters_v, label='Value Iteration')
    #plt.grid()
    #plt.xlabel('State Size')
    #plt.title('Forest Management - Convergence vs State Size')
    #plt.ylabel('Iterations')
    #plt.legend()
    #plt.show()

    value_vi = []
    policy_vi = []
    iters_vi = []
    time_vi = []
    gamma_vi = []
    mean_discrep_p = []

    for i in range(0, 10):
        forest_policy = PolicyIteration(P, R, (i+0.5)/10)
        forest_policy.run()
        gamma.append((i+0.5)/10)
        plt.imshow(np.atleast_2d(forest_policy.policy))
        time_.append(forest_policy.time)
        policy.append(forest_policy.policy)
        iters.append(forest_policy.iter)
        value.append(np.mean(forest_policy.V))

    for i in range(0, 10):
        forest_policy = ValueIteration(P, R, (i+0.5)/10)
        forest_policy.run()
        gamma_vi.append((i+0.5)/10)
        time_vi.append(forest_policy.time)
        policy_vi.append(forest_policy.policy)
        iters_vi.append(forest_policy.iter)
        value_vi.append(np.mean(forest_policy.V))

    #P, R = mdptoolbox.example.forest(S=1250, p=0.1)
    value_q = []
    policy_q = []
    iters_q = []
    time_q_arr = []
    gamma_q = []
    q_vals = []
    q_rewards = []
    mean_discrep = []


    env2 = gym.make('FrozenLake-v0')

    q_table = []

    for i in range(0, 10):
        start = time.time()
        q_policy = mdptoolbox.mdp.QLearning(P,R, 0.1)
        time_q = time.time() - start
        q_policy.run()
        q_rewards.append(np.mean(q_policy.V))
        value_q.append(np.mean(q_policy.V))
        policy_q.append(q_policy.policy)
        gamma_q.append((i+0.5)/10)
        q_vals.append(q_policy.Q)
        mean_discrep.append(q_policy.mean_discrepancy)
        #iters_q.append(q_policy.n_iters)
        time_q_arr.append(time_q)

    plt.plot(gamma, time_, label='Policy Iteration')
    plt.plot(gamma_vi, time_vi, label='Value Iteration')
    plt.plot(gamma_q, time_q_arr, label='Q-Learning')
    plt.xlabel('Gammas')
    plt.title('Forest Management - Computation Time - Policy Iteration vs Value Iteration vs Q-Learning')
    plt.ylabel('Computation Time')
    plt.grid()
    plt.legend()
    plt.show()
    

    plt.plot(gamma, value, label='Policy Iteration')
    plt.plot(gamma_vi, value_vi, label='Value Iteration')
    plt.plot(gamma_q, q_rewards, label='Q-Learning')
    plt.xlabel('Gammas')
    plt.title('Average Rewards - Policy Iteration vs Value Iteration vs Q-Learning')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(gamma, iters, label="Policy Iteration")
    plt.plot(gamma_vi, iters_vi, label="Value Iteration")
    #plt.plot(gamma_q, iters_q, label="Q-Learning")
    plt.xlabel('Gammas')
    plt.title('Iterations to Converge - Policy Iteration vs Value Iteration')
    plt.ylabel('Iterations')
    plt.grid()
    plt.legend()
    plt.show()

#citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.
    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.
    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.
    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    value_func_old = np.random.rand(env.nS)
    value_func_new = np.zeros(env.nS)
    for iteration in range(max_iterations):
        delta=0
        for s in range(env.nS):
            vs=0
            actions=[policy[s]]
            #if len(actions)==1: actions=[actions]
            for a in actions:
                for possible_next_state in env.P[s][a]:
                    prob_action = possible_next_state[0]
                    cur_reward=possible_next_state[2]
                    future_reward=gamma*value_func_old[possible_next_state[1]]
                    vs+=prob_action*(cur_reward+future_reward)
                #if env.P[s][a][3]:break
            diff=abs(value_func_old[s]-vs)
            delta=max(delta,diff)
            value_func_new[s]=vs
        #delta=math.sqrt(delta)
        if delta<=tol: break
        value_func_old = value_func_new
    return value_func_new, iteration


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.
    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.
    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy=np.zeros(env.nS,dtype='int')
    for s in range(env.nS):
        maxvsa=-1
        maxa=-1
        for a in range(env.nA):
            vsa=0
            for possible_next_state in env.P[s][a]:
                prob_action = possible_next_state[0]
                cur_reward = possible_next_state[2]
                future_reward = gamma * value_function[possible_next_state[1]]
                vsa+=prob_action * (cur_reward + future_reward)
            if vsa>maxvsa:
                maxvsa=vsa
                maxa=a
        policy[s]=maxa

    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.
    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.
    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.
    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    stable=True
    for s in range(env.nS):
        old_action=policy[s]
        maxvsa=-1
        maxa=-1
        for a in range(env.nA):
            vsa=0
            for possible_next_state in env.P[s][a]:
                prob_action = possible_next_state[0]
                cur_reward = possible_next_state[2]
                future_reward = gamma * value_func[possible_next_state[1]]
                vsa+=prob_action * (cur_reward + future_reward)
            if vsa>maxvsa:
                maxvsa=vsa
                maxa=a
        if maxa!=old_action: stable=False
        policy[s]=maxa
    return stable, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.
    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.
    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
    You should use the improve_policy and evaluate_policy methods to
    implement this method.
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.
    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    stable=False
    iters=0
    eval_iters=0
    while not stable:
        value_func,iter=evaluate_policy(env,gamma,policy)
        eval_iters+=iter
        stable,policy=improve_policy(env,gamma,value_func,policy)
        iters+=1
    return policy, value_func, iters, eval_iters


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.
    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.
    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func_old = np.random.rand(env.nS)
    value_func_new = np.zeros(env.nS)
    for iteration in range(max_iterations):
        delta=0
        for s in range(env.nS):
            maxvsa = -1
            for a in range(env.nA):
                vsa=0
                for possible_next_state in env.P[s][a]:
                    prob_action = possible_next_state[0]
                    cur_reward=possible_next_state[2]
                    if value_func_new[possible_next_state[1]]==0:
                        future_reward=gamma*value_func_old[possible_next_state[1]]
                    else:
                        future_reward = gamma * value_func_new[possible_next_state[1]]
                    vsa+=prob_action*(cur_reward+future_reward)
                if vsa>maxvsa:
                    maxvsa=vsa
            #diff=math.pow((value_func_old[s]-maxvsa),2)
            diff=abs(value_func_old[s]-maxvsa)
            delta=max(delta,diff)
            value_func_new[s]=maxvsa
        #delta=math.sqrt(delta)
        if delta<=tol: break
        value_func_old = value_func_new

    return value_func_new, iteration

#adaption of - https://rubikscode.net/2019/06/24/introduction-to-q-learning-with-python-and-open-ai-gym/
#and https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
def qlearner_2(env, gamma, eps):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    optimal = [0] * env.observation_space.n
    #alpha = 1.0
    #gamma = 1.0
    #episodes = 15000
    #epsilon = 0

    eps = eps
    gamma = gamma
    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            env.render()
            j+=1
            iters.append(j)
            a = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))
            s1, r, d, _ = env.step(a)
            # Update Q-Table with new knowledge
            Q[state, a] = Q[state, a] + eps * (r + gamma * np.max(Q[s1, :]) - Q[state, a])
            rAll += r
            state = s1
            if d == True:
                break
        rewards.append(rAll)
        env.render()
    return rewards, iters

def qlearner(env, gamma):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iters = []
    optimal = [0] * env.observation_space.n
    #alpha = 1.0
    #gamma = 1.0
    #episodes = 15000
    #epsilon = 0

    eta = .628
    gamma = gamma
    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            env.render()
            j+=1
            iters.append(j)
            a = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))
            s1, r, d, _ = env.step(a)
            # Update Q-Table with new knowledge
            Q[state, a] = Q[state, a] + eta * (r + gamma * np.max(Q[s1, :]) - Q[state, a])
            rAll += r
            state = s1
            if d == True:
                break
        rewards.append(rAll)
        env.render()
    return rewards, iters

def plot_visualization(title, policy, desc):
    x = policy.shape[0]
    y = policy.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, y), ylim=(0, x))
    lake_directions = {
        0: '⬅',
        1: '⬇',
        2: '➜',
        3: '⬆'
    }
    lake_colors = {
        b'G':'gold',
        b'H':'black',
        b'F': 'blue',
        b'S':'green'
    }
    plt.title(title)

    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x,y], 1, 1)
            p.set_facecolor(lake_colors[desc[i, j]])
            ax.add_patch(p)
            direction = policy[i, j]
            text = ax.text(x+0.45, y+0.45, lake_directions[direction],
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim(0, policy.shape[1])
    plt.ylim(0, policy.shape[0])
    plt.savefig(title + str('.png'))
    plt.tight_layout()
    plt.close()
    return plt



if __name__ == "__main__":
    #frozen_lake()
    forest_experiment()




