# Asynchronous Methods For Deep Reinforcement Learning

## Contents

* [Paper](Paper.pdf)

## Summary

A conceptually simple and lightweight framework for Deep Reinforcement Learning that uses Asynchronous Gradient Descent for Optimization of Deep Neural Network Controllers.

The Asynchronous Variant of Actor-Critic surpasses the current State-of-the-art on Atari domain.

## Background

The standard reinforcement learning setting where an agents interacts with an environment ![e](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BE%7D) over a number of discrete time steps. At each time step t, the agent receives a state ![state](https://latex.codecogs.com/gif.latex?s_t) and selects an action ![action](https://latex.codecogs.com/gif.latex?a_t) from a set of possible actions ![actions](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D) according to the its policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi), where ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) is a mapping from states ![state](https://latex.codecogs.com/gif.latex?s_t) to actions ![action](https://latex.codecogs.com/gif.latex?a_t). In return the agent receives the next state ![state](https://latex.codecogs.com/gif.latex?s_%7Bt&plus;1%7D) and receives a scalar reward ![reward](https://latex.codecogs.com/gif.latex?r_t). 

The return ![return](https://latex.codecogs.com/gif.latex?R_t%20%3D%20%5Csum_%7Bk%3D0%7D%5E%7B%5Cinfty%7D%20%5Cgamma%5Ek%20r_%7Bt&plus;k%7D) is the total accumulated return from the time step t with a discount factor ![gamma](https://latex.codecogs.com/gif.latex?%5Cgamma%20%5Cin%20%280%2C1%5D). The goal of the agent is to maximize the expected return from each state ![state](https://latex.codecogs.com/gif.latex?s_t).

The action value ![Q](https://latex.codecogs.com/gif.latex?Q%5E%7B%5Cpi%7D%5Cleft%20%28%20s%2Ca%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D%5Cleft%20%5B%20R_t%7Cs_t%3Ds%2Ca%20%5Cright%20%5D) is the expected return for selecting an action a in the state s and following policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi). The optimal value function ![Q_optimal](https://latex.codecogs.com/gif.latex?Q%5E*%5Cleft%20%28%20s%2Ca%20%5Cright%20%29%20%3D%20%5Cmax_%7B%5Cpi%7D%20Q%5E%7B%5Cpi%7D%5Cleft%20%28%20s%2Ca%20%5Cright%20%29) gives the maximum action value for state s and action a achievable by any policy.

Similarly the value function of the state s under policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) is defined as ![value](https://latex.codecogs.com/gif.latex?V%5E%5Cpi%20%5Cleft%20%28s%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D%5Cleft%20%5B%20R_t%7Cs_t%3Ds%20%5Cright%20%5D) and is the expected return for following the policy ![pi](https://latex.codecogs.com/gif.latex?%5Cpi) from the state s.

In value-based model-free reinforcement learning methods, the action value function is represented using a function approximator, such as a neural network. Let ![Q](https://latex.codecogs.com/gif.latex?Q%5Cleft%20%28%20s%2Ca%3B%5Ctheta%20%5Cright%20%29) be an approximate action-value function with parameters ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta). The updates to ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) can be derived from a variety of reinforcement learning algorithms. The parameters of the action value function are learned iteratively minimizing a sequence of loss functions ![loss](https://latex.codecogs.com/gif.latex?L_i%5Cleft%20%28%20%5Ctheta_i%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D%5Cleft%20%28%20r%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%5Cprime%7D%20Q%5Cleft%20%28%20s%5E%5Cprime%2C%20a%5E%5Cprime%3B%20%5Ctheta_%7Bi-1%7D%20%5Cright%20%29%20-Q%5Cleft%20%28%20s%2Ca%3B%5Ctheta_i%20%5Cright%20%29%20%5Cright%20%29%5E2) where ![s'](https://latex.codecogs.com/gif.latex?s%5E%5Cprime) is encountered after state s.

In one-step methods is that obtaining a reward r only directly affects the value of the state action pair s, a that led to the reward. The values of other state action pairs are affected only indirectly through the updated value ![Q](https://latex.codecogs.com/gif.latex?Q%5Cleft%20%28%20s%2Ca%20%5Cright%20%290).