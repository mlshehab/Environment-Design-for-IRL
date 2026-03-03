""" This concerns environment design for reward learning. """

import copy
from gym_minigrid.wrappers import *
from auxiliary.auxiliary import *
from auxiliary.mdp_solver import *
from maze_env import ConstructedMazeEnv
from env_design import *
from multi_env_birl import *
import time
from random import sample
import numpy as np
import cvxpy as cp
from scipy.linalg import orth

size = 40

from random_mdp import RandomMDP

def solve_orthogonal_env(base_env, gamma=0.95):
    n_states = base_env.state_space.n
    n_actions = base_env.action_space.n
    n_sa = n_states * n_actions

    # 1. Construct M1 = E - gamma * P1
    P = []
    for a in range(n_actions):
        P.append(base_env.P[:, a, :])
    P1 = np.concatenate(P, axis=0)
    E = np.zeros((n_sa, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            E[s * n_actions + a, s] = 1
    M1 = E - gamma * P1

    # 2. Get Orthonormal Basis U1 for col(M1) \ {1}
    full_basis = orth(M1) 
    ones_vec = np.ones((n_sa, 1)) / np.sqrt(n_sa)
    U1 = full_basis - ones_vec @ (ones_vec.T @ full_basis)
    U1 = orth(U1)

    # 3. Define Optimization Variables
    P2 = cp.Variable((n_sa, n_states), nonneg=True)
    M2 = E - gamma * P2
    
    # 4. Objective: Minimize || U1.T @ M2 ||_F^2
    objective = cp.Minimize(cp.sum_squares(U1.T @ M2))
    
    # 5. Constraints: Markovian / Row Stochastic + Sparsity
    zero_mask = (P1 == 0)
    constraints = [
        cp.sum(P2, axis=1) == 1,
        P2[zero_mask] == 0
    ]

    # 6. Solve the Quadratic Program
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if P2.value is None:
        print("Solver failed.")
        return None, None, None

    P_hat = P2.value
    row_sums = P_hat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero, though should not be needed
    P_hat = P_hat / row_sums
    M_hat = E - gamma * P_hat

    # 7. Dimension of Intersection (Grassmann Identity)
    tol = 1e-6
    r1 = np.linalg.matrix_rank(M1, tol=tol)
    r2 = np.linalg.matrix_rank(M_hat, tol=tol)
    r_joint = np.linalg.matrix_rank(np.hstack([M1, M_hat]), tol=tol)   
    intersection_dim = r1 + r2 - r_joint

    return P_hat, M_hat, intersection_dim

def build_orthogonal_env(base_env):
    """
    Takes in the base_env, deepcopies it, and overrides its transition dynamics
    with the newly found P2 (reshaped per action as in the original env).
    """
    P_hat, M_hat, intersection_dim = solve_orthogonal_env(base_env, gamma=base_env.gamma)
    if P_hat is None:
        print("Failed to create orthogonal P_hat.")
        return None

    n_states = base_env.state_space.n
    n_actions = base_env.action_space.n
    n_sa = n_states * n_actions

    # Reshape P_hat back to (n_states, n_actions, n_states)
    # Since in P1 and P_hat they are stacked as (a0, a1,...)
    new_P = np.zeros((n_states, n_actions, n_states))
    for a in range(n_actions):
        new_P[:, a, :] = P_hat[a*n_states:(a+1)*n_states, :]

    # Copy and set new transitions
    env_copy = copy.deepcopy(base_env)
    env_copy.P = new_P

    # If your env also caches other transition dynamics, clear or update them here as needed

    return env_copy, intersection_dim, P_hat, M_hat

# Example usage

if __name__ == "__main__":
    base_env = RandomMDP(n_states=40, n_actions=4, n_demo=10, n_test=10, rad_demo=0.5, rad_test=0.75)
    orth_env, intersection_dim, P_hat, M_hat = build_orthogonal_env(base_env)

    if orth_env is not None:
        print(f"New environment's transitions set. Intersection dimension: {intersection_dim}")
        # Optionally test/inspect transitions, ranks, etc.
        n_actions = orth_env.action_space.n
        n_states = orth_env.state_space.n
        n_sa = n_states * n_actions
        ones_vec = np.ones((n_sa, 1))
        # Compose E for check
        E = np.zeros((n_sa, n_states))
        for s in range(n_states):
            for a in range(n_actions):
                E[s * n_actions + a, s] = 1
        # Print checks
        print("Rank of M_hat: ", np.linalg.matrix_rank(M_hat))
        print("Rank of [M_hat, ones_vec]: ", np.linalg.matrix_rank(np.hstack([M_hat, ones_vec])))
    
        print(f"Original M1 Rank: {np.linalg.matrix_rank(E - base_env.gamma * np.concatenate([base_env.P[:,a,:] for a in range(n_actions)], axis=0))}")
        print(f"Intersection Dimension: {intersection_dim}")
    else:
        print("Orthogonal env construction failed.")