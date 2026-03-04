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

def orthogonal_complement_basis(A):
    """
    Finds an orthonormal basis for the orthogonal complement of the column space of A.

    Args:
        A (np.ndarray): The input matrix (spanning vectors as columns).

    Returns:
        np.ndarray: A matrix where columns are the orthonormal basis vectors for the orthogonal complement.
    """
    # The orthogonal complement of the column space of A is the null space of A.T
    # We can use SVD on A.T to find its null space.
    U, S, Vt = np.linalg.svd(A.T)

    # Singular values less than a tolerance are considered zero.
    # The number of non-zero singular values is the rank of the matrix.
    # The last 'n - rank' columns of Vt.T (or rows of Vt) form the basis for the null space.
    # We choose a small tolerance value.
    tol = np.max(S) * 1e-10
    rank = np.sum(S > tol)
    
    # The basis vectors for the null space are the last vectors in Vt (rows)
    # The number of null space vectors is n - rank (where n is the number of columns of A.T, i.e., rows of A)
    null_space_basis = Vt[rank:].T
    
    return null_space_basis

def solve_orthogonal_env(base_env):
    gamma = base_env.gamma
    n_states = base_env.state_space.n
    n_actions = base_env.action_space.n
    n_sa = n_states * n_actions

    # 1. M1 Construction
    P1 = np.vstack([base_env.P[:,a,:] for a in range(n_actions)])
    E = np.vstack([np.eye(n_states) for _ in range(n_actions)])
    M1 = E - gamma * P1

    # 2. Get the Orthogonal Complement (Null space of M1.T)
    U_perp = orthogonal_complement_basis(M1)
    
    print(f"The shape of U_perp is: {U_perp.shape}")
    # 3. Define Optimization
    P2 = cp.Variable((n_sa, n_states), nonneg=True)
    M2 = E - gamma * P2
    
    # 4. Objective: MAXIMIZE projection onto the complement 
    # Or MINIMIZE projection onto the existing basis U_1
    # Let's stick to minimizing alignment with the current basis:
    
    objective = cp.Minimize(cp.sum_squares(U_perp.T @ M2))
    # objective = cp.Minimize(cp.norm(U1.T @ M2, "fro"))
    
    # 5. Constraints
    constraints = [cp.sum(P2, axis=1) == np.ones(n_sa)]
    
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
    P_hat, M_hat, intersection_dim = solve_orthogonal_env(base_env)
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


def evaluate_objective_value(M , M_hat):
    # build orthonormal basis for the column space of M
    U = orthogonal_complement_basis(M)

    objective_value = np.linalg.norm(U.T @ M_hat, "fro")
    print(f"The objective value is: {objective_value}")


# Example usage

if __name__ == "__main__":

    # n_states = 3
    # n_actions = 2
    # n_sa = n_states * n_actions
    # E = np.vstack([np.eye(n_states) for _ in range(n_actions)])
    
    # print(E)
    

    base_env = RandomMDP(n_states=5, n_actions=2, n_demo=10, n_test=10, rad_demo=0.5, rad_test=0.75)
    

    debug_P_env = copy.deepcopy(base_env.P)
    debug_P = np.vstack([debug_P_env[:,a,:] for a in range(base_env.action_space.n)])
    
    new_P = np.zeros((base_env.state_space.n, base_env.action_space.n, base_env.state_space.n))
    for a in range(base_env.action_space.n):
        new_P[:, a, :] = np.eye(base_env.state_space.n)
    base_env.P = new_P

    
    original_P = np.vstack([base_env.P[:,a,:] for a in range(base_env.action_space.n)])
    E = np.vstack([np.eye(base_env.state_space.n) for _ in range(base_env.action_space.n)])


    orth_env, intersection_dim, P_hat, M_hat = build_orthogonal_env(base_env)

    



    print(np.round(P_hat, 2))
    M = E - base_env.gamma * original_P
    M_hat = E - base_env.gamma * P_hat
    test = np.hstack([M, M_hat])
    print(f"The optimal value is:")
    evaluate_objective_value(M, M_hat)

    print(f" With suboptimal, is:")
    evaluate_objective_value(M, E - base_env.gamma * debug_P)
    print(np.round(test, 2))
    print(f"The rank of the combined matrix is: {np.linalg.matrix_rank(test)}")
    print(f"The rank of debugging is: {np.linalg.matrix_rank(np.hstack([M, E - base_env.gamma * debug_P]))}")

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
    
        P1_check = np.zeros((n_sa, n_states))
        for s in range(n_states):
            for a in range(n_actions):
                P1_check[s * n_actions + a, :] = base_env.P[s, a, :]
        print(f"Original M1 Rank: {np.linalg.matrix_rank(E - base_env.gamma * P1_check)}")
        print(f"Intersection Dimension: {intersection_dim}")
    else:
        print("Orthogonal env construction failed.")