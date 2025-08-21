import cvxpy as cp
import numpy as np
from services.uncertainty import UncertaintySpec, theta_half_diag

def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value



def dot_np_expr(vec, expr_vec):
    # vec: 1-D numpy array; expr_vec: cvxpy Variable or Expression (shape (n,) or (n,1))
    return cp.sum(cp.multiply(vec, expr_vec))



def solve_mvo_core(mu: np.ndarray,
                   Q: np.ndarray,
                   *,
                   target_return: float,
                   allow_short: bool = False,
                   uncertainty: UncertaintySpec | None = None) -> np.ndarray:
    """
    Unified solver:
      - If `uncertainty is None` -> nominal constraint: mu^T x >= R
      - If `uncertainty.kind == "box"` -> mu^T x - δ^T|x| >= R
      - If `uncertainty.kind == "ellipsoid"` (Phase 2) -> mu^T x - ε ||Θ^{1/2} x||_2 >= R
    """
    # solve imcompatible inputs
    mu = np.asarray(mu).reshape(-1)            # make 1-D
    Q  = np.asarray(Q)
    if Q.ndim == 1:                            
        Q = np.diag(Q)

    if np.ndim(target_return) != 0:
        target_return = float(np.asarray(target_return).reshape(()))


    n = len(mu)
    x = cp.Variable(n)

    constraints = [cp.sum(x) == 1]
    if not allow_short:
        constraints += [x >= 0]

    if uncertainty is None:
        constraints += [ dot_np_expr(mu, x) >= target_return ]


    elif uncertainty.kind == "box":
        # δ_i = ε * σ_i / sqrt(T)
        theta = theta_half_diag(Q, uncertainty.T)         # σ_i / √T
        delta = uncertainty.epsilon * theta               # length-n vector

        if allow_short:
            # model |x| with auxiliary var y
            y = cp.Variable(n)
            constraints += [y >= x, y >= -x]
            constraints += [ dot_np_expr(mu, x) - dot_np_expr(delta, y) >= target_return ]
        else:
            # if x >= 0, |x| = x (no extra vars needed)
            constraints += [ dot_np_expr(mu, x) - dot_np_expr(delta, x) >= target_return ]

    elif uncertainty.kind == "ellipsoid":
        # reserved for Phase 2 (kept here so names don’t change later)
        theta_half = np.diag(theta_half_diag(Q, uncertainty.T))
        constraints += [ dot_np_expr(mu, x) - uncertainty.epsilon * cp.norm(theta_half @ x, 2) >= target_return ]
    else:
        raise ValueError(f"Unknown uncertainty kind: {uncertainty.kind}")

    objective = cp.Minimize(cp.quad_form(x, Q))  # min variance at target return R
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    return np.asarray(x.value).ravel()

def robust_mvo_box(mu: np.ndarray,
                   Q: np.ndarray,
                   *,
                   target_return: float,
                   T: int,
                   epsilon: float,
                   allow_short: bool = False) -> np.ndarray:
    """
    Convenience wrapper for Phase 1 (Box): δ_i = ε * σ_i / sqrt(T)
    """
    spec = UncertaintySpec(kind="box", epsilon=epsilon, T=T)
    return solve_mvo_core(mu, Q,
                          target_return=target_return,
                          allow_short=allow_short,
                          uncertainty=spec)