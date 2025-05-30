"""
Starter code for the problem "MPC feasibility".

Autonomous Systems Lab (ASL), Stanford University
"""

from itertools import product

import cvxpy as cvx

import matplotlib.pyplot as plt

import numpy as np

from scipy.linalg import solve_discrete_are

from tqdm.auto import tqdm


def do_mpc(
    x0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    N: int,
    rx: float,
    ru: float,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve the MPC problem starting at state `x0`."""
    n, m = Q.shape[0], R.shape[0]
    x_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # PART (a): YOUR CODE BELOW ###############################################
    # INSTRUCTIONS: Construct and solve the MPC problem using CVXPY.

    cost = 0.0
    constraints = [x_cvx[0,:]==x0]
    for k in range(0,N):
        constraints.append(x_cvx[k+1,:]==A@x_cvx[k,:]+B@u_cvx[k,:])
        constraints.append(cvx.norm(u_cvx[k,:],p=np.inf)<=ru)
        constraints.append(cvx.norm(x_cvx[k,:],p=np.inf)<=rx)
        cost+=cvx.quad_form(x_cvx[k,:],Q) + cvx.quad_form(u_cvx[k,:],R)
    constraints.append(cvx.quad_form(x_cvx[N,:],W)<=1)
    cost+=cvx.quad_form(x_cvx[N,:],P)


    # END PART (a) ############################################################

    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve(cvx.CLARABEL)
    x = x_cvx.value
    u = u_cvx.value
    status = prob.status

    return x, u, status

def generate_ellipsoid_points(M, num_points=100):
    """Generate points on a 2-D ellipsoid.

    The ellipsoid is described by the equation
    `{ x | x.T @ inv(M) @ x <= 1 }`,
    where `inv(M)` denotes the inverse of the matrix argument `M`.

    The returned array has shape (num_points, 2).
    """
    L = np.linalg.cholesky(M)
    θ = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack([np.cos(θ), np.sin(θ)])
    x = u @ L.T
    return x


n, m = 2, 1
A = np.array([[0.9, 0.6], [0.0, 0.8]])
B = np.array([[0.0], [1.0]])
Q = np.eye(n)
R = 1 * np.eye(m)
P = np.eye(n)
N = 4
T = 15
rx = 5.0
ru = 1.0

# Part (d) solve for M
M = cvx.Variable((n,n),symmetric=True)
constraints = [M>>0]
constraints+=[cvx.vstack([cvx.hstack([M,A@M]),cvx.hstack([M@A.T, M])])>>0]
constraints+=[M<<rx*rx*np.eye(n,n)]
cost = -cvx.log_det(M)
prob = cvx.Problem(cvx.Minimize(cost), constraints)
prob.solve(cvx.CLARABEL)
M = M.value
print(M)

# Plot:
xM = generate_ellipsoid_points(M)
xAM = xM@A.T
xr = generate_ellipsoid_points(rx*rx*np.eye(n,n))
plt.figure(figsize=(8, 6))
plt.plot(xM[:,0],xM[:,1],label='XT')
plt.plot(xAM[:,0],xAM[:,1],label='AXT')
plt.plot(xr[:,0],xr[:,1],label='X')
plt.legend()
plt.title('Ellipsoids')
plt.savefig("MPCEllipsoids.png", bbox_inches="tight")
plt.show()

# Part (e) Simulate
plt.figure(figsize=(8, 6))
plt.plot(xM[:,0],xM[:,1],label='XT')
plt.plot(xAM[:,0],xAM[:,1],label='AXT')
plt.plot(xr[:,0],xr[:,1],label='X')
x0 = np.array([0.0, -4.5])
x = np.copy(x0)
x_mpc = np.zeros((T, N + 1, n))
u_mpc = np.zeros((T, N, m))
for t in range(T):
    x_mpc[t], u_mpc[t], status = do_mpc(x, A, B, P, Q, R, N, rx, ru, np.linalg.inv(M))
    if status == "infeasible":
        x_mpc = x_mpc[:t]
        u_mpc = u_mpc[:t]
        break
    x = A @ x + B @ u_mpc[t, 0, :]
    plt.plot(x_mpc[t, :, 0], x_mpc[t, :, 1], "--*", color="k",label="Planned Trajectory")
plt.plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], "-o",label="Actual Trajectory")
# plt.plot(u_mpc[:, 0], "-o")
plt.legend()
plt.savefig("MPCTerminalIngredients.png", bbox_inches="tight")
plt.show()