############################################
# Title : Find K with iterative soluation in lqr system
# Author: Jebeom Chae
# Referench : Jaehyun Lim
# Date:   2024-07-11
###########################################
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

# system matrices
A = np.array([
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [-3.0,  2.0, -2.0,  1.0],
    [ 1.0, -1.0,  0.5, -1.0]
])
B = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.5]
])

# continuous-time lqr
Q = np.diag([1e2, 1e2, 1e0, 1e0])
R = 1e-2*np.eye(2)

n = A.shape[0]
m = B.shape[1]

max_iters = 1000
tolerance = 1e-9

### Iterative solution (continuous-time) ###
# Initialize K
K = np.zeros((m, n))

# Iterative Solution for find P & K
for _ in range(max_iters):
    A_BK = A - B @ K
    I = np.eye(n)
    X = np.kron(I, A_BK.T) + np.kron(A_BK.T, I)
    vecQ_KRK = (Q + K.T @ R @ K).flatten() 

    vecP = solve(-X, vecQ_KRK) # -Xvec(P) = vec(Q_KRK) 
    P = vecP.reshape((n, n))   # (n^2,1)vec to (n,n) matrix

    K_new = np.linalg.inv(R) @ B.T @ P

    if norm(K_new - K) < tolerance:
        K = K_new
        break

    K = K_new

### Using Python Library ###
P_lib = solve_continuous_are(A, B, Q, R)
K_lib = np.linalg.inv(R) @ B.T @ P

# initail state and reference
x0 = np.array([0.0, 0.0, 0.0, 0.0])
xref = np.array([0.5, 1.0, 0.0, 0.0])

# simulation
sol = solve_ivp(
    lambda t, x: A @ x + B @ K @ (xref - x),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    x0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)

# simulation using library
sol_lib = solve_ivp(
    lambda t, x: A @ x + B @ K_lib @ (xref - x),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    x0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)

######################################################################################

# discrete-time system
Ts = 0.05  # sampling time
Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)

### Iterative solution (discrete-time) ###
# Initial P Setting
Pd = Q.copy()

# Iterative Solution for find P
for _ in range(max_iters):
    Pd_next = Ad.T @ Pd @ Ad + Q - Ad.T @ Pd @ Bd @ np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad
    
    if norm(Pd_next - Pd) < tolerance:
        Pd = Pd_next
        break
    
    Pd = Pd_next

# Calculate optimal K gain
Kd = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

### Using Python Library ###
# discrete-time lqr
Pd_lib = solve_discrete_are(Ad, Bd, Q, R)
Kd_lib = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

# simulation
t = np.arange(0.0, 5.0, Ts)  # simulation time sequence
y = [x0.copy()]  # list of system state
for _ in t[1:]:
    y.append(Ad @ y[-1] + Bd @ Kd @ (xref - y[-1]))
y = np.array(y).T

# simulation using library
y_lib = [x0.copy()]  # List of system state
for _ in t[1:]:
    y_lib.append(Ad @ y_lib[-1] + Bd @ Kd_lib @ (xref - y_lib[-1]))
y_lib = np.array(y_lib).T

# Plotting
plt.figure("Continuous-Time LQR Comparison")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(sol.t, xref[k] * np.ones_like(sol.t), "k--", label=f"{label}ref")
    plt.plot(sol.t, sol.y[k, :], "r-", label="Iterative Solution")
    plt.plot(sol_lib.t, sol_lib.y[k, :], "b--", label="Library Solution")
    plt.legend()
    plt.ylabel(label)
plt.xlabel("t")
plt.tight_layout()
plt.figure("Discrete-Time LQR Comparison")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(t, xref[k] * np.ones_like(t), "k--", label=f"{label}ref")
    plt.plot(t, y[k, :], "r-", label="Iterative Solution")
    plt.plot(t, y_lib[k, :], "b--", label="Library Solution")
    plt.legend()
    plt.ylabel(label)
plt.xlabel("t")
plt.tight_layout()
plt.show()
