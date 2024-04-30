import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

a1, a2 = 0.7, 0.6
b12, b21 = 0.05, 0.05
c1, c2 = 0.1, 0.1

def competition_model(t, N):
    N1, N2 = N
    dN1_dt = a1 * N1 - b12 * N1 * N2 - c1 * N1**2
    dN2_dt = a2 * N2 - b21 * N1 * N2 - c2 * N2**2
    return [dN1_dt, dN2_dt]

def equilibrium(N):
    N1, N2 = N
    return [
        a1 * N1 - b12 * N1 * N2 - c1 * N1**2,
        a2 * N2 - b21 * N1 * N2 - c2 * N2**2
    ]

guesses = [[10, 5], [5, 10], [20, 20], [1, 1]]
equilibria = []

for guess in guesses:
    point = fsolve(equilibrium, guess)
    if not any(np.allclose(point, eq, atol=1e-4) for eq in equilibria):
        equilibria.append(point)

t_span = (0, 50)
t_values = np.linspace(t_span[0], t_span[1], 400)
sol = solve_ivp(competition_model, t_span, [10, 5], t_eval=t_values)

plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], label='Популяція N1')
plt.plot(sol.t, sol.y[1], label='Популяція N2')
plt.title('Динаміка популяцій')
plt.xlabel('Час')
plt.ylabel('Чисельність популяції')
plt.legend()
plt.grid(True)

plt.figure(figsize=(6, 6))
plt.plot(sol.y[0], sol.y[1], '-o', markersize=3, label='Фазова траєкторія')
for point in equilibria:
    plt.scatter(point[0], point[1], color='red', zorder=5)
plt.title('Фазовий портрет з рівноважними точками')
plt.xlabel('Популяція N1')
plt.ylabel('Популяція N2')
plt.legend()
plt.grid(True)
plt.show()

print("Рівноважні точки (N1, N2):", equilibria)

