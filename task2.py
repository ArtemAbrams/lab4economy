import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def population_model(t, N):
    return -0.001 * N**2 + 0.15 * N

t_span = (0, 30)
initial_conditions = [20, 180]
t_values = np.linspace(t_span[0], t_span[1], 300)

plt.figure(figsize=(10, 6))
for N0 in initial_conditions:
    sol = solve_ivp(population_model, t_span, [N0], t_eval=t_values)
    plt.plot(sol.t, sol.y[0], label=f'Початкове N0={N0}')

plt.title('Динаміка популяції алігаторів')
plt.xlabel('Час (місяці)')
plt.ylabel('Чисельність популяції N')
plt.legend()
plt.grid(True)
plt.show()

for N0 in initial_conditions:
    sol = solve_ivp(population_model, t_span, [N0], t_eval=[30])
    print(f'Чисельність популяції з початковим N0={N0} на момент t=30 місяців: {sol.y[0][0]:.2f}')