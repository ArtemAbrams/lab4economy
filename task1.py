import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def population_growth(N):
    beta = 1
    gamma = 0.1
    p = 0.4
    return beta * (N**2 / (1 + N)) - gamma * N - p * N**2

critical_points = fsolve(population_growth, [-1, 0.1, 0.7])
critical_points = [point for point in critical_points if point > 0]

critical_points = np.round(critical_points, 5)
print("Релевантні критичні точки:", critical_points)

lower_critical = min(critical_points)
upper_critical = max(critical_points)

initial_conditions = {
    'менше половини нижньої межі': 0.5 * lower_critical * 0.5,
    'більше половини нижньої межі': 0.5 * lower_critical * 1.5,
    'нижня критична межа': lower_critical,
    'між нижньою і верхньою межею': (lower_critical + upper_critical) / 2,
    'верхня критична межа': upper_critical,
    'перевищує верхню межу': upper_critical * 1.5
}

print(f"Нижня критична точка: {lower_critical:.2f}")
print(f"Верхня критична точка: {upper_critical:.2f}")
print("Початкові умови для аналізу:", initial_conditions)

t_span = [-50, 100]
t_values = np.linspace(t_span[0], t_span[1], 300)

plt.figure(figsize=(10, 6))
for label, N0 in initial_conditions.items():
    t_values = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(lambda t, N: population_growth(N), t_span, [N0], t_eval=t_values)
    plt.plot(sol.t, sol.y[0], label=f'Початкове N0={N0:.2f}')

plt.title('Динаміка популяції за різними початковими умовами')
plt.xlabel('Час')
plt.ylabel('Чисельність популяції N')
plt.legend()
plt.grid(True)
plt.show()