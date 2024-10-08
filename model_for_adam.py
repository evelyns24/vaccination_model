import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import random

class Status(Enum):
    DEAD = -1
    NEVER_INFECTED = 0
    RECOVERED = 1
    INFECTED = 2

class Population:
    def __init__(self, N, init_infections, max_connectivity, max_severity, vaccine_immunization, partial_immunization, seed=0):
        self.N = N
        self.init_infections = init_infections
        self.max_connectivity = max_connectivity
        self.max_severity = max_severity
        self.vaccine_immunization = vaccine_immunization
        self.partial_immunization = partial_immunization
        self.seed = seed
        self.population = self.init_population()

    def init_population(self):
        Person = [('connectivity', 'i'),
                  ('severity', 'f'),
                  ('infection_period', 'i'),
                  ('status', 'i')]
        population = np.zeros(self.N, dtype=Person)

        np.random.seed(self.seed)
        population['connectivity'] = np.random.randint(0, self.max_connectivity + 1, size=self.N)

        severity_values = np.random.random(size=self.N)
        population['severity'] = np.select(
            [
                severity_values >= .925,
                severity_values >= .914,
                severity_values >= .084
            ],
            [
                100,
                80,
                2
            ],
            default=0
        )

        population['infection_period'] = np.random.randint(8, 14, size=self.N)
        population['status'] = Status.NEVER_INFECTED.value

        population['status'][np.random.choice(self.N, self.init_infections, replace=False)] = Status.INFECTED.value

        return population

    def evolve_state(self, C, death_threshold):
        infected_population_mask = (self.population['status'] == Status.INFECTED.value)
        infected_population = self.population[infected_population_mask]
        uninfected_population_mask = (self.population['status'] == Status.NEVER_INFECTED.value) | (
                    self.population['status'] == Status.RECOVERED.value)
        uninfected_population = self.population[uninfected_population_mask]

        infected_edges = np.sum(infected_population['connectivity'])
        total_edges = np.sum(self.population['connectivity'])
        infection_rate = C * (infected_edges / total_edges)

        prob_infection = infection_rate * uninfected_population['connectivity']
        random = np.random.random(size=uninfected_population.size)
        should_infect_mask = random < prob_infection
        uninfected_indices = np.where(uninfected_population_mask)[0]
        newly_infected_indices = uninfected_indices[should_infect_mask]
        self.population['status'][newly_infected_indices] = Status.INFECTED.value

        dead_population_indices = np.where((self.population['status'] == Status.INFECTED.value) &
                                           (self.population['infection_period'] == 0) &
                                           (self.population['severity'] == 100))[0]
        self.population['status'][dead_population_indices] = -1

        recovered_population_indices = np.where((self.population['status'] == Status.INFECTED.value) &
                                                (self.population['infection_period'] == 0) &
                                                (self.population['severity'] <= 100))[0]
        self.population['status'][recovered_population_indices] = Status.RECOVERED.value
        self.population['infection_period'][recovered_population_indices] = np.random.randint(4, 7)
        self.population['severity'][recovered_population_indices] *= self.partial_immunization

        remain_infected_population_indices = np.where((self.population['status'] == Status.INFECTED.value) &
                                                      (self.population['infection_period'] != 0))[0]
        self.population['infection_period'][remain_infected_population_indices] -= 1

    def compute_score(self, w_c):
        score = (self.population['connectivity'] * w_c / self.max_connectivity) + (self.population['severity'] * (1 - w_c) / self.max_severity)
        score[self.population['status'] != Status.NEVER_INFECTED.value] = 0
        return score

    def vaccinate(self, who, how_many):
        if who == 'mixed_score':
            w_c = 0.3
            scores = self.compute_score(w_c)
            vaccinate_population_indices = np.argsort(scores)[::-1][:how_many]
        else:
            vaccinate_population_indices = np.argsort(self.population[who])[::-1][:how_many]

        self.population['severity'][vaccinate_population_indices] *= self.vaccine_immunization

    def compute_cost(self):
        was_infected = (self.population['status'] == Status.DEAD.value) | (self.population['status'] == Status.RECOVERED.value) | (
                    self.population['status'] == Status.INFECTED.value)

        low_severity = (self.population['severity'] == 0) & was_infected
        medium_severity = (self.population['severity'] == 2) & was_infected
        high_severity = (self.population['severity'] == 80) & was_infected
        death = (self.population['severity'] == 100) & was_infected

        low_cost = np.sum(self.population['severity'][low_severity])
        medium_cost = np.sum(self.population['severity'][medium_severity])
        high_cost = np.sum(self.population['severity'][high_severity])
        death_cost = np.sum(self.population['severity'][death])

        total_cost = low_cost + medium_cost + high_cost + death_cost

        return {
            'low_severity': low_cost,
            'medium_severity': medium_cost,
            'high_severity': high_cost,
            'death': death_cost,
            'total': total_cost
        }

    def people_breakdown(self):
        never_sick = self.population['status'] == Status.NEVER_INFECTED.value
        asymptomatic = (self.population['status'] == Status.RECOVERED.value) & (self.population['severity'] == 0)
        low_severity = (self.population['status'] == Status.RECOVERED.value) & (self.population['severity'] == 2)
        high_severity = (self.population['status'] == Status.RECOVERED.value) & (self.population['severity'] == 80)
        fatal = self.population['status'] == Status.DEAD.value

        breakdown = {
            'never_sick': np.sum(never_sick),
            'asymptomatic': np.sum(asymptomatic),
            'low_severity': np.sum(low_severity),
            'high_severity': np.sum(high_severity),
            'fatal': np.sum(fatal)
        }

        return breakdown

    def run_simulation(self, timesteps, C, death_threshold):
        for _ in range(timesteps):
            self.evolve_state(C, death_threshold)

        cost_breakdown = self.compute_cost()
        ppl_breakdown = self.people_breakdown()
        return cost_breakdown, ppl_breakdown

    def to_graph(self):
        breakdown = self.people_breakdown()
        categories = ['Never Sick', 'Asymptomatic', 'Minimal', 'Severe', 'Fatal']
        counts = [breakdown['never_sick'], breakdown['asymptomatic'], breakdown['low_severity'],
                  breakdown['high_severity'], breakdown['fatal']]

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax.bar(categories, counts, color=bar_colors)

        ax.set_ylabel('Number of People')
        ax.set_title('Population Outcomes After Simulation')

        plt.tight_layout()
        plt.show()



# Test cases
N = 107000
init_infections = 10
max_connectivity = 50
max_severity = 100
vaccine_immunization = 0.1
partial_immunization = 0.1
timesteps = 120
C = 0.0375
death_threshold = 8
months = random_number = random.randint(3, 12)
cost_results = {}
people_results = {}



# Scenario 1: No policy (no vaccines)
print("Scenario 1: No policy (no vaccines)")
total_cost = 0
for _ in range(months):
    pop = Population(N, init_infections, max_connectivity, max_severity, vaccine_immunization, partial_immunization, seed=0)
    cost_breakdown, _ = pop.run_simulation(30, C, death_threshold)
    total_cost += cost_breakdown['total']
print(f"Cost for control: {total_cost}")

# Scenario 2: Exactly 3000 vaccines per month
print("\nScenario 2: Exactly 3000 vaccines per month")
for policy in ['connectivity', 'severity', 'mixed_score']:
    total_cost = 0
    for _ in range(months):
        vaccine = 3000
        pop = Population(N, init_infections, max_connectivity, max_severity, vaccine_immunization, partial_immunization, seed=0)
        pop.vaccinate(policy, vaccine)
        cost_breakdown, _ = pop.run_simulation(30, C, death_threshold)
        total_cost += cost_breakdown['total']
    print(f"Cost for {policy} policy: {total_cost}")

# Scenario 3: 2000 - 4000 vaccines per month
print("\nScenario 3: 2000 - 4000 vaccines per month")
for policy in ['connectivity', 'severity', 'mixed_score']:
    total_cost = 0
    for _ in range(months):
        vaccine = np.random.randint(2000, 4001)
        pop = Population(N, init_infections, max_connectivity, max_severity, vaccine_immunization, partial_immunization, seed=0)
        pop.vaccinate(policy, vaccine)
        cost_breakdown, _ = pop.run_simulation(30, C, death_threshold)
        total_cost += cost_breakdown['total']
    print(f"Cost for {policy} policy: {total_cost}")


def plot_policy_outcomes(policy_results):
    # Define the specific categories to plot
    categories = ['Never Sick', 'Asymptomatic', 'Fatal']
    bar_width = 0.25
    index = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # Colors for each policy

    # Loop through each policy's breakdown to create the bars
    for i, (policy, breakdown) in enumerate(policy_results.items()):
        counts = [breakdown['never_sick'], breakdown['asymptomatic'], breakdown['fatal']]
        ax.bar(index + i * bar_width, counts, bar_width, label=policy, color=colors[i])

    ax.set_xlabel('Outcomes')
    ax.set_ylabel('Number of People')
    ax.set_title('Final Outcomes for Each Policy (Selected Categories)')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(categories)
    ax.legend(title='Vaccination Policy')

    plt.tight_layout()
    plt.show()


policy_results = {}

# Simulate for each policy
for policy in ['connectivity', 'severity', 'mixed_score']:
    pop = Population(N, init_infections, max_connectivity, max_severity, vaccine_immunization, partial_immunization, seed=0)
    vaccine = 3000
    pop.vaccinate(policy, vaccine)
    _, ppl_breakdown = pop.run_simulation(30, C, death_threshold)
    policy_results[policy] = ppl_breakdown

# Plot the outcomes for each policy
plot_policy_outcomes(policy_results)