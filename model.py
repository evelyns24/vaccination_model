import numpy as np

"""
# State variables:
connectivity: integer between 0 and 40 
– "How many people do I closely interact with in each timestep?"

case_severity: decimal between 0 and 5
- "If infected, how bad does it get for this person?"

infection_period: integer between 0 and 6
- "If I'm currently uninfected, how many days am I going to be infectious for?" -> randomly between 4 and 6 days.
- "If I am infected, how many more days am I infectious for?" -> decrements on each timestep.

is_infected: boolean
- True if the person is currently infected.


# State evolution:
Given state at time t, to calculate state at time t+1,

sick_edges = filter for members of the population who are sick, and sum their connectivity
all_edges = sum connectivity
ratio = sick_edges/all_edges

for all members m in the population, 
if (!is_infected) :
    p_infection = ratio * m.connectivity # probability of infection
    generate random number between 0 and 1. If num < p_infection,
    m.is_sick = true
else:
    if m.infection_period > 0:
        m.infection_period--
    else:
        m.is_infected = false

# Cost function        



"""

# define a structured data type to represent a person
# 5 fields: is_sick, severity, connectivity, days_infectious, status
person = [('is_sick','b'),""
          ('severity','i'),
          ('connectivity','i'),
          ('day_infectious','i'),
          ('status','U20')]

status_terms = ['never infected',
                'partially immunized',
                'severely crippled',
                'dead']

N=107000
population=np.zeros(N, dtype=person)

population['connectivity'] = N*np.random.random(size=N)
population['severity'] = 10*np.random.random(size=N)
population['days_infectious'] = np.random.randint(0, 8, size=N)  
population['is_sick'] = population['days_infectious'] > 0
population['status'] = np.random.choice(status_terms, size=N)



# compute infection of rate of connections
sick_population=population[population['is_sick']==True]
sum_connectivity_sick = np.sum(sick_population['connectivity'])
total_connectivity=np.sum(population['connectivity'])
infection_rate=sum_connectivity_sick/total_connectivity


not_sick_population=population[population['is_sick']==False]


    





import random

#No reinfection is allowed in here
class Person:
    def __init__(self, post_infection, severity, connectivity, days_infected):
        self.post_infection = post_infection
        self.severity = severity
        self.connectivity = connectivity
        self.days_infected = days_infected


def timestep_infection(population, C, T):
    total_connectivity = sum(person.connectivity for person in population if person.post_infection < T)
    infected_connectivity = sum(
        person.connectivity for person in population if person.post_infection >= 0 and person.days_infected > 0)

    infection_rate = infected_connectivity / total_connectivity

    for person in population:
        if person.post_infection == -1:
            p = person.connectivity * infection_rate * C
            if random.random() < p:
                person.post_infection = 0
                person.days_infected = 14  # Example: infected for 14 days
        elif person.post_infection >= 0:
            if person.days_infected > 0:
                person.days_infected -= 1
                if person.days_infected == 0:
                    person.post_infection = person.severity


def value_function(population, T):
    total_value = 0
    for person in population:
        if person.post_infection == -1:
            total_value += 1  # Never infected
        elif person.post_infection < T:
            if person.days_infected > 0:
                # Account for individuals mid-infection at the end of the simulation
                final_severity = person.severity
                total_value += 1 - (final_severity / 10)  # Scale severity to [0, 1] range
            else:
                total_value += 1 - (person.post_infection / 10)  # Scale post_infection to [0, 1] range
    return total_value / len(population)


def run_simulation(population, timesteps, C, T):
    for _ in range(timesteps):
        timestep_infection(population, C, T)

    overall_impact = value_function(population, T)
    return overall_impact


def analyze_post_infection_distribution(population, T):
    never_infected = sum(1 for person in population if person.post_infection == -1)
    mild_outcome = sum(1 for person in population if 0 <= person.post_infection < 3)
    moderate_outcome = sum(1 for person in population if 3 <= person.post_infection < 6)
    severe_outcome = sum(1 for person in population if 6 <= person.post_infection < T)
    deceased = sum(1 for person in population if person.post_infection >= T)

    total_population = len(population)

    print("Post-Infection Distribution:")
    print(f"Never Infected: {never_infected} ({never_infected / total_population * 100:.2f}%)")
    print(f"Mild Outcome: {mild_outcome} ({mild_outcome / total_population * 100:.2f}%)")
    print(f"Moderate Outcome: {moderate_outcome} ({moderate_outcome / total_population * 100:.2f}%)")
    print(f"Severe Outcome: {severe_outcome} ({severe_outcome / total_population * 100:.2f}%)")
    print(f"Deceased: {deceased} ({deceased / total_population * 100:.2f}%)")


# Test case
population_size = 100000
initial_infected = 100
timesteps = 120  # Simulating 120 days
C = 0.1
T = 8  # Threshold for considering an individual as deceased

population = [Person(-1, random.uniform(0, 10), random.randint(1, 20), 0) for _ in range(population_size)]

# Infect initial individuals
for i in range(initial_infected):
    population[i].post_infection = 0
    population[i].days_infected = 14

overall_impact = run_simulation(population, timesteps, C, T)
print(f"Overall Impact (Value Function): {overall_impact:.4f}")

analyze_post_infection_distribution(population, T)



