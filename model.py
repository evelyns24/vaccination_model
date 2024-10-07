import numpy as np

"""
Define the state of the system

# State variables:
connectivity: integer between 0 and 40 
– "How many people do I closely interact with in each timestep?"

case_severity: decimal between 0 and 10
- "If infected, how bad does it get for this person?"

infection_period: integer between 0 and 6
- "If I'm currently uninfected, how many days am I going to be infectious for?" -> randomly between 4 and 6 days.
- "If I am infected, how many more days am I infectious for?" -> decrements on each timestep.

current_state: integer between -1 and 1
- -1 if dead
- 0 if uninfected
- 1 if infected
"""
t = 0 # time step we are currently at
N=107000 # Population size
# define a structured data type to represent a person
Person = [  ('connectivity','i'),
            ('case_severity','f'),
            ('infection_period','i'),
            ('current_state','i')] 

population=np.zeros(N, dtype=Person)
init_infections = 10 # how many people were infected at start of simulation

# Initialize the state
population['connectivity'] = np.random.randint(0, 41, size=N)
population['case_severity'] = np.random.random(size=N)*10
population['infection_period'] = np.random.randint(4, 7, size=N)  
population['current_state'] = 0  # Initially, all are uninfected

# Randomly chooses `init_infections` number of people and sets them to infected
population['current_state'][np.random.choice(N, init_infections, replace=False)] = 1


"""
Takes state and evolves it naturally by one timestep. Mutates population.

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
"""
def evolve_state(population, C, death_threshold):
    # Filter who is infected currently
    infected_population_mask=population['current_state']==1
    infected_population = population[infected_population_mask]
    # Filter who is uninfected currently
    uninfected_population_mask = population['current_state']==0
    uninfected_population = population[uninfected_population_mask]

    ## Deal with the uninfected people
    # compute the general infection rate in the population
    infected_edges = np.sum(infected_population['connectivity'])
    total_edges = np.sum(population['connectivity'])
    infection_rate = C * (infected_edges / total_edges)

    # compute likelihood of infection for the uninfected population
    prob_infection = infection_rate * uninfected_population['connectivity']
    
    # Determine whether or not person will be infected and updating states for newly infected people
    # random outcomes for uninfected people
    random=np.random.random(size=uninfected_population.size)
    # create a mask for individuals who should be infected based on their probability of infection
    should_infect_mask=random < prob_infection
    # get indices of newly infected people in the population
    uninfected_indices = np.where(uninfected_population_mask)[0]
    newly_infected_indices = uninfected_indices[should_infect_mask]
    # update current_state
    population['current_state'][newly_infected_indices] = 1

    ## Deal with the infected people
    # Handle people whose infection period ended (checking case_severity to see if they died)
    dead_population_indices = np.where((population['current_state'] == 1) & 
                                       (population['infection_period'] == 0) & 
                                       (population['case_severity'] >= death_threshold))[0]
    population['current_state'][dead_population_indices] = -1 # mark them as dead

    # handle infected people whose infection period did not end yet
    remain_infected_population_indices = np.where((population['current_state'] == 1) &
                                                  (population['infection_period'] != 0))[0]
    population['infection_period'][remain_infected_population_indices] -= 1 
    


"""
Simulates the effect of vaccinating the population filtered by `who` (reduces case_sensitivity by 90%). 
Mutates population.
Parameters:
    - 'who' should be 'connectivity', 'case_sensitivity", or 'mixed'
    - how_many: integer between 2000 and 4000
"""
def vaccinate(population, who, how_many):
    if who=='mixed':
        # how do u do mixed??????
        a=0
    else:
        # sort indices based on 'who' and select the top individuals
        vaccinate_population_indices = np.argsort(population[who])[::-1][:how_many]
        
    # reduce case_severity by 90% for selected people 
    population['case_severity'][vaccinate_population_indices] *= 0.1


"""
Compute the cost to society that has been incurred at the current state.
""" 
def compute_cost(population):
    was_infected = population['infection_period'] < 4
    return np.sum(population['case_severity'][was_infected])




def run_simulation(population, timesteps, C, T):
    for _ in range(timesteps):
        evolve_state(population, C, T)

    overall_cost = compute_cost(population)
    return overall_cost


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




timesteps = 120  # Simulating 120 days
months = 3
C = 0.1
T = 8  # Threshold for considering an individual as deceased

# no policy: no vaccines 
overall_cost = run_simulation(population, timesteps, C, T)
print(f"Overall Cost: {overall_cost:.4f}")


## scenario 1: exactly 3000 vaccines available each month
print("Scenario 1: exactly 3000 vaccines per month")
# policy 1: distributing vaccines based on connectivity
for _ in range(months):
    vaccine = 3000
    vaccinate(population,'connectivity', vaccine)
    overall_cost = run_simulation(population, 30, C, T)
print(f"Overall Cost: {overall_cost:.4f}")

# policy 2: distributing vaccines based on case_severity
for _ in range(months):
    vaccine = 3000
    vaccinate(population,'case_severity', vaccine)
    overall_cost = run_simulation(population, 30, C, T)
print(f"Overall Cost: {overall_cost:.4f}")



## scenario 2: vaccines available each month can be anywhere between 2000 and 4000
print("Scenario 2: 2000 - 4000 vaccines per month")
# policy 1: distributing vaccines based on connectivity
for _ in range(months):
    vaccine = np.random.randint(2000, 4001)
    vaccinate(population,'connectivity', vaccine)
    overall_cost = run_simulation(population, 30, C, T)
print(f"Overall Cost: {overall_cost:.4f}")

# policy 2: distributing vaccines based on case_severity
for _ in range(months):
    vaccine = np.random.randint(2000, 4001)
    vaccinate(population,'case_severity', vaccine)
    overall_cost = run_simulation(population, 30, C, T)
print(f"Overall Cost: {overall_cost:.4f}")

    


