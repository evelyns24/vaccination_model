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

current_state: integer between -1 and 2
- -1 if dead
- 0 if never infected
- 1 if recovered (and so is uninfected now)
- 2 if infected

immunity_period: integer between 0 and 14
- track how long a person is immune after recovering or recieving a vaccine
- once this period is over, they become susceptible to reinfection again
"""
N=107000 # Population size
init_infections = 10 # how many people were infected at start of simulation
max_connectivity = 40
max_severity = 10
vaccine_immunization = 0.1 # vaccines reduce case_severity by 90%
partial_immunization = 0.1 # reduce case_severity by 90% after recovering from infection
max_immunity_period = 14 # immune for 14 timesteps after recovery/vaccination

# define a structured data type to represent a person
Person = [  ('connectivity','i'),
            ('case_severity','f'),
            ('infection_period','i'),
            ('current_state','i'),
            ('immunity_period','i')] 

"""
initializes a population and returns it
assign parameter 'seed' different values to get different population initializations 
"""
def init_population(N, init_infections, max_connectivity, max_severity, seed=0):
    population=np.zeros(N, dtype=Person)

    # for reproducibilty
    np.random.seed(seed)
    # Initialize the state
    population['connectivity'] = np.random.randint(0, max_connectivity+1, size=N)
    population['case_severity'] = np.random.random(size=N)*max_severity
    population['infection_period'] = np.random.randint(4, 7, size=N)  
    population['current_state'] = 0  # Initially, all are never infected (uninfected)
    population['immunity_period'] = 0  # Initially, no one is immune

    # Randomly chooses `init_infections` number of people and sets them to infected
    population['current_state'][np.random.choice(N, init_infections, replace=False)] = 2

    return population


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
    infected_population_mask=(population['current_state']==0) | (population['current_state']==1)
    infected_population = population[infected_population_mask]
    # Filter who is uninfected and has no immunity
    uninfected_population_mask = (population['current_state']==0) & (population['immunity_period']==0)
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
    population['current_state'][newly_infected_indices] = 2 # they are now marked as infected

    ## Deal with the infected people
    # Handle people whose infection period ended (checking case_severity to see if they died)
    dead_population_indices = np.where((population['current_state'] == 2) & 
                                       (population['infection_period'] == 0) & 
                                       (population['case_severity'] >= death_threshold))[0]
    population['current_state'][dead_population_indices] = -1 # mark them as dead


    recovered_population_indices = np.where((population['current_state'] == 2) & 
                                       (population['infection_period'] == 0) & 
                                       (population['case_severity'] < death_threshold))[0]
    population['current_state'][recovered_population_indices] = 1 # mark them as recovered
    # reset their infection_period
    population['infection_period'][recovered_population_indices] = np.random.randint(4, 7)  
    # reduce case_severity due partial immunization
    population['case_severity'][recovered_population_indices] *= partial_immunization
    # set immunity period
    population['immunity_period'][recovered_population_indices] = max_immunity_period

    # handle infected people whose infection period did not end yet
    remain_infected_population_indices = np.where((population['current_state'] == 2) &
                                                  (population['infection_period'] != 0))[0]
    population['infection_period'][remain_infected_population_indices] -= 1 

    # Decrease immunity_period for those who have immunity left
    immune_population_indices = np.where(population['immunity_period'] > 0)[0]
    population['immunity_period'][immune_population_indices] -= 1 
    

"""
Returns the score of each person in the population. The score of an individual is a 
weighted average of connectivity and case_severity. 
People with highest score will get vaccine in mixed_score policy
Parameter:
    - w_c is the weighting factor for connectivity and so (1-w_c) is the weighting factor for case_severity
"""
def compute_score(population, w_c):
    score = (population['connectivity']*w_c/max_connectivity) + (population['case_severity']*(1-w_c)/max_severity)
    return score


"""
Simulates the effect of vaccinating the population filtered by `who` (reduces case_sensitivity by 90%). 
Mutates population.
Parameters:
    - 'who' should be 'connectivity', 'case_sensitivity", or 'mixed_score'
    - how_many: integer between 2000 and 4000
"""
def vaccinate(population, who, how_many):
    if who=='mixed_score':
        # compute score
        w_c = 0.3  # connectivity gets a lower weight 
        scores = compute_score(population, w_c)

        # sort sort indices based on 'who' and select the top individuals
        vaccinate_population_indices = np.argsort(scores)[::-1][:how_many]

    else:
        # sort indices based on 'who' and select the top individuals
        vaccinate_population_indices = np.argsort(population[who])[::-1][:how_many]
        
    # reduce case_severity by 90% for selected people 
    population['case_severity'][vaccinate_population_indices] *= vaccine_immunization

    # set immunity_period for vaccinated people to max_immunity_period
    population['immunity_period'][vaccinate_population_indices] = max_immunity_period


"""
Compute the cost to society that has been incurred at the current state.
""" 
def compute_cost(population):
    # find indices of people who were infected (including recovered and dead people)
    was_infected = (population['current_state']==-1) | (population['current_state']==1) | (population['current_state']==2)
    infected_case_severity = np.sum(population['case_severity'][was_infected])
    return infected_case_severity




def run_simulation(population, timesteps, C, T):
    for _ in range(timesteps):
        evolve_state(population, C, T)

    overall_cost = compute_cost(population)
    return overall_cost




#--------------------Similuations--------------------------------------------


timesteps = 120  # Simulating 120 days
months = 4
C = 0.1
T = 8  # Threshold for considering an individual as deceased

# no policy: no vaccines 
overall_cost = run_simulation(init_population(N, init_infections, max_connectivity, max_severity, seed=0), 
                              timesteps, 
                              C, 
                              T)
print(f"Overall Cost if no vaccines at all: {overall_cost:.4f}")


## scenario 1: exactly 3000 vaccines available each month
print("Scenario 1: exactly 3000 vaccines per month")
# policy 1: distributing vaccines based on connectivity
for _ in range(months):
    vaccine = 3000
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'connectivity', vaccine)
    overall_cost = run_simulation(pop, 30, C, T)
print(f"Overall Cost with Connectivity Policy: {overall_cost:.4f}")

# policy 2: distributing vaccines based on case_severity
for _ in range(months):
    vaccine = 3000
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'case_severity', vaccine)
    overall_cost = run_simulation(pop, 30, C, T)
print(f"Overall Cost with Case_Severity Policy: {overall_cost:.4f}")

# policy 3: distributing vaccines based on mixed_score 
# (weighted average of connetivity and case_severity)
for _ in range(months):
    vaccine = 3000
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'mixed_score', vaccine)
    overall_cost = run_simulation(pop, 30, C, T)
print(f"Overall Cost with Mixed_Score Policy : {overall_cost:.4f}")




## scenario 2: vaccines available each month can be anywhere between 2000 and 4000
print("Scenario 2: 2000 - 4000 vaccines per month")
# policy 1: distributing vaccines based on connectivity
for m in range(1,months+1):
    vaccine = np.random.randint(2000, 4001)
    # print(f"Vaccines available in month {m}: {vaccine}")
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'connectivity', vaccine)
    overall_cost = run_simulation(pop, 30, C, T)
print(f"Overall Cost with Connectivity Policy: {overall_cost:.4f}")

# policy 2: distributing vaccines based on case_severity
for m in range(1,months+1):
    vaccine = np.random.randint(2000, 4001)
    # print(f"Vaccines available in month {m}: {vaccine}")
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop,'case_severity', vaccine)
    overall_cost = run_simulation(pop, 30, C, T)
print(f"Overall Cost with Case_Severity Policy: {overall_cost:.4f}")

# policy 3: distributing vaccines based on mixed_score 
# (weighted average of connetivity and case_severity)
for m in range(1,months+1):
    vaccine = np.random.randint(2000, 4001)
    # print(f"Vaccines available in month {m}: {vaccine}")
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop,'mixed_score', vaccine)
    overall_cost = run_simulation(pop, 30, C, T)
print(f"Overall Cost with Mixed_Score Policy: {overall_cost:.4f}")
    


