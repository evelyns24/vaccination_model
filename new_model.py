import numpy as np

"""
Define the state of the system

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
"""
t = 0 # time step we are currently at
N=107000 # Population size
# define a structured data type to represent a person
Person = [  ('connectivity','i'),
            ('case_severity','i'),
            ('infection_period','i'),
            ('is_infected','b')] 

population=np.zeros(N, dtype=Person)
init_infections = 10 # how many people were infected at start of simulation

# Initialize the state
population['connectivity'] = np.random.random(size=N)*40
population['case_severity'] = np.random.random(size=N)*5
population['infection_period'] = np.random.randint(0, 8, size=N)  
population['is_infected'][np.random.choice(N, init_infections, replace=False)] = True # Randomly chooses `init_infections` number of people and sets them to infected.


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
    ## Filter who is infected currently
    infected_population = population[population['is_infected']==True]
    
    ## Deal with the uninfected people
    # compute the general infection rate in the population
    infected_edges = np.sum(infected_population['connectivity'])
    total_edges = np.sum(population['connectivity'])
    infection_rate = C * (infected_edges / total_edges)

    # compute likelihood of infection for the uninfected population
    uninfected_population = population[population['is_infected']==False]
    prob_infection = infection_rate * uninfected_population['connectivity']
    
    # determine whether or not person will be infected and updating states for newly infected people
    random=np.random.random(size=uninfected_population.size) # random outcomes for uninfected people
    newly_infected_people = uninfected_population[random<prob_infection] # who should be infected in this timestep (the people)
    newly_infected_indices = np.isin(population, newly_infected_people)
    population['is_infected'][newly_infected_indices] = True

    ## Deal with the infected people
    # decrement infected people's infection_period by 1
    infected_indices = population['is_infected']
    population['infection_period'][infected_indices] -= 1 # filter by column `infection_period` and then keep only rows with infected people.
    
    # if infection_period==0, then person has recoverd so change is_infected to False
    recovered_indices = (population['infection_period'] == 0) & infected_indices
    population['is_infected'][recovered_indices] = False

    return population

    # TODO: How do you include people dying? create a mask for alive people and then filter while calculating total_edges?

"""
Simulates the effect of vaccinating the population filtered by `who`. Mutates population.
"""
def vaccinate(population, who, how_many):
    return population
   
"""
Compute the cost to society that has been incurred at the current state.
""" 
def compute_cost(population):
    was_infected = population['infection_period'] < 4
    return np.sum(population['case_severity'][was_infected])
