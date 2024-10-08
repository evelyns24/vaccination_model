import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

"""
Define the state of the system

# State variables:
connectivity: integer between 0 and 40 
– "How many people do I closely interact with in each timestep?"

severity: float between 0 and 100
- "If infected, what is the cost to society of this person's sickness?"

infection_period: integer between 0 and 6
- "If I'm currently uninfected, how many days am I going to be infectious for?" -> randomly between 4 and 6 days.
- "If I am infected, how many more days am I infectious for?" -> decrements on each timestep.

status: integer between -1 and 2
- -1 if dead
- 0 if never infected
- 1 if recovered (and so is uninfected now)
- 2 if infected
"""
N=107000 # Population size
init_infections = 10 # how many people were infected at start of simulation
max_connectivity = 40
max_severity = 100
vaccine_immunization = 0.1 # vaccines reduce severity by 90%
partial_immunization = 0.1 # reduce severity by 90% after recovering from infection

class Status(Enum):
    DEAD = -1
    NEVER_INFECTED = 0
    RECOVERED = 1
    INFECTED = 2

"""
initializes a population and returns it
"""
def init_population(N, init_infections, max_connectivity, max_severity, seed=0):
    # define a structured data type to represent a person
    Person = [  ('connectivity','i'),
                ('severity', 'f'),
                ('infection_period','i'),
                ('status','i')] 
    population=np.zeros(N, dtype=Person)

    np.random.seed(seed) # use the same random seed for reproducibilty
    # Initialize the state
    population['connectivity'] = np.random.randint(0, max_connectivity+1, size=N)

    # Generate random values between 0 and 100
    severity_values = np.random.random(size=N) * 100

    # Create severity categories based on the values
    population['severity'] = np.select(
        [
            severity_values >= 92.5,
            severity_values >= 51.4,
            severity_values >= 8.4
        ],
        [
            100,
            80,
            2
        ],
        default=0
    )

    population['infection_period'] = np.random.randint(4, 7, size=N)  
    population['status'] = Status.NEVER_INFECTED.value  # Initially, all are never infected (uninfected)

    # Randomly chooses `init_infections` number of people and sets them to infected
    population['status'][np.random.choice(N, init_infections, replace=False)] = Status.INFECTED.value

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
    infected_population_mask=(population['status']==Status.INFECTED.value) 
    infected_population = population[infected_population_mask]
    # Filter who is uninfected currently
    uninfected_population_mask = (population['status']==Status.NEVER_INFECTED.value) | (population['status']==Status.RECOVERED.value)
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
    # update status
    population['status'][newly_infected_indices] = Status.INFECTED.value # they are now marked as infected

    ## Deal with the infected people
    # Handle people whose infection period ended (checking severity to see if they died)
    dead_population_indices = np.where((population['status'] == Status.INFECTED.value) & 
                                       (population['infection_period'] == 0) & 
                                       (population['severity'] == 100))[0]
    population['status'][dead_population_indices] = -1 # mark them as dead

    recovered_population_indices = np.where((population['status'] == Status.INFECTED.value) & 
                                       (population['infection_period'] == 0) & 
                                       (population['severity'] <= 100))[0]
    population['status'][recovered_population_indices] = Status.RECOVERED.value # mark them as recovered
    # reset their infection_period
    population['infection_period'][recovered_population_indices] = np.random.randint(4, 7)  
    # reduce severity by partial immunization
    population['severity'][recovered_population_indices] *= partial_immunization

    # handle infected people whose infection period did not end yet
    remain_infected_population_indices = np.where((population['status'] == Status.INFECTED.value) &
                                                  (population['infection_period'] != 0))[0]
    population['infection_period'][remain_infected_population_indices] -= 1 
    

"""
Returns the score of each person in the population. The score of an individual is a 
weighted average of connectivity and severity. 
People with highest score will get vaccine in mixed_score policy
Parameter:
    - w_c is the weighting factor for connectivity and so (1-w_c) is the weighting factor for severity
"""
def compute_score(population, w_c):
    score = (population['connectivity']*w_c/max_connectivity) + (population['severity']*(1-w_c)/max_severity)
    
    # only vaccinate people not previously infected to avoid double counting immunity
    # assign score=0 to those people
    score[population['status']!=Status.NEVER_INFECTED.value] = 0
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
        
    # reduce severity by 90% for selected people 
    population['severity'][vaccinate_population_indices] *= vaccine_immunization


"""
Compute the cost to society that has been incurred at the current state.
"""
def compute_cost(population):
    # Find indices of people who were infected (including recovered and dead people)
    was_infected = (population['status']==Status.DEAD.value) | (population['status']==Status.RECOVERED.value) | (population['status']==Status.INFECTED.value)
        
    # Create masks for each severity level
    low_severity = (population['severity'] == 0) & was_infected
    medium_severity = (population['severity'] == 2) & was_infected
    high_severity = (population['severity'] == 80) & was_infected
    death = (population['severity'] == 100) & was_infected

    # Compute costs for each severity level
    low_cost = np.sum(population['severity'][low_severity])
    medium_cost = np.sum(population['severity'][medium_severity])
    high_cost = np.sum(population['severity'][high_severity])
    death_cost = np.sum(population['severity'][death])

    # Compute total cost
    total_cost = low_cost + medium_cost + high_cost + death_cost

    return {
        'low_severity': low_cost,
        'medium_severity': medium_cost,
        'high_severity': high_cost,
        'death': death_cost,
        'total': total_cost
    }


"""
Progress the simulation forward by `timesteps` and return cost_breakdown at the end of the simulation period.
"""
def run_simulation(population, timesteps, C, T):
    for _ in range(timesteps):
        evolve_state(population, C, T)

    cost_breakdown = compute_cost(population)
    return cost_breakdown

"""
Generates a stacked bar plot comparing the results of different vaccination policies.

Parameters:
    policies (list): List of policy names (e.g., ['No policy', 'Connectivity', 'Case Severity', 'Mixed Score'])
    values (dict): Dictionary where keys are scenario names (e.g., 'Scenario 1', 'Scenario 2') 
                   and values are lists of costs for each policy in the same order as the policies list.
    title (str): Title for the plot
    ylabel (str): Label for the y-axis

Returns:
    None: Displays the plot using plt.show()
"""
def generate_bar_plot(policies, values, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = list(values.keys())
    bottom = np.zeros(len(policies))
    
    for scenario in scenarios:
        ax.bar(policies, values[scenario], bottom=bottom, label=scenario)
        bottom += values[scenario]
    
    ax.set_xlabel('Vaccination Policies')
    ax.set_ylabel('Overall Cost to Society')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()

##--------------------Simulations--------------------------------------------
timesteps = 120  # Simulating 120 days
months = 4
C = 0.05 # to convert connectivity to probability
death_threshold = 8  # Threshold for considering an individual as deceased
results = {}
categories = ['low_severity', 'medium_severity', 'high_severity', 'death']

# no policy: no vaccines 
cost_breakdown = run_simulation(init_population(N, init_infections, max_connectivity, max_severity, seed=0), 
                              timesteps, 
                              C, 
                              death_threshold)
print(f"Overall Cost if no vaccines at all: {cost_breakdown['total']}")
results['No policy'] = cost_breakdown


## scenario 1: exactly 3000 vaccines available each month
print("Scenario 1: exactly 3000 vaccines per month")
# policy 1: distributing vaccines based on connectivity
for _ in range(months):
    vaccine = 3000
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'connectivity', vaccine)
    cost_breakdown = run_simulation(pop, 30, C, death_threshold)
print(f"Overall Cost with Connectivity Policy: {cost_breakdown['total']}")
results['Connectivity'] = cost_breakdown

# policy 2: distributing vaccines based on severity
for _ in range(months):
    vaccine = 3000
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'severity', vaccine)
    cost_breakdown = run_simulation(pop, 30, C, death_threshold)
print(f"Overall Cost with Severity Policy: {cost_breakdown['total']}")
results['Case Severity'] = cost_breakdown

# policy 3: distributing vaccines based on mixed_score 
# (weighted average of connetivity and severity)
for _ in range(months):
    vaccine = 3000
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'mixed_score', vaccine)
    cost_breakdown = run_simulation(pop, 30, C, death_threshold)
print(f"Overall Cost with Mixed_Score Policy : {cost_breakdown['total']}")
results['Mixed Score'] = cost_breakdown

# Generate bar plot
values = {category: [results[policy][category] for policy in results] for category in categories}
generate_bar_plot(results.keys(), values, 'Cost to society for different vaccination policies when 3000 vaccines are given each month.')

## scenario 2: vaccines available each month can be anywhere between 2000 and 4000
print("Scenario 2: 2000 - 4000 vaccines per month")
# policy 1: distributing vaccines based on connectivity
for m in range(1,months+1):
    vaccine = np.random.randint(2000, 4001)
    # print(f"Vaccines available in month {m}: {vaccine}")
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop, 'connectivity', vaccine)
    cost_breakdown = run_simulation(pop, 30, C, death_threshold)
print(f"Overall Cost with Connectivity Policy: {cost_breakdown['total']}")
results['Connectivity'] = cost_breakdown

# policy 2: distributing vaccines based on severity
for m in range(1,months+1):
    vaccine = np.random.randint(2000, 4001)
    # print(f"Vaccines available in month {m}: {vaccine}")
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop,'severity', vaccine)
    cost_breakdown = run_simulation(pop, 30, C, death_threshold)
print(f"Overall Cost with Severity Policy: {cost_breakdown['total']}")
results['Case Severity'] = cost_breakdown

# policy 3: distributing vaccines based on mixed_score 
# (weighted average of connetivity and severity)
for m in range(1,months+1):
    vaccine = np.random.randint(2000, 4001)
    # print(f"Vaccines available in month {m}: {vaccine}")
    pop = init_population(N, init_infections, max_connectivity, max_severity, seed=0)
    vaccinate(pop,'mixed_score', vaccine)
    cost_breakdown = run_simulation(pop, 30, C, death_threshold)
print(f"Overall Cost with Mixed_Score Policy: {cost_breakdown['total']}")
results['Mixed Score'] = cost_breakdown

# Generate bar plot
values = {category: [results[policy][category] for policy in results] for category in categories}
generate_bar_plot(results.keys(), values, 'Cost to society for different vaccination policies when 2000-4000 vaccines are given each month.')