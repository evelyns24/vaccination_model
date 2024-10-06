import numpy as np

# define a structured data type to represent a person
# 5 fields: is_sick, severity, connectivity, days_infectious, status
person = [('is_sick','b'),
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


    




