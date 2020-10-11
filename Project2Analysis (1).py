#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install mlrose


# In[33]:


import six
import sys
import time
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from matplotlib import pyplot as plt


# In[35]:


#n queens is best solved by simulated annealing
#citation: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html


# In[63]:


fitness = mlrose.Queens()

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

   # Initialize counter
    fitness_cnt = 0

          # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # Check for horizontal, diagonal-up and diagonal-down attacks
                if (state[j] != state[i])                     and (state[j] != state[i] + (j - i))                     and (state[j] != state[i] - (j - i)):

                    # If no attacks, then increment counter
                    fitness_cnt += 1

    return fitness_cnt

    # Initialize custom fitness function object
    fitness_cust = mlrose.CustomFitness(queens_max)


# In[64]:


problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = True, max_val = 25)


# In[99]:


#N QUEENS PROBLEM DOMAIN 
schedule = mlrose.GeomDecay(init_temp=3.0, decay=0.99, min_temp=0.001)

# Define initial state
init_state = np.array([0,1,2,3,4,5,6,7])

# Solve problem using simulated annealing
start_sa = time.time()
best_state, best_fitness, sa_curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 800, max_iters = 1000,
                                                      init_state = init_state, random_state = 1, curve=True)
end_sa = time.time()
sa_total_time = end_sa - start_sa 


start_mimic = time.time()
best_state_2, best_fitness_2, curve = mlrose.mimic(problem, pop_size=150, keep_pct=0.1, max_attempts = 800, max_iters = 1000, 
                                        curve=True, random_state = None, fast_mimic=True)

end_mimic = time.time()
mimic_total_time = end_mimic - start_mimic 


start_ga = time.time()
best_state_3, best_fitness_3, ga_curve = mlrose.genetic_alg(problem, random_state = 0,mutation_prob=0.8,curve=True, max_attempts=800, max_iters=1000)
end_ga = time.time()
ga_total_time = end_ga - start_ga

start_rhc = time.time()
best_state_4, best_fitness_4, rhc_curve = mlrose.random_hill_climb(problem, max_attempts=500, max_iters=1000, restarts=5,curve=True, init_state=init_state)
end_rhc = time.time()
rhc_total_time = end_rhc - start_rhc

algo_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness value")
plt.plot(curve, label='mimic, N=8')
plt.plot(sa_curve, label='SA, N=8')
plt.plot(ga_curve, label='GA, N=8')
plt.plot(rhc_curve, label='RHC, N=8')
plt.legend()




time_fig = plt.figure()
ax = time_fig.add_axes([0,0,1,1])
times = [sa_total_time, mimic_total_time, ga_total_time, rhc_total_time]
algorithms = ["SA", "MIMIC", "GA", "RHC"]
ax.bar(algorithms, times)
ax.set_xlabel("Optimization Algorithm")
ax.set_ylabel("Computation Time")

plt.axis()
plt.show()

print("GA Time", ga_total_time)
print("SA Time", sa_total_time)
print("RHC Time", rhc_total_time)
print("MIMIC Time", mimic_total_time)

print("SA", best_state)
print("SA", best_fitness)
print("Mimic", best_state_2)
print("Mimic", best_fitness_2)
print("GA", best_state_3)
print("GA", best_fitness_3)
print("RHC", best_state_4)
print("RHC", best_fitness_4)



# In[7]:


#four peaks is ideal for MIMIC algorithm


# In[96]:


fitness = mlrose.FourPeaks(t_pct=0.15)

problem = mlrose.DiscreteOpt(length = 13, fitness_fn = fitness, maximize = True)

# Define initial state
init_state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])

# Solve problem using simulated annealing
start_mimic = time.time()
best_state, best_fitness, mimic_curve = mlrose.mimic(problem, pop_size=150, max_attempts=900, keep_pct=0.01, fast_mimic=True, curve=True)
end_mimic = time.time()
mimic_total = end_mimic-start_mimic

start_SA = time.time()
best_state_2, best_fitness_2, sa_curve = mlrose.simulated_annealing(problem,
                                                      max_attempts = 500, max_iters = 500, curve=True)
end_SA = time.time()
SA_total = end_SA - start_SA

start_GA = time.time()
best_state_3, best_fitness_3, ga_curve = mlrose.genetic_alg(problem, pop_size=25, max_iters=800, max_attempts=100, mutation_prob=0.01, curve=True)
end_GA = time.time()
GA_total = end_GA - start_GA

start_RHC = time.time()
best_state_4, best_fitness_4, rhc_curve = mlrose.random_hill_climb(problem, max_attempts=400, max_iters=800, restarts=0, init_state=init_state, curve=True)
end_RHC = time.time()
RHC_total = end_RHC - start_RHC

time_fig = plt.figure()
ax = time_fig.add_axes([0,0,1,1])
times = [SA_total, mimic_total, GA_total, RHC_total]
algorithms = ["SA", "MIMIC", "GA", "RHC"]
ax.bar(algorithms, times)
ax.set_xlabel("Optimization Algorithm")
ax.set_ylabel("Computation Time")

plt.axis()
plt.show()

algo_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness value")
plt.plot(ga_curve, label='GA, N=13')
plt.plot(mimic_curve, label='mimic, N=13')
plt.plot(sa_curve, label='SA, N=13')
plt.plot(rhc_curve, label='RHC, N=13')
plt.legend()
plt.show()

print("GA Time", GA_total)
print("SA Time", SA_total)
print("RHC Time", RHC_total)
print("MIMIC Time", mimic_total)



print("MIMIC", best_state)
print("MIMIC", best_fitness)
print("SA", best_state_2)
print("SA", best_fitness_2)
print("GA", best_state_3)
print("GA", best_fitness_3)
print("RHC", best_state_4)
print("RHC", best_fitness_4)


# In[9]:


#traveling salesman is best solved by the genetic algo 


# In[95]:


#citation: https://mlrose.readthedocs.io/en/stable/source/tutorial2.html#:~:text=What%20is%20a%20Travelling%20Salesperson%20Problem%3F&text=The%20travelling%20salesperson%20problem%20(TSP,the%20other%20cities%20exactly%20once.
# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (6, 8), (9,10), (11, 12), (14, 15), (15, 16), (17, 18), (18, 19)]
schedule = mlrose.GeomDecay(init_temp=4.0, decay=0.99, min_temp=0.10)



# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)

# Create list of distances between pairs of cities
dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426),              (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000),              (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426),              (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721),              (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056),              (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623),              (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

# Initialize fitness function object using dist_list
fitness_dists = mlrose.TravellingSales(distances = dist_list)

# Define optimization problem object
problem_fit = mlrose.TSPOpt(length = 15, fitness_fn = fitness_coords, maximize=True)

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (6, 8), (9,10), (11, 12), (14, 15), (15, 16), (17, 18), (18, 19)]

# Define optimization problem object
problem_no_fit = mlrose.TSPOpt(length = 15, coords = coords_list, maximize=True)

start_ga = time.time()
best_state, best_fitness, ga_curve = mlrose.genetic_alg(problem_fit, mutation_prob=0.5, random_state = 2, curve=True, max_attempts=350, max_iters=1000)
end_ga = time.time()
ga_total = end_ga - start_ga

start_mimic = time.time()
best_state_2, best_fitness_2, mimic_curve = mlrose.mimic(problem_fit, pop_size=200, keep_pct=0.1, max_attempts = 200, max_iters = 1000, random_state = None, fast_mimic=False, curve=True)
end_mimic = time.time()
mimic_total = end_mimic - start_mimic

start_sa = time.time()
best_state_3, best_fitness_3, sa_curve = mlrose.simulated_annealing(problem_fit, schedule = schedule,
                                                      max_attempts = 10, max_iters = 1000, random_state = 1, curve=True)
end_sa = time.time()
sa_total = end_sa - start_sa

start_rhc = time.time()
best_state_4, best_fitness_4, rhc_curve = mlrose.random_hill_climb(problem_fit, max_attempts=80, max_iters=500, restarts=2, init_state=None, random_state=None, curve=True)
end_rhc = time.time()

rhc_total = end_rhc - start_rhc

algo_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness value")
plt.plot(ga_curve, label='GA, N=15')
plt.plot(mimic_curve, label='mimic, N=15')
plt.plot(sa_curve, label='SA, N=15')
plt.plot(rhc_curve, label='RHC, N=15')
plt.legend()
plt.show()

time_fig = plt.figure()
ax = time_fig.add_axes([0,0,1,1])
times = [sa_total, mimic_total, ga_total_time, rhc_total]
algorithms = ["SA", "MIMIC", "GA", "RHC"]
ax.bar(algorithms, times)
ax.set_xlabel("Optimization Algorithm")
ax.set_ylabel("Computation Time")

plt.axis()
plt.show()

print("GA Time", ga_total)
print("SA Time", sa_total)
print("RHC Time", rhc_total)
print("MIMIC Time", mimic_total)


print("GA", best_state)
print("GA", best_fitness)
print("Mimic", best_state_2)
print("Mimic", best_fitness_2)
print("SA", best_state_3)
print("SA", best_fitness_3)
print("RHC", best_state_4)
print("RHC", best_fitness_4)


# In[11]:


#Neural Network Optimization


# In[37]:


#load in dataset used in assignment 1
digits = load_digits()
data_features = digits.data[:, 0:-1]
label = digits.data[:, -1]
digits_trainingX, digits_testingX, digits_trainingY, digits_testingY = train_test_split        (data_features, label, test_size=0.3, random_state=0,
                     stratify=label)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(digits_trainingX)
X_test_scaled = scaler.transform(digits_testingX)
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(digits_trainingY.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(digits_testingY.reshape(-1, 1)).todense()


# In[82]:


#citation: https://mlrose.readthedocs.io/en/stable/source/tutorial3.html#what-is-a-machine-learning-weight-optimization-problem
# Initialize neural network object and fit object
accuracy_curve_test = []
accuracy_curve_train = []
time_rhc = []


for i in range(i):
    start_time = time.time()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [3], activation = 'relu',                                  algorithm = 'random_hill_climb', max_iters = i,                                  bias = True, is_classifier = True, learning_rate = 0.0001,                                  early_stopping = False, max_attempts = 215,                                  random_state = 3, curve=True, restarts=4)
    nn_model1.fit(digits_trainingX, y_train_hot)
    end_time = time.time()
    y_train_pred = nn_model1.predict(X_train_scaled)
    accuracy_curve_train.append(accuracy_score(y_train_hot, y_train_pred))
    y_test_pred = nn_model1.predict(X_test_scaled)
    accuracy_curve_test.append(accuracy_score(y_test_hot, y_test_pred))
    time_rhc.append(end_time - start_time)


# In[83]:


nn_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(accuracy_curve_train, label='training')
plt.plot(accuracy_curve_test, label='testing')
plt.legend()
plt.show()


# In[84]:



y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(y_train_accuracy)
# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(X_test_scaled)

y_test_accuracy_rhc = accuracy_score(y_test_hot, y_test_pred)
print(y_test_accuracy_rhc)


# In[104]:


accuracy_curve_test_rhc = []
accuracy_curve_train_rhc = []
time_sa = []

for i in range(15): 
    start_time = time.time()
    nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [1], activation = 'relu',                                  algorithm = 'simulated_annealing', max_iters = i,                                  bias = True, is_classifier = True, learning_rate = 0.0001,                                  early_stopping = False, schedule=mlrose.GeomDecay(init_temp=3.0, decay=0.99, min_temp=0.001), max_attempts = 250,                                  random_state = 3, curve=True)
    nn_model2.fit(digits_trainingX, y_train_hot)
    end_time = time.time()
    y_train_pred = nn_model2.predict(X_train_scaled)
    y_train_accuracy_sa = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model2.predict(X_test_scaled)
    y_test_accuracy_sa = accuracy_score(y_test_hot, y_test_pred)
    accuracy_curve_train_rhc.append(y_train_accuracy_sa)
    accuracy_curve_test_rhc.append(y_test_accuracy_sa)
    time_sa.append(end_time-start_time)


# In[105]:


accuracy_curve_test_ga = []
accuracy_curve_train_ga = []
time_ga = []

for i in range(15): 
    start_time = time.time()
    nn_model3 = mlrose.NeuralNetwork(hidden_nodes = [1], activation = 'relu',                                  algorithm = 'genetic_alg', max_iters = i,                                  bias = True, is_classifier = True, learning_rate = 0.0001,                                  early_stopping = True , pop_size = 200, max_attempts = 200,                                  random_state = 3, curve=True)
    nn_model3.fit(digits_trainingX, y_train_hot)
    end_time = time.time()
    y_train_pred = nn_model3.predict(X_train_scaled)
    y_train_accuracy_ga = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model3.predict(X_test_scaled)
    y_test_accuracy_ga = accuracy_score(y_test_hot, y_test_pred)
    accuracy_curve_train_ga.append(y_train_accuracy_ga)
    accuracy_curve_test_ga.append(y_test_accuracy_ga)
    time_ga.append(end_time-start_time)
    
    


# In[106]:


accuracy_curve_test_backprop = []
accuracy_curve_train_backprop = []
time_backprop = []

for i in range(15): 
    start_time = time.time()
    nn_model4 = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu',                                  algorithm = 'gradient_descent', max_iters = i,                                  bias = True, is_classifier = True, learning_rate = 0.001,                                  early_stopping = False, max_attempts = 200, curve=True)
    nn_model4.fit(digits_trainingX, y_train_hot)
    end_time = time.time()
    y_train_pred = nn_model4.predict(X_train_scaled)
    y_train_accuracy_b = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model4.predict(X_test_scaled)
    y_test_accuracy_b = accuracy_score(y_test_hot, y_test_pred)
    accuracy_curve_train_backprop.append(y_train_accuracy_b)
    accuracy_curve_test_backprop.append(y_test_accuracy_b)
    time_backprop.append(end_time-start_time)
    
    


# In[109]:


nn_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness value")
plt.plot(nn_model1.fitness_curve, label='random hill climbing')
plt.plot(nn_model2.fitness_curve, label='simulated annealing')
plt.plot(nn_model3.fitness_curve, label='genetic algorithm')
plt.plot(nn_model4.fitness_curve, label='backpropagation')
plt.legend()
plt.show()


# In[108]:


nn_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Computation Time")
plt.plot(time_rhc, label='random hill climbing')
plt.plot(time_sa, label='simulated annealing')
plt.plot(time_ga, label='genetic algorithm')
plt.plot(time_backprop, label='backpropagation')
plt.legend()
plt.show()


# In[61]:


nn_model2.fitness_curve


# In[92]:


nn_curve = plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(accuracy_curve_test_rhc, label='random hill climbing')
plt.plot(accuracy_curve_test, label='simulated annealing')
plt.plot(accuracy_curve_test_ga, label='genetic algorithm')
plt.plot(accuracy_curve_test_backprop, label='backpropagation')
plt.legend()
plt.show()


# In[76]:


accuracy_curve_test_rhc


# In[81]:


time_rhc


# In[ ]:




