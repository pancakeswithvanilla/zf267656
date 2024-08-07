from skopt import gp_minimize
from skopt.space import Integer
from wgan import train
import sys
import os
sys.path.append('/work/zf267656/peltfolder')
results_cache = {}
def objective(critic_iterations):
    critic_iterations = int(critic_iterations[0])
    real_accuracy, fake_accuracy = train(critic_iterations)
    if critic_iterations not in results_cache:
        real_accuracy, fake_accuracy = train(critic_iterations)
        results_cache[critic_iterations] = (real_accuracy, fake_accuracy)
    return abs(fake_accuracy - 0.5) 


search_space = [Integer(1, 50, name='critic_iterations')]
result = gp_minimize(objective, search_space, n_calls=20, random_state=42, n_jobs=-1)
best_critic_iterations = result.x[0]
best_real_accuracy, best_fake_accuracy = results_cache[best_critic_iterations]
print("Best critic_iterations:", best_critic_iterations)
print("Real_accuracy:", best_real_accuracy)
print("Fake_accuracy:", best_fake_accuracy)