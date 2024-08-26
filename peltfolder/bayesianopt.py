from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import sys
import os
import signal
import pickle
from wgan import train

sys.path.append('/work/zf267656/peltfolder')
disc_acc_file_name = '/work/zf267656/peltfolder/discriminator_accuracy.txt'
def load_previous_results(file_name = disc_acc_file_name):
    previous_x = []
    previous_y = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        if not lines:
            print("The file is empty.")
            return previous_x, previous_y
        i = 0
        while i < len(lines):
            if i % 5 == 0:
                try:
                    critic_iterations = int(lines[i].strip())
                    previous_x.append([critic_iterations])
                except ValueError:
                    print(f"Skipping invalid critic_iterations line: {lines[i].strip()}")
            if i % 5 == 4:
                try:
                    fake_acc = float(lines[i].split(":")[1].strip())
                    previous_y.append(abs(fake_acc - 0.5))
                except ValueError:
                    print(f"Skipping invalid Fake_accuracy line: {lines[i].strip()}")
            i+=1
    return previous_x, previous_y
previous_x, previous_y = load_previous_results()
search_space = [Integer(1, 50, name='critic_iterations')]

# Define the objective function
@use_named_args(search_space)
def objective(critic_iterations):
    if [critic_iterations] in previous_x:
        index = previous_x.index([critic_iterations])
        return previous_y[index]  # Use cached value if already evaluated
    else:
        real_accuracy, fake_accuracy = train(critic_iterations)  # Implement your training function
        return abs(fake_accuracy - 0.5)

# Run the Bayesian optimization
if not previous_x or not previous_y:
    result = gp_minimize(
        objective,
        search_space,
        n_calls=20,
        random_state=42
    )
else:
    result = gp_minimize(
        objective, 
        search_space, 
        n_calls=20, 
        x0=previous_x, 
        y0=previous_y, 
        random_state=42
    )

# After optimization, you can print the best results
best_critic_iterations = result.x[0]
print("Best critic_iterations:", best_critic_iterations)
