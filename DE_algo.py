import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from functools import reduce
import concurrent.futures as futures
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
# loading dataset
digits = load_digits()
x = digits.data
y = digits.target
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

def relu(x):
    return np.maximum(0, x)
def leaky_relu(x, a=0.01):
    return np.maximum(a*x, x)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
class Layer:
    def __init__(self, input_units, output_units, activation_function=None):
#         random_array = np.random.uniform(low, high, size=(input_units, output_units))
        self.W = np.random.uniform(-1, 1, size=(input_units, output_units))
        self.b = np.random.uniform(-1, 1, size=(output_units))
        self.activation_function = activation_function

    def forward(self, inputs):
        linear_output = np.dot(inputs, self.W) + self.b
        if self.activation_function is None:
            return linear_output
        else:
            return self.activation_function(linear_output)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.model_weights = []
        self.model_biases = []
        self.accuracy = None

    def add(self, layer):
        self.layers.append(layer)
        self.model_weights.append(layer.W)
        self.model_biases.append(layer.b)

    def predict_prob(self, input_data):
        return [reduce(lambda output, layer: layer.forward(output), self.layers, data) for data in input_data]

    def predict(self, input_data):
        prob_predictions = self.predict_prob(input_data)
        return [np.argmax(prediction) for prediction in prob_predictions]


    def print_architecture(self):
        print("Model Architecture:")
        for i, layer in enumerate(self.layers):
            activation = layer.activation_function.__name__ if layer.activation_function else "None"
            print(f"Layer {i+1}: Units={layer.W.shape[1]}, Activation={activation}")

    def print_weights(self):
        print("Model Weights:")
        for i, w in enumerate(self.model_weights):
            print(f"Layer {i+1} shape : {w.shape} weights:")
            print(w)
            
    def set_accuracy(self, input_data, true_labels):
        predicted_labels = self.predict(input_data)
        self.accuracy= accuracy_score(true_labels,predicted_labels)*100
    def get_accuracy(self):
        return self.accuracy
def gen_individual():
    neural_network = NeuralNetwork()
    first_layer = Layer(64, 32, leaky_relu)
    second_layer = Layer(32, 32, leaky_relu)
    last_layer = Layer(32, 10, softmax)
    neural_network.add(first_layer)
    neural_network.add(second_layer)
    neural_network.add(last_layer)
    return neural_network
def euclidean_distance(ind1, ind2):
    total_distance = 0
    for w1, w2, b1, b2 in zip(ind1.model_weights, ind2.model_weights, ind1.model_biases, ind2.model_biases):
        total_distance += np.sum((w1 - w2) ** 2)
        total_distance += np.sum((b1 - b2) ** 2)
    return np.sqrt(total_distance)
def fitness(individual):
    individual.set_accuracy(x, y)
    accuracy = individual.get_accuracy()
    return accuracy
def shared_fitness(individual, population, sigma_share=20.5, alpha=1.0):
    distances = np.array([euclidean_distance(individual, other) for other in population if individual != other])
    sharing_sum = np.sum(1 - (distances[distances < sigma_share] / sigma_share) ** alpha)
    raw_fitness = fitness(individual)
    adjusted_fitness = raw_fitness / (1 + sharing_sum)
    return adjusted_fitness

def generate_population(population_size):
    population = []
    fitnesses = []
    for i in range(population_size):
        individual = gen_individual()
        population.append(individual)
        fitnesses.append(fitness(individual))
    return population, fitnesses
def new_child(p1, p2, p3, target,cr, f):
    mutant=gen_individual()
    trial=gen_individual()
    for i in range(len(mutant.model_weights)):
      differencew = f * (p1.model_weights[i] - p2.model_weights[i])
      differenceb = f * (p1.model_biases[i] - p2.model_biases[i])
      mutant.model_weights[i]= p3.model_weights[i] + differencew
      mutant.model_biases[i]= p3.model_biases[i]  + differenceb

      for j in range(len(trial.model_weights[i])):
        trial.model_weights[i][j]= mutant.model_weights[i][j] if random.random() <= cr else target.model_weights[i][j]
      for j in range(len(trial.model_biases[i])):
        trial.model_biases[i][j]= mutant.model_biases[i][j] if random.random() <= cr else target.model_biases[i][j]
    return trial
    
def evaluate_target(args):
    target, population, f, cr, sigma_share, alpha = args
    a, b, c = random.sample(list(population), 3)
    trial = new_child(a, b, c, target, cr, f)
    trial_acc = shared_fitness(trial, population, sigma_share, alpha)
    target_acc = shared_fitness(target, population, sigma_share, alpha)
    if trial_acc > target_acc:
        result = [trial, trial_acc]
    else:
        result = [target, target_acc]
    return result
def differential_evolution(population_size, f=1.5, cr=0.9, max_iters=240, sigma_share=130.0, alpha=1.0, max_stall_iters=10, percentage_to_keep=0.25):
    # Initialize history dictionary to store accuracies
    history = {
        "best": [],
        "average": [],
        "worst": []
    }
    population, fitnesses = generate_population(population_size)
    best = population[np.argmax(fitnesses)]
    best_acc = max(fitnesses)
    print(best_acc)
    num_iters = 0
    stall_iters = 0
    while num_iters < max_iters:
        with futures.ThreadPoolExecutor(max_workers=12) as executor:
            args_list = [(target, population, f, cr, sigma_share, alpha) for target in population]
            try:
                results = list(executor.map(evaluate_target, args_list))
            except Exception as e:
                print(f"Error: {e}")
                break
  
        results = np.array(results)
        
        new_population = results[:,0]
        new_fitnesses = results[:,1]



        best_index = np.argmax(new_fitnesses)
        new_best_acc = new_fitnesses[best_index]

        if new_best_acc > best_acc:
            best_acc = new_best_acc
            best = new_population[best_index]
            stall_iters = 0
        else:
            stall_iters += 1

        population = new_population
        num_iters += 1
        avr=np.mean(new_fitnesses)

        # Store accuracies in history dictionary
        history["best"].append(best_acc)
        history["average"].append(avr)
        history["worst"].append(np.min(new_fitnesses))

        print(f"Iteration: {num_iters} -> Best Accuracy: {best_acc:.2f} % -> Average:{avr:.2f} ")

        # Restart if best accuracy has not improved for max_stall_iters iterations
        if stall_iters == max_stall_iters:
            print(f"Restarting population due to stalling for {max_stall_iters} iterations...")
            sorted_indices = np.argsort(-new_fitnesses)
            keep = int(percentage_to_keep*population_size)
            new_indx = sorted_indices[:keep]
            keep_pop = new_population[new_indx]
            population, fitnesses = generate_population(population_size-keep)
            # population.append(best)
            # fitnesses.append(best_acc)
            population = np.concatenate((keep_pop, population))
            best = population[np.argmax(fitnesses)]
            best_acc = max(fitnesses)
            stall_iters = 0

    return population, best, best_acc, history
def plot_history(history):
    fig = go.Figure()

    for key in history.keys():
        fig.add_trace(go.Scatter(x=list(range(1, len(history[key]) + 1)), y=history[key], mode='lines', name=key))

    fig.update_layout(title='Evolution of Accuracy',
                      xaxis_title='Iteration',
                      yaxis_title='Accuracy',
                      legend_title='Accuracy Type')

    pio.show(fig)

