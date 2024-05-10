import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from functools import reduce
import time

df = pd.read_csv('train.csv')

x = df.drop(["price_range"], axis=1)
y = df["price_range"].values.reshape(df.shape[0], 1)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


class Layer:
    def __init__(self, input_units, output_units, activation_function=None):
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
            print(f"Layer {i + 1}: Units={layer.W.shape[1]}, Activation={activation}")

    def print_weights(self):
        print("Model Weights:")
        for i, w in enumerate(self.model_weights):
            print(f"Layer {i + 1} shape : {w.shape} weights:")
            print(w)

    def set_accuracy(self, input_data, true_labels):
        predicted_labels = self.predict(input_data)
        self.accuracy = accuracy_score(true_labels, predicted_labels) * 100

    def get_accuracy(self):
        return self.accuracy


def gen_individual():
    neural_network = NeuralNetwork()
    first_layer = Layer(20, 32, relu)
    second_layer = Layer(32, 32, relu)
    third_layer = Layer(32, 32, relu)
    last_layer = Layer(32, 4, softmax)
    neural_network.add(first_layer)
    neural_network.add(second_layer)
    neural_network.add(third_layer)
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


def new_child(p1, p2, p3, target, cr, f):
    mutant = gen_individual()
    trial = gen_individual()
    for i in range(len(mutant.model_weights)):
        differencew = f * (p1.model_weights[i] - p2.model_weights[i])
        differenceb = f * (p1.model_biases[i] - p2.model_biases[i])
        mutant.model_weights[i] = p3.model_weights[i] + differencew
        mutant.model_biases[i] = p3.model_biases[i] + differenceb

        for j in range(len(trial.model_weights[i])):
            trial.model_weights[i][j] = mutant.model_weights[i][j] if random.random() <= cr else \
                target.model_weights[i][j]
        for j in range(len(trial.model_biases[i])):
            trial.model_biases[i][j] = mutant.model_biases[i][j] if random.random() <= cr else target.model_biases[i][j]
    return trial


def evaluate_target(target, population, f, cr, sigma_share, alpha):
    a, b, c = random.sample(list(population), 3)
    trial = new_child(a, b, c, target, cr, f)
    trial_acc = shared_fitness(trial, population, sigma_share, alpha)
    target_acc = shared_fitness(target, population, sigma_share, alpha)
    if trial_acc > target_acc:
        result = [trial, trial_acc]
    else:
        result = [target, target_acc]
    return result


from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QHBoxLayout, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys


class EAWithGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Differential Evolution')
        self.setStyleSheet('background-color: #524C42;')
        self.input_fields = []
        self.output_text = None
        self.accuracy_label = None
        self.visual_output_layout = None
        self.generations_records = []
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.initializing_gui()

    def initializing_gui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_box = QWidget(self)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_box.setStyleSheet('background-color: #E2DFD0')
        main_layout.addWidget(left_box, 2)

        inputs_layout = QVBoxLayout(left_box)
        inputs_layout.setContentsMargins(20, 20, 20, 20)
        inputs_layout.setAlignment(Qt.AlignTop)
        inputs = ['Population Size', 'f Parameter', 'cr Parameter', 'Number of Generations', 'Sigma Share',
                  'alpha Parameter', 'Max Stall Iters', 'Percentage To Keep']
        for i in range(len(inputs)):
            label = QLabel(f'{inputs[i]}:')
            label.setStyleSheet('color: #524C42; font-size: 14px; font-weight: bold;')
            input_field = QLineEdit()
            input_field.setStyleSheet(
                'color: #E2DFD0; background-color: #524C42; border: 4px solid #32012F; padding: 5px;')
            input_layout = QHBoxLayout()
            input_layout.setContentsMargins(0, 0, 0, 20)
            input_layout.addWidget(label)
            input_layout.addWidget(input_field)
            input_layout.setStretch(0, 1)
            input_layout.setStretch(1, 2)
            inputs_layout.addLayout(input_layout)
            self.input_fields.append(input_field)

        submit_button = QPushButton("Submit")
        submit_button.setStyleSheet('background-color: #524C42; color: #E2DFD0; border: none; padding: 10px;')
        submit_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        submit_button.clicked.connect(self.submit_inputs)
        inputs_layout.addWidget(submit_button)

        right_box = QWidget(self)
        right_box.setStyleSheet('background-color: #E2DFD0')
        main_layout.addWidget(right_box, 2)

        output_layout = QVBoxLayout(right_box)
        output_layout.setContentsMargins(20, 20, 20, 20)
        output_layout.setAlignment(Qt.AlignTop)

        top_box = QWidget(self)
        output_layout.addWidget(top_box, 1)

        results_layout = QVBoxLayout(top_box)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setAlignment(Qt.AlignTop)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet('background-color: #524C42; border: none;')
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        results_layout.addWidget(scroll_area, 4)

        self.output_text = QTextEdit()
        self.output_text.setStyleSheet('color: #E2DFD0; background-color: #524C42; padding: 10px;')
        self.output_text.setReadOnly(True)
        scroll_area.setWidget(self.output_text)

        self.accuracy_label = QLabel('Best Accuracy: ')
        self.accuracy_label.setStyleSheet(
            'color: #E2DFD0; font-size: 16px; font-weight: bold; background-color: #524C42; padding: 10px;')
        results_layout.addWidget(self.accuracy_label, 1)

        visual_output_layout = QWidget(self)
        visual_output_layout.setStyleSheet('background-color: #524C42;')
        output_layout.addWidget(visual_output_layout, 1)

        self.visual_output_layout_container = QVBoxLayout(visual_output_layout)
        self.visual_output_layout_container.setContentsMargins(0, 0, 0, 0)
        self.visual_output_layout_container.setAlignment(Qt.AlignTop)

        main_widget = QWidget(self)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def submit_inputs(self):
        population_size_input = int(self.input_fields[0].text())
        f_parameter_input = float(self.input_fields[1].text())
        cr_parameter_input = float(self.input_fields[2].text())
        num_generations_input = int(self.input_fields[3].text())
        sigma_share_input = float(self.input_fields[4].text())
        alpha_parameter_input = float(self.input_fields[5].text())
        max_stall_iters_input = int(self.input_fields[6].text())
        percentage_to_keep = float(self.input_fields[7].text())

        def differential_evolution(population_size, f=1.5, cr=0.9, max_iters=300, sigma_share=130.0, alpha=1.0,
                                   max_stall_iters=10, percentage_to_keep=0.3):
            population, fitnesses = generate_population(population_size)
            best = population[np.argmax(fitnesses)]
            best_acc = max(fitnesses)
            print(best_acc)
            num_iters = 0
            stall_iters = 0
            restart_output = None
            while num_iters < max_iters:
                pool = Pool(12)

                results = pool.starmap(evaluate_target,
                                       [(target, population, f, cr, sigma_share, alpha) for target in population])

                pool.close()

                results = np.array(results)

                new_population = results[:, 0]
                new_fitnesses = results[:, 1]

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
                avr = np.mean(new_fitnesses)
                print(f"Iteration: {num_iters} -> Best Accuracy: {best_acc:.2f} % -> Average:{avr:.2f} ")
                output = f'Iteration: {num_iters} -> Best Accuracy: {best_acc:.2f} % -> Average: {avr:.2f}'

                # Restart if best accuracy has not improved for max_stall_iters iterations
                if stall_iters == max_stall_iters:
                    restart_output = "Restarting population due to stalling for {max_stall_iters} iterations..."
                    sorted_indices = np.argsort(-new_fitnesses)
                    keep = int(percentage_to_keep * population_size)
                    new_indx = sorted_indices[:keep]
                    keep_pop = new_population[new_indx]
                    population, fitnesses = generate_population(population_size - keep)
                    # population.append(best)
                    # fitnesses.append(best_acc)
                    population = np.concatenate((keep_pop, population))
                    best = population[np.argmax(fitnesses)]
                    best_acc = max(fitnesses)
                    stall_iters = 0

                self.generations_records.append((num_iters, best_acc))
                if restart_output:
                    self.output_text.append(restart_output)
                    self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())
                restart_output = None
                self.output_text.append(output)
                self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())
                self.output_text.repaint()

            return population, best, best_acc

        start_time = time.time()
        population, best_solution, best_acc = differential_evolution(population_size_input, f_parameter_input,
                                                                     cr_parameter_input, num_generations_input,
                                                                     sigma_share_input, alpha_parameter_input,
                                                                     max_stall_iters_input, percentage_to_keep)
        self.accuracy_label.setText(self.accuracy_label.text() + str(best_acc))
        self.accuracy_label.adjustSize()
        self.accuracy_label.repaint()
        end_time = time.time()
        print((end_time - start_time) / 60)

        generations, accuracies = zip(*self.generations_records)
        self.ax.plot(generations, accuracies, marker='o', linestyle='-',
                     label='Best Accuracy')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Accuracy (%)')
        self.ax.set_title('Evolution of Accuracy')
        self.ax.legend()

        if self.canvas not in self.visual_output_layout_container.children():
            self.visual_output_layout_container.layout().addWidget(self.canvas)

        self.fig.tight_layout()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EAWithGUI()
    window.showMaximized()
    sys.exit(app.exec_())
