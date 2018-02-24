import numpy as np
from lr import model
from lr_utils import load_dataset
import matplotlib.pyplot as plt

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print('train_set_x shape: ' + str(train_set_x_orig.shape))
print('train_set_y shape: ' + str(train_set_y.shape))
print('test_set_x shape: ' + str(test_set_x_orig.shape))
print('test_set_y shape: ' + str(test_set_y.shape))
print()

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print('train_set_x_flatten shape: ' + str(train_set_x_flatten.shape))
print('train_set_y shape: ' + str(train_set_y.shape))
print('test_set_x_flatten shape: ' + str(test_set_x_flatten.shape))
print('test_set_y shape: ' + str(test_set_y.shape))
print()

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# learning_rate=0.005
d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

costs = d['costs']
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Learning rate =' + str(d['learning_rate']))
plt.show()
print()

# Choice of learning rate
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print('learning rate is: ' + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x,
                           test_set_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n-------------------------------------------------------\n')

for i in learning_rates:
    plt.plot(models[str(i)]['costs'],
             label=str(models[str(i)]['learning_rate']))

plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
