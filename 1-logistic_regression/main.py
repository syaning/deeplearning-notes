import numpy as np
from lr import model
from lr_utils import load_dataset
import matplotlib.pyplot as plt


# 1. Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# 2. Example of a picture
index = 5
plt.imshow(train_set_x_orig[index])
print("y = %s, it's a %s picture.\n" % (
    str(train_set_y[:, index]),
    classes[np.squeeze(train_set_y[:, index])].decode('utf-8')
))
plt.show()


# 3. Inspect dimensions
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print('Number of training examples: m_train = ' + str(m_train))
print('Number of testing examples: m_test = ' + str(m_test))
print('Height/Width of each image: num_px = ' + str(num_px))
print('Each image is of size: (' + str(num_px) + ', ' + str(num_px) + ', 3)')
print('train_set_x shape: ' + str(train_set_x_orig.shape))
print('train_set_y shape: ' + str(train_set_y.shape))
print('test_set_x shape: ' + str(test_set_x_orig.shape))
print('test_set_y shape: ' + str(test_set_y.shape) + '\n')


# 4. Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print('train_set_x_flatten shape: ' + str(train_set_x_flatten.shape))
print('train_set_y shape: ' + str(train_set_y.shape))
print('test_set_x_flatten shape: ' + str(test_set_x_flatten.shape))
print('test_set_y shape: ' + str(test_set_y.shape) + '\n')


# 5. Standardize dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# 6. Train
d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)


# 7. Plot learning curve (with costs)
costs = d['costs']
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Learning rate = ' + str(d['learning_rate']))
plt.show()


# 8. Choice of learning rate
print('\n-------------------------------------------------------\n')
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print('learning rate is: ' + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y,
                           num_iterations=1500, learning_rate=i, print_cost=False)
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
