#imports 


import numpy as np
import tensorflow as tf
from scipy.stats import qmc
import matplotlib.pyplot as plt
import time

n_bc = 4
n_data_per_bc = 8

engine = qmc.LatinHypercube(d=1)
data = np.zeros([n_bc, n_data_per_bc, 3])

for i, j in zip(range(n_bc), [-1, +1, -1, +1]):
    points = (engine.random(n=n_data_per_bc)[:, 0] - 0.5) * 2
    if i < 2:
        data[i,:,0] = j
        data[i,:,1] = points
    else:
        data[i,:,0] = points
        data[i,:,1] = j

data[1,:,2] = 1.
data[2,:,2] = 1.
data[3,:,2] = 1.

data = data.reshape(n_data_per_bc*n_bc, 3)

#
x_d, y_d, t_d = map(lambda x: np.expand_dims(x, axis=1), 
                    [data[:, 0], data[:, 1], data[:, 2]])


print(x_d.shape, y_d.shape, t_d.shape)

#
N_c = 10000
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(n=N_c)
colloc = 2 * (colloc -0.5)

x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), 
               [colloc[:, 0], colloc[:, 1]])

# Transform to float32
x_c, y_c, x_d, y_d, t_d =map(lambda x: tf.convert_to_tensor(x,dtype=tf.float32),
                             [x_c, y_c, x_d, y_d, t_d])


print(x_c.shape, y_c.shape, x_d.shape, y_d.shape, t_d.shape)
print(t_d)

# plt.figure("", figsize=(6, 6))
# plt.title("Boundary Data points and Collocation points")

# plt.scatter(x_d, y_d, marker='x', c='k', label='BDP')
# plt.scatter(x_c, y_c, s=.2, marker=".", c="r", label="CP")
# plt.show()

### model builder function
def DNN_builder(in_shape=2, out_shape=1, n_hidden_layers=10, 
                neuron_per_layer=20, actfn="tanh"):
    # input layer
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    
    # hidden layers
    hidden = [tf.keras.layers.Dense(neuron_per_layer, activation=actfn)(input_layer)]
    for i in range(n_hidden_layers-1):
        new_layer = tf.keras.layers.Dense(neuron_per_layer,
                                          activation=actfn,
                                          activity_regularizer=None)(hidden[-1])
        hidden.append(new_layer)
    # output layer
    output_layer = tf.keras.layers.Dense(1, activation=None)(hidden[-1])
    # building the model
    name = f"DNN-{n_hidden_layers}"
    model = tf.keras.Model(input_layer, output_layer, name=name)
    return model

layer = 3
loss_dict = {}
train_time = {}


@tf.function
def f(x, y):
    u0 = u(x, y)
    u_x = tf.gradients(u0, x)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    F = u_xx + u_yy
    return tf.reduce_mean(tf.square(F))

@tf.function
def u(x, y):
    u = model(tf.concat([x, y], axis=1))
    return u
@tf.function
def mse(y, y_):
    return tf.reduce_mean(tf.square(y-y_))

model = DNN_builder(2, 1, layer, 20, "tanh")

loss = 0
epochs = 100
opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

# 用于计算预测
n = 300
X = np.linspace(-1, +1, n)
Y = np.linspace(-1, +1, n)
X0, Y0 = np.meshgrid(X, Y)
X = X0.reshape([n*n, 1])
Y = Y0.reshape([n*n, 1])
X_T = tf.convert_to_tensor(X)
Y_T = tf.convert_to_tensor(Y)

epoch = 0
loss_values = np.array([])
start = time.time()
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        T_ = u(x_d, y_d)
        l = mse(t_d, T_)

        L = f(x_c, y_c)
        loss = l+L
    g = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(g, model.trainable_weights))
    loss_values = np.append(loss_values, loss)

    # make predictions / 20 epoch
    if epoch % 20 == 0 or epoch == epochs-1:
        print(f"{epoch:5}, {loss.numpy():.3f}")
        S = u(X_T, Y_T)*10.
        S = S.numpy().reshape(n, n)
        
end = time.time()
print(f"\ncomputation time: {end-start:.3f}\n")

loss_dict[layer] = loss_values
train_time[layer] = end-start

plt.figure("", figsize=(8, 6), dpi=200)

#
X = np.linspace(-2, +2, n)
Y = np.linspace(-2, +2, n)
X0, Y0 = np.meshgrid(X, Y)
X = X0.reshape([n*n, 1])
Y = Y0.reshape([n*n, 1])
X_T = tf.convert_to_tensor(X)
Y_T = tf.convert_to_tensor(Y)

plt.subplot(111)
S = u(X_T, Y_T)
S = S.numpy().reshape(n, n)
plt.pcolormesh(-X0, Y0, 10.*S, cmap="jet")
# plt.title("PINN 8 Hidden Layers")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.colorbar()
plt.show()