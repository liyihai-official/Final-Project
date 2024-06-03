#imports 
import numpy as np
import tensorflow as tf
from scipy.stats import qmc
import matplotlib.pyplot as plt
import time

# Data Generation
num_boundary_condition = 4
num_data_per_bc = 32

engine = qmc.LatinHypercube(d=1)
data = np.zeros([num_boundary_condition, num_data_per_bc, 3])

for i, j in zip(range(num_boundary_condition), [-1, +1, -1, +1]):
    points = (engine.random(n=num_data_per_bc)[:, 0] - 0.5) * 2
    if i < 2:
        data[i,:,0] = j
        data[i,:,1] = points
    else:
        data[i,:,0] = points
        data[i,:,1] = j

data[1,:,2] = 1.
data[2,:,2] = 1.
data[3,:,2] = 1.

data = data.reshape(num_boundary_condition*num_data_per_bc, 3)

x_d, y_d, u_d = map(lambda x: np.expand_dims(x, axis=1), 
                    [data[:, 0], data[:, 1], data[:, 2]])

t_d = np.random.rand(128, 1)
t_d *= 100

# 区块中间的随机样本 用于训练
N_c = 10000
engine = qmc.LatinHypercube(d=3)
colloc = engine.random(n=N_c)
colloc = 2 * (colloc -0.5)

x_c, y_c, t_c = map(lambda x: np.expand_dims(x, axis=1), 
               [colloc[:, 0], colloc[:, 1], colloc[:, 2]])

# Transform to float32
x_c, y_c, t_c, x_d, y_d, t_d, u_d =map(lambda x: tf.convert_to_tensor(x,dtype=tf.float32),
                             [x_c, y_c, t_c, x_d, y_d, t_d, u_d])

# ---------------------------------------------------------------------------------------- #

def dnn(in_shape = 3, out_shape=1, n_hidden_layer=10, 
  neuron_per_layer=20, act_fn="tanh"):
    
  input_layer = tf.keras.layers.Input(shape=(in_shape,))

  hidden = [tf.keras.layers.Dense(neuron_per_layer, activation=act_fn)(input_layer)]
  for i in range(n_hidden_layer-1):
      new_layer = tf.keras.layers.Dense(neuron_per_layer,
                                        activation=act_fn,
                                        activity_regularizer=None)(hidden[-1])
      hidden.append(new_layer)
    
  output_layer = tf.keras.layers.Dense(1, activation=None)(hidden[-1])

  name = f"DNN-{n_hidden_layer}"
  model = tf.keras.Model(input_layer, output_layer, name=name)

  return model

layer = 3
model = dnn(3, 1, layer, 20, "tanh")

epochs = 500
opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

loss_dict = {}
train_time = {}


@tf.function
def u(x, y, t):
  u = model(tf.concat([x, y, t], axis=1))
  return u

@tf.function
def f(x, y, t):
  u0 = u(x, y, t)
  u_x = tf.gradients(u0, x)[0]
  u_y = tf.gradients(u0, y)[0]
  u_t = tf.gradients(u0, t)[0]
  u_xx = tf.gradients(u_x, x)[0]
  u_yy = tf.gradients(u_y, y)[0]
  F = u_xx + u_yy + u_t
  return tf.reduce_mean(tf.square(F))

@tf.function
def mse(y, y_):
    return tf.reduce_mean(tf.square(y-y_))


epoch = 0
loss_values = np.array([])
start = time.time()

for epoch in range(epochs):
  with tf.GradientTape() as tape:
    U_ = u(x_d, y_d, t_d)
    l = mse(u_d, U_)

    L = f(x_c, y_c, t_c)
    loss = l+L
  g = tape.gradient(loss, model.trainable_weights)
  opt.apply_gradients(zip(g, model.trainable_weights))
  loss_values = np.append(loss_values, loss)

end = time.time()
print(f"\ncomputation time: {end-start:.3f}\n")


plt.figure("", figsize=(8, 6), dpi=200)

#
n = 100

X = np.linspace(-1, +1, n)
Y = np.linspace(-1, +1, n)
T = np.linspace(-1, +1, n)
X0, Y0, T0 = np.meshgrid(X, Y, T)
X = X0.reshape([n*n*n, 1])
Y = Y0.reshape([n*n*n, 1])
T = T0.reshape([n*n*n, 1])
X_T = tf.convert_to_tensor(X)
Y_T = tf.convert_to_tensor(Y)
T_T = tf.convert_to_tensor(T)

plt.subplot(221)
S = u(X_T, Y_T, T_T)
S = S.numpy().reshape(n, n, n)
plt.pcolormesh(-X0[:,:,1], Y0[:,:,1], 10.*S[:,:,1], cmap="jet")
# # plt.title("PINN 8 Hidden Layers")
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.colorbar()

plt.subplot(222)
S = u(X_T, Y_T, T_T)
S = S.numpy().reshape(n, n, n)
plt.pcolormesh(-X0[:,:,1], Y0[:,:,1], 10.*S[:,:,25], cmap="jet")

plt.subplot(223)
S = u(X_T, Y_T, T_T)
S = S.numpy().reshape(n, n, n)
plt.pcolormesh(-X0[:,:,1], Y0[:,:,1], 10.*S[:,:,50], cmap="jet")

plt.subplot(224)
S = u(X_T, Y_T, T_T)
S = S.numpy().reshape(n, n, n)
plt.pcolormesh(-X0[:,:,1], Y0[:,:,1], 10.*S[:,:,99], cmap="jet")
plt.show()