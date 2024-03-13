%%time
import tensorflow as tf
import time


print(tf.config.list_physical_devices('CPU'))
print(tf.config.list_physical_devices('GPU'))


class _Dense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(_Dense, self).__init__(name='Dense')
        self.num_outputs = num_outputs
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.1)
        
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                     shape = [int(input_shape[-1]),
                                             self.num_outputs])
        
    def call(self, inputs):
        return  self.bn(tf.matmul(inputs, self.kernel))
    
class Neural_Net(tf.keras.Model):
    def __init__(self, neurons):
        super(Neural_Net, self).__init__(name='Model')

        # Get activations
        self.activations = []
        self.custom_layers = []  # Use a different name to store custom layers
        for i, neu in enumerate(neurons):
            layer = _Dense(neu)
            activation = tf.nn.tanh if i < len(neurons) - 1 else tf.identity
            self.activations.append(activation)
            self.custom_layers.append(layer)  # Use the new name for storing layers

        self.neurons = neurons
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.1)

    def call(self, input_tensor, training=False):
        x = self.bn(input_tensor)
        for i in range(len(self.neurons)):
            x = self.custom_layers[i](x)
            x = self.activations[i](x)
        return x
    
    

dtype = tf.float32

T, N, d = 1., 1, 100

batch_size = 16

neurons = [d+100, d+100, 1]

training_steps = 750000

lr_boundaries =[250001, 500001]

lr_values = [0.001, 0.0001, 0.00001]

mc_rounds, mc_freq = 10, 100 # Monte Carlo Methods

xi = tf.random.uniform(shape=(batch_size, d), 
                      minval=0.,
                      maxval=1.,
                      dtype=dtype)

x_sde = xi + tf.random.normal(shape=(batch_size, d),
                             stddev = np.sqrt(2.*T/N),
                             dtype = dtype)

def phi(x):
    return tf.reduce_sum(x ** 2, axis=1, keepdims=True)


model = Neural_Net(neurons)
model(xi)
model.summary()


# Define custom loss function
def custom_loss(model, x, y, training):
    y_ = model(x, training=training)
    
    return tf.reduce_mean((y_- y)**2)

l = custom_loss(model, xi, phi(x_sde), training=False)
print("Loss test: {}".format(l))








# Define the gradient function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = custom_loss(model, inputs, targets, training = True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)






# optimizer = tf.keras.optimizers.legacy.Adam(1e-3)
# Compile the model
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries, values=lr_values)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    
model.compile(optimizer=optimizer, loss=custom_loss)



def approx_err(model, x, u_reference, mc_rounds):
    l1_err, l2_err, li_err = 0., 0., 0.
    rel_l1_err, rel_l2_err, rel_li_err = 0., 0., 0.
    
    for _ in range(mc_rounds):
        u_approx = model(x, training=False)
        err = tf.abs(u_approx - u_reference)
        
        l1_err += tf.reduce_mean(err)
        l2_err += tf.reduce_mean(err**2)
        li_err = tf.maximum(li_err, tf.reduce_mean(err))
        
        rel_err = err / tf.maximum(u_reference, 1e-8)
        rel_l1_err += tf.reduce_mean(rel_err)
        rel_l2_err += tf.reduce_mean(rel_err**2)
        rel_li_err = tf.maximum(rel_li_err, tf.reduce_max(rel_err))
    
    l1_err /= mc_rounds
    l2_err = np.sqrt(l2_err / mc_rounds)
    rel_l1_err /= mc_rounds
    rel_l2_err = np.sqrt(rel_l2_err / mc_rounds)
    
    return err, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err



num_epochs = training_steps
file_name = "draft3.txt"
# Open the file for writing results
with open(file_name, 'w') as file_out:
    file_out.write('step,err,l1_err,l2_err,li_err,l1_rel,l2_rel,li_rel,learning_rate,time_train,time_mc\n')

    for epoch in range(num_epochs):
        start_time = time.time()
    
        loss_value, grads = grad(model, xi, phi(x_sde))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        l.append(loss_value.numpy())

        if (epoch+1)%mc_freq == 0:
            t1_training = time.time()
            err, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err = approx_err(model, xi, phi(x_sde), mc_rounds)
            end_time = time.time()
            
            current_lr = lr_schedule(optimizer.iterations).numpy()

            # Write results to file
            file_out.write(f'{epoch+1}, {tf.reduce_max(err)}, {l1_err}, {l2_err}, {li_err}, {rel_l1_err}, {rel_l2_err}, {rel_li_err}, {current_lr}, {t1_training - start_time}, {end_time - t1_training}\n')
            file_out.flush()
        
            print(epoch+1, current_lr)
            
            
            
import matplotlib.pyplot as plt
import pandas as pd
labels = ['$L_{\infty}$',
         '$L^2$',
         '$L^1$']


df = pd.read_csv('draft3.txt')
df.head()
suffix = "-second"

fig = plt.figure(figsize = (6,5),dpi=500)
plt.plot(df['step'], df['li_rel'],c='r',label=labels[0]+suffix)
plt.plot(df['step'],df['l2_rel'],c='r',linestyle='dotted', label=labels[1]+suffix)
plt.plot(df['step'],df['l1_rel'],c='r',linestyle='--', label=labels[2]+suffix)


df = pd.read_csv('first_test.txt')
df.head()
suffix = "-first"

plt.plot(df['step'], df['li_rel'],c='b',label=labels[0]+suffix)
plt.plot(df['step'],df['l2_rel'],c='b',linestyle='dotted', label=labels[1]+suffix)
plt.plot(df['step'],df['l1_rel'],c='b',linestyle='--', label=labels[2]+suffix)


plt.xlim([0,300*1000])
plt.ylim([0,1.2])
plt.legend()

plt.savefig('history.png')







