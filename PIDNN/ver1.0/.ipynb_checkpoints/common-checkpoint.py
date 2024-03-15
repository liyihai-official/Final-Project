import numpy as np
import time 
import tensorflow as tf


def neural_net(input_shape, neurons, dtype=tf.float32):
    '''
    This function create a neural net with 3 hidden layers, 
    the given parameters.

    Parameters:
    -------------------------------------------------------------
        input_shape: int
            the length of input vector (the number of samples in 
            interval [a, b])

        neurons: list
            with 3 integers, represents the size of hidden layers

        dtype: tensorflow.datatype
            Default is float32

            
    Returns: 
    -------------------------------------------------------------
        model: tensorflow.model
            Return the model created.
    
    '''

    def _layer(x, units, activation_fn, name=None):
        with tf.name_scope(name):
            # Create the weight variable using Xavier initializer
            w = tf.Variable(tf.initializers.glorot_normal()(shape=(x.shape[-1], units)), trainable=True, name='weights')

            # Create the biases variable
            b = tf.Variable(tf.zeros((units,), name='biases'), trainable=True)

            # Matrix multiplication
            z = tf.matmul(x, w) + b
            return activation_fn(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-6)(tf.matmul(x, w)))
    
    inputs = tf.keras.Input(shape = input_shape, dtype=dtype)

    x = inputs
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-6)(x)
    for i, neu in enumerate(neurons):
        x = _layer(x, neu,
                tf.nn.tanh if i < len(neurons) - 1 else tf.identity,
                name='layer_%i' % (i + 1))
    
    model = tf.keras.Model(inputs=inputs, outputs = x)

    return model


def approximate_errors(model, xi, u_reference, mc_rounds):
    '''
    This function calculate the errors of approximation

    Parameters:
    -----------------------------------------------------------
        model: 
            The tensorflow neural model
        xi: 
            the training data
        u_reference: 
            True solution for evaluation purpose
        mc_rounds: 
            Do Monte-Carlo Methods mc_rounds times.

    Returns:
    -----------------------------------------------------------
        l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err
    '''

    # Calculate errors over multiple Monte Carlo rounds
    l1_err, l2_err, li_err = 0., 0., 0.
    rel_l1_err, rel_l2_err, rel_li_err = 0., 0., 0.
    for _ in range(mc_rounds):
        u_approx = model(xi, training=False)
        err = tf.abs(u_approx - u_reference)
        l1_err += tf.reduce_mean(err)
        l2_err += tf.reduce_mean(err ** 2)
        li_err = max(li_err, tf.reduce_max(err))

        rel_err = err / tf.maximum(u_reference, 1e-8)
        rel_l1_err += tf.reduce_mean(rel_err)
        rel_l2_err += tf.reduce_mean(rel_err ** 2)
        rel_li_err = max(rel_li_err, tf.reduce_max(rel_err))
        

    l1_err /= mc_rounds
    l2_err = np.sqrt(l2_err / mc_rounds)
    rel_l1_err /= mc_rounds
    rel_l2_err = np.sqrt(rel_l2_err / mc_rounds)

    return err, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err


def kolmogorov_train_and_test(xi, x_sde, 
    phi, u_reference, neurons, 
    lr_boundaries, lr_values, train_steps, 
    mc_rounds, mc_freq, 
    file_name, dtype=tf.float32):
    '''
    Kolmogorov_train_and_test main function
    '''

    input_shape = xi.shape[1]
    model = neural_net(input_shape, neurons)
    
    # Define custom loss function
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean((y_pred - y_true) ** 2)

    # Compile the model
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries, values=lr_values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss=custom_loss)
    
    # Open the file for writing results
    with open(file_name, 'w') as file_out:
        file_out.write('step,err,l1_err,l2_err,li_err,l1_rel,l2_rel,li_rel,learning_rate,time_train,time_mc\n')
        
        for epoch in range(train_steps):

            start_time = time.time()

            with tf.GradientTape() as tape:
                # Forward Propagation
                u_approx = model(xi, training=True)
                loss = custom_loss(u_approx, phi(x_sde))

            # Calculate Gradient
            grads = tape.gradient(loss, model.trainable_variables)
            
            # Apply Gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            if epoch % mc_freq == 0:
                t1_training = time.time()
                # Monte Carlo approximation of errors
                err, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err = approximate_errors(model, xi, u_reference, mc_rounds)
                end_time = time.time()

                # Get current learning rate
                current_lr = lr_schedule(optimizer.iterations).numpy()

                print(f"In {epoch} = {rel_li_err}, err = {err}")
                # Write results to file
                file_out.write(f'{epoch}, {tf.reduce_max(err)}, {l1_err}, {l2_err}, {li_err}, {rel_l1_err}, {rel_l2_err}, {rel_li_err}, {current_lr}, {t1_training - start_time}, {end_time - t1_training}\n')
                file_out.flush()