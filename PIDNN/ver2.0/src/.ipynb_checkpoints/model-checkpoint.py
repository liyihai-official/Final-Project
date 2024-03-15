from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.nn import tanh
from tensorflow import identity, matmul


class dense(Layer):
    def __init__(self, num_outputs):
        super(dense, self).__init__(name='Dense')
        self.num_outputs = num_outputs
        self.bn = BatchNormalization(momentum=0.1)
        
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                     shape = [int(input_shape[-1]),
                                             self.num_outputs])
        
    def call(self, inputs):
        return  self.bn(matmul(inputs, self.kernel))

class My_fnn(Model):
    def __init__(self, neurons):
        super().__init__()
        self.activations = []
        self.custom_layers = []
        self.neurons = neurons

        for i, neu in enumerate(neurons):
            layer = dense(neu)
            activation = tanh if i < len(neurons) - 1 else identity
            self.activations.append(activation)
            self.custom_layers.append(layer)
        self.bn = BatchNormalization(momentum=0.1)
    
    def call(self, input_tensor, training=False):
        x = self.bn(input_tensor)
        for i in range(len(self.neurons)):
            x = self.custom_layers[i](x)
            x = self.activations[i](x)
        return x