# Geometric matching module

import tensorflow as tf
from tf.keras import layers
from tf.keras import Model
from .utils import *
from .stn import spatial_transformer_network, affine_grid_generator, bilinear_sampler

class GMM(Model):

    def __init__(self):
        super().__init__()
        self.extractorA = FeatureExtractor()
        self.extractorB = FeatureExtractor()
        self.correlator = FeatureCorrelator()
        self.regressor = FeatureRegressor()
        self.grid_gen = GridTTPSGridGeneratorransformer()

    def call(self, batch_inputs):
        return None

class FeatureExtractor(Model):
    def __init__(self, starting_out_channels=64, n_down_layers=4):
        super().__init__()
        models = []

        for i in range(n_down_layers):
            out_channels = starting_out_channels * (2 ** (i + 1))
            models += [make_down_conv_sequence(out_channels)]
        
        models += make_conv_layer(512)
        models += make_dropout_layer()
        models += make_conv_layer(512)
        
        self.model = tf.keras.Sequential(models)
        
    def call(self, batch_inputs):
        return self.model(batch_inputs)

class FeatureL2Norm(Model):
    # TODO: HAve not done yet
    def __init__(self):
        super().__init__()

    def call(self, batch_input):
        epsilon = 1e-6
        norm = (batch_input ** 2 + 1 + epsilon ) ** 0.5
        # norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return batch_input / norm

class FeatureCorrelator(Model):
    def __init__(self):
        super().__init__()
        
    def call(self, batch_input_1, batch_input_2):
        # batch_input shape (b, h, w, c)
        assert batch_input_1.shape == batch_input_2.shape
        b, h, w, c = batch_input_1.shape
        feature1 = tf.transpose(batch_input_1, perm=[b, w, h, c])
        feature1 = tf.reshape(feature1, [b, w * h, c])

        feature2 = tf.reshape(batch_input_2, [b, h * w, c])
        feature2 = tf.transpose(feature2, perm=[b, c, h * w])

        # Batch matrix multiplication
        # correlation_tensor shape (b, h * w, h * w)
        correlation_tensor = feature1 @ feature2
        correlation_tensor = tf.reshape(correlation_tensor, [b, h, w, h * w])
        return correlation_tensor

class FeatureRegressor(Model):
    def __init__(self, output_theta_dim=6):
        super().__init__()
        self.conv = tf.keras.Sequential(
            [
                make_down_conv_sequence(512),
                make_down_conv_sequence(256),
                make_conv_layer(128),
                make_conv_layer(64)
            ], 
            name="regressor_sequence"
        )
        self.dense = make_dense_layer(output_theta_dim, activation='tanh')
        
    def call(self, batch_input):
        tensor = self.conv(batch_input)
        tensor = tf.keras.layers.Flatten()(tensor)
        out = self.dense(tensor)
        return out

class TPSGridGenerator(Model):
    def __init__(self, out_h=256, out_w=192, grid_size=3, reg_factor=0):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))

    def call(self, batch_inputs):

        return None
