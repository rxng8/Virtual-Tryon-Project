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
        self.grid_gen = TPSGridGenerator()

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
        feature1 = tf.transpose(batch_input_1, perm=[0, 2, 1, 3])
        feature1 = tf.reshape(feature1, [b, w * h, c])

        feature2 = tf.reshape(batch_input_2, [b, h * w, c])
        feature2 = tf.transpose(feature2, perm=[0, 2, 1])

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

        # Assuming that we use regular grid
        # Create set of control points (in the original grid)
        axis_coords = tf.linspace(-1.0, 1.0, grid_size)
        self.n_control_points = grid_size * grid_size
        P_Y, P_X = tf.meshgrid(axis_coords, axis_coords)
        P_X = tf.reshape(P_X,(-1,1)) # size (N,1)
        P_Y = tf.reshape(P_Y,(-1,1)) # size (N,1)
        self.P_X_base = P_X.clone()
        self.P_Y_base = P_Y.clone()
        self.Li = tf.expand_dims(self.compute_L_inverse(P_X, P_Y), axis=0)
        self.P_X = tf.transpose(tf.reshape(P_X, shape=(*P_X.shape, 1, 1, 1)), perm=[1,1,1,1,self.n_control_points])
        self.P_Y = tf.transpose(tf.reshape(P_Y, shape=(*P_Y.shape, 1, 1, 1)), perm=[1,1,1,1,self.n_control_points])
    
    def call(self, batch_input):

        return None

    # This code's logic is derived from the original CP-VTON github
    def compute_L_inverse(self, X, Y):
        # X and Y here is the column vector of the original control points
        # X and Y shape (N, 1)
        N = X.shape[0]

        # Construct K
        X_mat = tf.broadcast_to(X, shape=(N, N))
        Y_mat = tf.broadcast_to(Y, shape=(N, N))

        P_dist_squared = (X_mat - tf.transpose(X_mat)) ** 2  +  (Y_mat - tf.transpose(Y_mat)) ** 2
        P_dist_squared[P_dist_squared == 0] = 1

        K = P_dist_squared * tf.math.log(P_dist_squared)

        # construct matrix L
        O = tf.ones_like(X)
        Z = tf.zeros((3, 3))     
        P = tf.concat([O, X, Y], axis=1)
        L = tf.concat(
            [
                tf.concat([K, P], axis=1), 
                tf.concat([tf.tranpose(P), Z], axis=1)
            ], 
            axis=0
        )
        Li = tf.linalg.inv(L)
        return Li

    def apply_transformation(self, theta, points):
        pass