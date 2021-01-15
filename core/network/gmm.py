#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Author: Alex Nguyen
    Gettysburg College
    gmm.py: This file contains models of geometric matching. The code logic in
    this module is derived from here:
        https://github.com/sergeywong/cp-vton/blob/master/networks.py
"""

# Geometric matching module

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np

from .utils import *
from .stn import affine_grid_generator, bilinear_sampler

class GMMTPS(Model):

    def __init__(self, batch_size=16, grid_size=5):
        super().__init__()
        self.batch_size = batch_size
        self.grid_size = grid_size
        
        self.extractorA = FeatureExtractor()
        self.extractorB = FeatureExtractor()
        self.normalizer = FeatureL2Norm()
        self.correlator = FeatureCorrelator()
        self.regressor = FeatureRegressor(output_theta_dim=2*grid_size*grid_size)
        self.grid_gen = TPSGridGenerator(grid_size=grid_size)
        self.transformer = BilinearSampler()

    def call(self, batch_input_image, batch_input_cloth):

        image_tensor = self.extractorA(batch_input_image)
        image_tensor = self.normalizer(image_tensor)
        # print("Done image_tensor")

        cloth_tensor = self.extractorB(batch_input_cloth)
        cloth_tensor = self.normalizer(cloth_tensor)
        # print("Done cloth_tensor ")

        correlation_tensor = self.correlator(image_tensor, cloth_tensor)
        # print("Done correlation_tensor ")

        theta_tensor = self.regressor(correlation_tensor)
        # print("Done theta_tensor ")

        tps_grid = self.grid_gen(theta_tensor)
        # print("Done tps_grid ")

        x_s = tps_grid[:, :, :, 0]
        y_s = tps_grid[:, :, :, 1]

        transformed = self.transformer(batch_input_cloth, x_s, y_s)
        # print("Done transformed ")

        return theta_tensor, tps_grid, transformed

class GMMSTN(Model):

    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.extractorA = FeatureExtractor()
        self.extractorB = FeatureExtractor()
        self.normalizer = FeatureL2Norm()
        self.correlator = FeatureCorrelator()
        self.regressor = FeatureRegressor()
        self.grid_gen = AffineGridGenerator()
        self.transformer = BilinearSampler()

    def call(self, batch_input_image, batch_input_cloth):

        image_tensor = self.extractorA(batch_input_image)
        image_tensor = self.normalizer(image_tensor)
        # print("Done image_tensor")

        cloth_tensor = self.extractorB(batch_input_cloth)
        cloth_tensor = self.normalizer(cloth_tensor)
        # print("Done cloth_tensor ")

        correlation_tensor = self.correlator(image_tensor, cloth_tensor)
        # print("Done correlation_tensor ")

        theta_tensor = self.regressor(correlation_tensor)
        theta_tensor = tf.reshape(theta_tensor, [self.batch_size, 2, 3])
        # print("Done theta_tensor ")

        affine_grid = self.grid_gen(theta_tensor)
        # print("Done affine_grid ")

        x_s = affine_grid[:, 0, :, :]
        y_s = affine_grid[:, 1, :, :]

        transformed = self.transformer(batch_input_cloth, x_s, y_s)
        # print("Done transformed ")

        return theta_tensor, affine_grid, transformed


class SimpleGMM(Model):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.extractorA = FeatureExtractor()
        self.extractorB = FeatureExtractor()
        self.normalizer = FeatureL2Norm()
        self.correlator = FeatureCorrelator()
        self.generator = ImageRegenerator()

    def call(self, batch_input_image, batch_input_cloth):
        image_tensor = self.extractorA(batch_input_image)
        image_tensor = self.normalizer(image_tensor)
        # print(f"Done image_tensor: {image_tensor.shape}")

        cloth_tensor = self.extractorB(batch_input_cloth)
        cloth_tensor = self.normalizer(cloth_tensor)
        # print(f"Done cloth_tensor: {cloth_tensor.shape} ")

        correlation_tensor = self.correlator(image_tensor, cloth_tensor)
        # print(f"Done correlation_tensor {correlation_tensor.shape} ")

        out = self.generator(correlation_tensor)
        # print(f"Done out {out.shape} ")
        return out

class FeatureExtractor(Model):
    def __init__(self, starting_out_channels=64, n_down_layers=4):
        super().__init__()
        models = []

        for i in range(n_down_layers):
            out_channels = starting_out_channels * (2 ** (i + 1))
            models += [make_down_conv_sequence(out_channels)]
        
        models.append(make_conv_layer(512))
        models.append(make_dropout_layer())
        models.append(make_conv_layer(512))
        
        self.model = tf.keras.Sequential(models)
        
    def call(self, batch_inputs):
        return self.model(batch_inputs)

class FeatureL2Norm(Model):
    def __init__(self):
        super().__init__()

    def call(self, batch_inputs):
        # Normalize channel
        # Expect batch_inputs shape (B, H, W, C)
        epsilon = 1e-6
        norm = (tf.sum(batch_inputs ** 2, axis=3) + epsilon ) ** 0.5
        norm = tf.expand_dims(norm, axis=3)
        norm = tf.broadcast_to(norm, shape=batch_inputs.shape)
        return batch_inputs / norm

class ImageRegenerator(Model):
    def __init__(self, starting_out_channels=64, n_up_layers=4):
        super().__init__()
        models = []

        for i in range(n_up_layers - 1, -1, -1):
            out_channels = starting_out_channels * (2 ** i)
            models += [make_up_conv_layer(out_channels)]
        
        models.append(make_deconv_layer(1, activation="sigmoid"))
        
        self.model = tf.keras.Sequential(models)
        

    def call(self, batch_inputs):
        return self.model(batch_inputs)

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
        
    def call(self, batch_inputs):
        tensor = self.conv(batch_inputs)
        tensor = tf.keras.layers.Flatten()(tensor)
        out = self.dense(tensor)
        return out

class BilinearSampler(Model):
    def __init__(self):
        super().__init__()

    def call(self, batch_inputs, x_s, y_s):
        # Expect x_s = batch_grids[:, 0, :, :]
        #        y_s = batch_grids[:, 1, :, :] returned from affine grid gen
        return bilinear_sampler(batch_inputs, x_s, y_s)

class AffineGridGenerator(Model):
    def __init__(self, out_h=256, out_w=192):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w

    def call(self, batch_theta):
        return affine_grid_generator(self.out_h, self.out_w, batch_theta)
    

class TPSGridGenerator(Model):
    def __init__(self, out_h=256, out_w=192, grid_size=5, reg_factor=0):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))

        # Assuming that we use regular grid
        # Create set of control points (in the original grid)
        axis_coords = tf.linspace(-1.0, 1.0, grid_size)
        self.n_control_points = grid_size * grid_size
        P_Y, P_X = tf.meshgrid(axis_coords, axis_coords)
        P_X = tf.reshape(P_X,(-1,1)) # size (N,1)
        P_Y = tf.reshape(P_Y,(-1,1)) # size (N,1)

        # Column vector of original control points axis x
        self.P_X_base = P_X.clone()
        # Column vector of original control points axis x
        self.P_Y_base = P_Y.clone()

        self.Li = tf.expand_dims(self.compute_L_inverse(P_X, P_Y), axis=0)
        # Shape after (1, 1, 1, 1, n_control_points) <=> (B, H, W, C, n_control_points)
        self.P_X = tf.transpose(tf.reshape(P_X, shape=(*P_X.shape, 1, 1, 1)), perm=[4, 1, 2, 3, 0])
        self.P_Y = tf.transpose(tf.reshape(P_Y, shape=(*P_Y.shape, 1, 1, 1)), perm=[4, 1, 2, 3, 0])
    
    def call(self, theta):
        warped_grid = self.apply_transformation(theta, tf.concat([self.grid_X,self.grid_Y], axis=3))
        return warped_grid

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
        # Expected theta to be shape (B, kernels)

        # Expects points to be the original meshgrid of the images
        # Shape (B, W, H, 2). With:
        #   points(B, W, H, 0) is the x coord of the meshgrid, and
        #   points(B, W, H, 1) is the y coord of the meshgrid.
        
        # Theta is the param to compute target control points.
        # Theta shape (B, 2 * grid_size * grid_size, 1, 1)
        if len(theta.shape) == 2:
            theta = tf.reshape(theta, (*theta.shape, 1, 1))

        # input are the corresponding control points P_i, the target control points Q_i
        # Shape (B, grid_size^2, 1) == (B, n_control_points, 1)
        Q_X = tf.squeeze(theta[:,:self.n_control_points,:,:], [3])
        Q_Y = tf.squeeze(theta[:,self.n_control_points:,:,:], [3])

        Q_X = Q_X + tf.broadcast_to(self.P_X_base, shape=Q_X.shape)
        Q_Y = Q_Y + tf.broadcast_to(self.P_Y_base, shape=Q_Y.shape)

        # get spatial dimensions of points
        n_batch = points.shape[0]
        n_height = points.shape[1]
        n_width = points.shape[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = tf.broadcast_to(self.P_X, shape=(1,n_height,n_width,1,self.n_control_points))
        P_Y = tf.broadcast_to(self.P_Y, shape=(1,n_height,n_width,1,self.n_control_points))

        # compute weigths for non-linear part
        # Shape (B, N, N) x (B, N, 1) = (B, N, 1)
        W_X = tf.broadcast_to(
            self.Li[:, :self.n_control_points, :self.n_control_points], 
            shape=(n_batch, *self.Li.shape[1:])
        ) @ Q_X
        W_Y = tf.broadcast_to(
            self.Li[:, :self.n_control_points, :self.n_control_points], 
            shape=(n_batch, *self.Li.shape[1:])
        ) @ Q_Y

        # reshape
        # W_X, W_Y: size [B,N,1,1,1]
        W_X = tf.reshape(W_X, shape=(*W_X.shape, 1, 1))
        # W_X, W_Y: size [B,1,1,1,N]
        W_X = tf.transpose(W_X, perm=[0, 4, 2, 3, 1])
        # W_X, W_Y: size [B,H,W,1,N]
        W_X = tf.tile(W_X, (1, n_height, n_width, 1, 1))

        # W_X, W_Y: size [B,N,1,1,1]
        W_Y = tf.reshape(W_Y, shape=(*W_Y.shape, 1, 1))
        # W_X, W_Y: size [B,1,1,1,N]
        W_Y = tf.transpose(W_Y, perm=[0, 4, 2, 3, 1])
        # W_X, W_Y: size [B,H,W,1,N]
        W_Y = tf.tile(W_Y, (1, n_height, n_width, 1, 1))

        # compute weights for affine part (The bottom left part of the Li matrix)
        A_X = tf.broadcast_to(
            self.Li[:, self.n_control_points:, :self.n_control_points], 
            shape=(n_batch, 3, self.n_control_points)
        ) @ Q_X

        A_Y = tf.broadcast_to(
            self.Li[:, self.n_control_points:, :self.n_control_points], 
            shape=(n_batch, 3, self.n_control_points)
        ) @ Q_Y

        # reshape (similar to reshape above)
        # A_X, A_Y: size [B,H,W,1,3]
        A_X = tf.reshape(A_X, shape=(*A_X.shape, 1, 1))
        A_X = tf.transpose(A_X, perm=[0, 4, 2, 3, 1])
        A_X = tf.tile(A_X, (1, n_height, n_width, 1, 1))

        A_Y = tf.reshape(A_Y, shape=(*A_Y.shape, 1, 1))
        A_Y = tf.transpose(A_Y, perm=[0, 4, 2, 3, 1])
        A_Y = tf.tile(A_Y, (1, n_height, n_width, 1, 1))


        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = tf.reshape(points[:,:,:,0], shape=(*points.shape[:-1], 1, 1))
        points_X_for_summation = tf.broadcast_to(
            points_X_for_summation, 
            shape=(*points.shape[:-1], 1, self.n_control_points)
        )

        points_Y_for_summation = tf.reshape(points[:,:,:,1], shape=(*points.shape[:-1], 1, 1))
        points_Y_for_summation = tf.broadcast_to(
            points_Y_for_summation, 
            shape=(*points.shape[:-1], 1, self.n_control_points)
        )

        # use expanded P_X,P_Y in batch dimension
        delta_X = points_X_for_summation - tf.broadcast_to(P_X, shape=points_X_for_summation.shape)
        delta_Y = points_Y_for_summation - tf.broadcast_to(P_Y, shape=points_Y_for_summation.shape)

        dist_squared = delta_X ** 2 + delta_Y ** 2

        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0] = 1 # avoid NaN in log computation
        U = dist_squared * tf.log(dist_squared) 

        # expand grid in batch dimension if necessary
        points_X_batch = tf.expand_dims(points[:,:,:,0], axis=3)
        points_Y_batch = tf.expand_dims(points[:,:,:,1], axis=3)

        points_X_prime = A_X[:,:,:,:,0]+ \
                       (A_X[:,:,:,:,1] * points_X_batch) + \
                       (A_X[:,:,:,:,2] * points_Y_batch) + \
                       tf.sum(W_X * tf.broadcast_to(U, shape=W_X.shape), axis=4)

        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       (A_Y[:,:,:,:,1] * points_X_batch) + \
                       (A_Y[:,:,:,:,2] * points_Y_batch) + \
                       tf.sum(W_Y * tf.broadcast_to(U, shape=W_Y.shape), axis=4)

        return tf.concat([points_X_prime,points_Y_prime], axis=3)