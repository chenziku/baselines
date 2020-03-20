import os

import cloudpickle
import numpy as np
import tensorflow as tf

import numpy as np
import os
import json

def normalize(data):
    return data/255.0


def denormalize(data):
    return data * 255.0


class ConvVAE(object):
  def __init__(self, z_size=512, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, is_training=True, reuse=False, gpu_mode=True):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance
    self.reuse = reuse
    with tf.variable_scope('conv_vae', reuse=self.reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self._build_graph()
      else:
        tf.logging.info('Model using gpu.')
        self._build_graph()
    self._init_session()

  def _build_graph(self):
    self.g = tf.Graph()
    with self.g.as_default():

        self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

        # # Encoder
        # h = tf.layers.conv2d(self.input_tensor, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
        # h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
        # h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
        # h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
        # # h = tf.reshape(h, [-1, 3 * 8 * 256])
        # h = conv_to_fc(h)

        # # VAE
        # self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
        # self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
        # self.sigma = tf.exp(self.logvar / 2.0)
        # self.epsilon = tf.random_normal([self.batch_size, self.z_size])
        # # self.epsilon = tf.random_normal([None, self.z_size])
        # # self.z = self.mu + self.sigma * self.epsilon
        # if self.is_training:
        #     self.z = self.mu + self.sigma * self.epsilon
        # else:
        #     self.z = self.mu

        # # Decoder
        # # h = tf.layers.dense(self.z, 3 * 8 * 256, name="dec_fc")
        # # h = tf.reshape(h, [-1, 3, 8, 256])
        # h = tf.layers.dense(self.z, 3 * 3 * 256, name="dec_fc")
        # h = tf.reshape(h, [-1, 3, 3, 256])
        # h = tf.layers.conv2d_transpose(h, 128, 3, strides=2, activation=tf.nn.relu, name="dec_deconv1")
        # h = tf.layers.conv2d_transpose(h, 64, 3, strides=2, activation=tf.nn.relu, name="dec_deconv2")
        # h = tf.layers.conv2d_transpose(h, 32, 4, strides=2, activation=tf.nn.relu, name="dec_deconv3")
        # self.output_tensor = tf.layers.conv2d_transpose(h, 3, 2, strides=2, activation=tf.nn.sigmoid,
        #                                                 name="dec_deconv4")


        # Encoder
        h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
        h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
        h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
        h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
        h = tf.reshape(h, [-1, 3*8*256])

        # VAE
        self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
        self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
        self.sigma = tf.exp(self.logvar / 2.0)
        self.epsilon = tf.random_normal([self.batch_size, self.z_size])
        self.z = self.mu + self.sigma * self.epsilon

        # Decoder
        h = tf.layers.dense(self.z, 3*8*256, name="dec_fc")
        h = tf.reshape(h, [-1, 3, 8, 256])
        h = tf.layers.conv2d_transpose(h, 128, 4, strides=2, activation=tf.nn.relu, name="dec_deconv1")
        h = tf.layers.conv2d_transpose(h, 64, 4, strides=2, activation=tf.nn.relu, name="dec_deconv2")
        h = tf.layers.conv2d_transpose(h, 32, 5, strides=2, activation=tf.nn.relu, name="dec_deconv3")
        self.y = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")

        # train ops
        if self.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            eps = 1e-6 # avoid taking log of zero

            # reconstruction loss
            self.r_loss = tf.reduce_sum(
              tf.square(self.x - self.y),
              reduction_indices = [1,2,3]
            )
            self.r_loss = tf.reduce_mean(self.r_loss)

            # augmented kl loss per dim
            self.kl_loss = - 0.5 * tf.reduce_sum(
              (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
              reduction_indices = 1
            )
            self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
            self.kl_loss = tf.reduce_mean(self.kl_loss)

            self.loss = self.r_loss + self.kl_loss

            # training
            self.lr = tf.Variable(self.learning_rate, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')

        # initialize vars
        self.init = tf.global_variables_initializer()

  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)

  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()

  def encode(self, x):
    return self.sess.run(self.z, feed_dict={self.x: x})

  def decode(self, z):
    return self.sess.run(self.y, feed_dict={self.z: z})

  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names

  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up!
    return rparam

  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = self.sess.run(var).shape
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op = var.assign(p.astype(np.float)/10000.)
        self.sess.run(assign_op)
        idx += 1

  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)

  def save_json(self, jsonfile='vae.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)

  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vae')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one

  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)





class VAEController:
    def __init__(self, z_size=512, image_size=(80, 160, 3),
                 learning_rate=0.0001, kl_tolerance=0.5,
                 epoch_per_optimization=10, batch_size=64,
                 buffer_size=500):
        # VAE input and output shapes
        self.z_size = z_size
        self.image_size = image_size

        # VAE params
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance

        # Training params
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size

        # Buffer
        self.buffer_size = buffer_size
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_reset()

        self.vae = ConvVAE(z_size=self.z_size,
                           batch_size=self.batch_size,
                           learning_rate=self.learning_rate,
                           kl_tolerance=self.kl_tolerance,
                           is_training=True,
                           reuse=False,
                           gpu_mode=True)

        self.target_vae = ConvVAE(z_size=self.z_size,
                                  batch_size=1,
                                  is_training=False,
                                  reuse=False,
                                  gpu_mode=True)

    def buffer_append(self, arr):
        assert arr.shape == self.image_size
        self.buffer_pos += 1
        if self.buffer_pos > self.buffer_size - 1:
            self.buffer_pos = 0
            self.buffer_full = True
        self.buffer[self.buffer_pos] = arr

    def buffer_reset(self):
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer = np.zeros((self.buffer_size,
                                self.image_size[0],
                                self.image_size[1],
                                self.image_size[2]),
                               dtype=np.uint8)

    def buffer_get_copy(self):
        if self.buffer_full:
            return self.buffer.copy()
        return self.buffer[:self.buffer_pos]

    def encode(self, arr):
        assert arr.shape == self.image_size
        # Normalize
        arr = arr.astype(np.float)/255.0
        # Reshape
        arr = arr.reshape(1,
                          self.image_size[0],
                          self.image_size[1],
                          self.image_size[2])
        return self.target_vae.encode(arr)

    def decode(self, arr):
        assert arr.shape == (1, self.z_size)
        # Decode
        arr = self.target_vae.decode(arr)
        # Denormalize
        arr = arr * 255.0
        return arr

    def optimize(self):
        ds = self.buffer_get_copy()
        # TODO: may be do buffer reset.
        # self.buffer_reset()

        num_batches = int(np.floor(len(ds)/self.batch_size))

        for epoch in range(self.epoch_per_optimization):
            np.random.shuffle(ds)
            for idx in range(num_batches):
                batch = ds[idx * self.batch_size:(idx + 1) * self.batch_size]
                obs = batch.astype(np.float) / 255.0
                feed = {self.vae.x: obs, }
                (train_loss, r_loss, kl_loss, train_step, _) = self.vae.sess.run([
                    self.vae.loss,
                    self.vae.r_loss,
                    self.vae.kl_loss,
                    self.vae.global_step,
                    self.vae.train_op
                ], feed)
                if ((train_step + 1) % 50 == 0):
                    print("VAE: optimization step",
                          (train_step + 1), train_loss, r_loss, kl_loss)
        self.set_target_params()

    def save(self, path):
        self.target_vae.save_json(path)

    def load(self, path):
        self.target_vae.load_json(path)

    def set_target_params(self):
        params, _, _ = self.vae.get_model_params()
        self.target_vae.set_model_params(params)
