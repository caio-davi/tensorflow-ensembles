import tensorflow as tf
from swarm_nn import utils


class fss:
    def __init__(
        self,
        loss_op,
        layer_sizes,
        iter=2000,
        pop_size=30,
        w_scale=100,
        stepInd=0.01,
        stepVol=0.01,
        x_min=-1,
        x_max=1,
        xavier_init=False,
        verbose=False,
    ):
        self.loss_op = loss_op
        self.layer_sizes = layer_sizes
        self.iter = iter
        self.pop_size = pop_size
        self.stepInd = stepInd
        self.stepIndDecay = self.stepInd / self.iter
        self.stepVol = stepVol
        self.stepVolDecay = self.stepVol / self.iter
        self.w_scale = w_scale
        self.x_min = x_min
        self.x_max = x_max
        self.xavier_init = xavier_init
        self.dim = utils.dimensions(layer_sizes)
        self.X, self.w = self.make_pop()
        self.f_X = self.f(self.X)
        self.verbose = verbose

    def individual_fn(self, particle):
        w, b = utils.decode(particle, self.layer_sizes)
        loss, _ = self.loss_op(w, b)
        return -loss

    @tf.function
    def f(self, x):
        f_x = tf.vectorized_map(self.individual_fn, x)
        return f_x[:, None]

    def individual(self):
        step = (
            tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
            * self.stepInd
        )
        x1 = tf.add(self.X, step)
        f_x1 = self.f(x1)
        x1 = tf.where(f_x1 > self.f_X, x1, self.X)
        f_x1 = tf.where(f_x1 > self.f_X, f_x1, self.f_X)
        step = tf.where(f_x1 > self.f_X, step, tf.zeros([self.pop_size, self.dim]))
        return x1, step, f_x1

    def instictive(self, x1, step, f_x1):
        self.X = tf.add(x1, step)
        self.f_X = self.f(self.X)

    def bari(self):
        den = tf.reduce_sum(tf.multiply(self.w[:, None], self.X), 0)
        return den / tf.reduce_sum(self.w)

    def feed(self, x1):
        f_x1 = self.f(x1)
        df = tf.add(f_x1, -self.f_X)
        df_mean = df / tf.reduce_max(df)
        return tf.add(self.w, tf.reshape(df_mean, [self.pop_size])), f_x1

    def volitive(self):
        rand = tf.scalar_mul(self.stepVol, tf.random.uniform([self.pop_size, 1], 0, 1))
        bari_vector = utils.replacenan(tf.add(self.X, -self.bari()))
        step = tf.multiply(rand, bari_vector)
        x_contract = tf.add(self.X, -step)
        x_expand = tf.add(self.X, step)
        w1, f_x_contract = self.feed(x_contract)
        self.X = tf.where(w1[:, None] > self.w[:, None], x_contract, x_expand)
        self.f_X = tf.where(w1[:, None] > self.w[:, None], f_x_contract, self.f_X)
        self.w = w1

    def make_pop(self):
        if self.xavier_init:
            X = self._make_pop_NN()
        else:
            X = tf.Variable(
                tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
            )
        w = tf.Variable([self.w_scale / 2] * self.pop_size)
        return X, w

    def _make_pop_NN(self):
        xavier_init_nns = []
        for i in range(self.pop_size):
            w, b = utils.initialize_NN(self.layer_sizes)
            new_nn = utils.encode(w, b)
            xavier_init_nns.append(new_nn)
        return tf.Variable(xavier_init_nns, dtype=tf.float32)

    def update_steps(self):
        self.stepInd = self.stepInd - self.stepIndDecay
        self.stepVol = self.stepVol - self.stepVolDecay

    def train(self):
        for i in range(self.iter):
            x1, step, f_x1 = self.individual()
            self.instictive(x1, step, f_x1)
            self.volitive()
            self.update_steps()
            if self.verbose and i % (self.iter / 10) == 0:
                utils.progress((i / self.iter) * 100)
        if self.verbose:
            utils.progress(100)

    def get_best(self):
        return utils.decode(
            tf.unstack(self.X)[tf.math.argmax(tf.reshape(self.f_X, [self.pop_size]))],
            self.layer_sizes,
        )

    def get_swarm(self):
        return self.X
