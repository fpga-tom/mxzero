# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections
import functools
import math
import typing
from typing import Dict, List, Optional

import numpy
import mmap
from absl import flags
import pickle
import uuid
import os
import time
import string

from mxnet import nd, metric, optimizer as opt, lr_scheduler, gpu, cpu, autograd
from mxnet.gluon import nn, loss, Trainer

flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', 16,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))

flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')

flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')


# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


def get_ctx():
    return cpu(0)

def get_train_ctx():
    return gpu(0)

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    # print(value)
    # print(numpy.shape(value))
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


class MuZeroConfig(object):

  def __init__(self,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               num_actors: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = int(10)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

  def new_game(self):
    return Game(self.action_space_size, self.discount)


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 30:
      return 1.0
    else:
      return 0.0  # Play according to the max.

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=max_moves,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=100,
      batch_size=128,
      td_steps=max_moves,  # Always use Monte Carlo return.
      num_actors=8,
      lr_init=lr_init,
      lr_decay_steps=400e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      known_bounds=KnownBounds(-1, 1))


def make_go_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


def make_chess_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


def make_shogi_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)


def make_c_config() -> MuZeroConfig:
  return make_board_game_config(
    action_space_size=Environment.ACTIONS, max_moves=16, dirichlet_alpha=0.03, lr_init=0.01)

def make_atari_config() -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 500e3:
      return 1.0
    elif training_steps < 750e3:
      return 0.5
    else:
      return 0.25

  return MuZeroConfig(
      action_space_size=18,
      max_moves=27000,  # Half an hour at action repeat 4.
      discount=0.997,
      dirichlet_alpha=0.25,
      num_simulations=50,
      batch_size=1024,
      td_steps=10,
      num_actors=350,
      lr_init=0.05,
      lr_decay_steps=350e3,
      visit_softmax_temperature_fn=visit_softmax_temperature)


class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

  def __str__(self):
    return str(self.index)


class Player(object):
  pass


class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()


class Environment(object):
  """The environment MuZero is interacting with."""

  ACTIONS = len(string.printable)
  HISTORY = 8
  N = 16

  def __init__(self):
    self.f = open('example.txt', 'rb', 0)
    self.s = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
    self.seq = []

  def __getstate__(self):
      state = dict(self.__dict__)
      del state['f']
      del state['s']
      return state

  def step(self, action):
    self.seq.append(action)
    w = ''.join([string.printable[c] for c in self.seq])
    print(self.seq, w)
    if self.s.find(str.encode(w)) != -1:
      print('reward' , 1)
      return 1
    else:
      print('reward', -1)
      return -1

  @staticmethod
  def get_actions():
    return [Action(i) for i in range(Environment.ACTIONS)]


class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float):
    self.environment = Environment()  # Game specific environment.
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    if len(self.rewards) > 0:
        return self.rewards[-1] == -1
    return False

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return self.environment.get_actions()

  def apply(self, action: Action):
    reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    img = self.environment.seq[state_index - Environment.HISTORY:state_index]
    return numpy.reshape(numpy.pad(img, (Environment.HISTORY - len(img), 0), 'constant', constant_values=0),
                         [-1, 1, Environment.HISTORY])

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      if current_index < len(self.root_values):
        targets.append((value, self.rewards[current_index],
                        self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, 0, []))
    return targets

  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)

  def save(self, filename):
      with open('replay_buffer/' + filename, 'wb') as f:
          pickle.dump(self, f)

  @staticmethod
  def load(filename):
      with open('replay_buffer/' + filename, 'rb') as f:
          return pickle.load(f)


class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)
    game.save(str(uuid.uuid1()))


  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    # print(len(self.buffer))
    while not os.listdir('replay_buffer'):
        time.sleep(5)
    f = [files for root, dir, files in os.walk('replay_buffer')][0]
    f = numpy.random.choice(f)
    return Game.load(f)
    # return self.buffer[numpy.random.randint(0, len(self.buffer))]

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return numpy.random.randint(0, len(game.history))


# class NetworkOutput(typing.NamedTuple):
#   value: float
#   reward: float
#   policy_logits: Dict[Action, float]
#   hidden_state: List[float]

NetworkOutput = typing.NamedTuple('NetworkOutput', [('value', float),
                                                    ('reward', float),
                                                    ('policy_logits', Dict[Action, float]),
                                                    ('hidden_state', List[float])])


class Representation(nn.Block):
    def __init__(self, ctx, params, **kwargs):
        super(Representation, self).__init__(**kwargs)
        self.ctx = ctx

        mg_batchn = functools.partial(
            nn.BatchNorm,
            axis=1,
            momentum=.95,
            epsilon=1e-5,
            center=True,
            scale=True)

        mg_conv2d = functools.partial(
            nn.Conv2D,
            channels=params['conv_width'],
            kernel_size=3,
            padding=1,
            layout='NCHW',
            use_bias=False)

        mg_global_avgpool2d = functools.partial(
            nn.AvgPool2D,
            pool_size=Environment.N,
            strides=1,
            padding=0,
            layout='NCHW')

        mg_activation = functools.partial(
            nn.Activation,
            activation='relu'
        )

        def residual_inner():
            net = nn.Sequential()
            net.add(mg_conv2d())
            net.add(mg_batchn())
            net.add(mg_activation())
            net.add(mg_conv2d())
            net.add(mg_batchn())
            return net

        def mg_res_layer():
            class Res(nn.Block):
                def __init__(self, **kwargs):
                    super(Res, self).__init__(**kwargs)
                    self.act = mg_activation()
                    self.r = residual_inner()

                def forward(self, x):
                    return self.act(x + self.r(x))

            return Res()

        initial_block = nn.Sequential()
        with initial_block.name_scope():
            initial_block.add(mg_conv2d())
            initial_block.add(mg_batchn())
            initial_block.add(mg_activation())

        # the shared stack
        shared_output = initial_block
        for _ in range(params['trunk_layers']):
            shared_output.add(mg_res_layer())

        shared_output.add(mg_conv2d(channels=params['policy_conv_width'], kernel_size=1))
        shared_output.add(mg_batchn(center=False, scale=False))
        shared_output.add(mg_activation())
        # shared_output.add(nd.Reshape())
        shared_output.add(nn.Dense(Environment.ACTIONS))

        self.net = shared_output
        self.embedding = nd.random.normal(0, 0.1, shape=[8,Environment.ACTIONS], ctx=self.ctx)

    def forward(self, x):
        return self.net(nd.Embedding(nd.ndarray.array(x, ctx=self.ctx), self.embedding, input_dim=Environment.HISTORY, output_dim=Environment.ACTIONS))


class Prediction(nn.Block):
    def __init__(self, ctx, params, **kwargs):
        super(Prediction, self).__init__(**kwargs)
        self.ctx = ctx

        mg_batchn = functools.partial(
            nn.BatchNorm,
            axis=1,
            momentum=.95,
            epsilon=1e-5,
            center=True,
            scale=True)

        mg_conv2d = functools.partial(
            nn.Conv2D,
            channels=params['conv_width'],
            kernel_size=3,
            padding=1,
            layout='NCHW',
            use_bias=False)

        mg_global_avgpool2d = functools.partial(
            nn.AvgPool2D,
            pool_size=Environment.N,
            strides=1,
            padding=0,
            layout='NCHW')

        mg_activation = functools.partial(
            nn.Activation,
            activation='relu'
        )

        def residual_inner():
            net = nn.Sequential()
            net.add(mg_conv2d())
            net.add(mg_batchn())
            net.add(mg_activation())
            net.add(mg_conv2d())
            net.add(mg_batchn())
            return net

        def mg_res_layer():
            class Res(nn.Block):
                def __init__(self, **kwargs):
                    super(Res, self).__init__(**kwargs)
                    self.act = mg_activation()
                    self.r = residual_inner()

                def forward(self, x):

                    return self.act(x + self.r(x))

            return Res()

        initial_block = nn.Sequential()
        with initial_block.name_scope():
            initial_block.add(mg_conv2d())
            initial_block.add(mg_batchn())
            initial_block.add(mg_activation())

        # the shared stack
        shared_output = initial_block
        for _ in range(params['trunk_layers']):
            shared_output.add(mg_res_layer())

        self.policy = nn.Sequential()
        self.policy.add(shared_output)
        self.policy.add(mg_conv2d(channels=params['policy_conv_width'], kernel_size=1))
        self.policy.add(mg_batchn(center=False, scale=False))
        self.policy.add(mg_activation())
        # shared_output.add(nd.Reshape())
        self.policy.add(nn.Dense(Environment.ACTIONS))

        self.value = nn.Sequential()
        self.value.add(shared_output)
        self.value.add(mg_conv2d(channels=params['value_conv_width'], kernel_size=1))
        self.value.add(mg_batchn(center=False, scale=False))
        self.value.add(mg_activation())
        # shared_output.add(nd.Reshape())
        self.value.add(nn.Dense(params['fc_width']))
        self.value.add(nn.Dense(1))




    def forward(self, x):
        xx = nd.ndarray.array(x, ctx=self.ctx)
        return self.policy(xx), nd.tanh(self.value(xx))


class Dynamics(nn.Block):
    def __init__(self, ctx, params, **kwargs):
        super(Dynamics, self).__init__(**kwargs)
        self.ctx = ctx

        mg_batchn = functools.partial(
            nn.BatchNorm,
            axis=1,
            momentum=.95,
            epsilon=1e-5,
            center=True,
            scale=True)

        mg_conv2d = functools.partial(
            nn.Conv2D,
            channels=params['conv_width'],
            kernel_size=3,
            padding=1,
            layout='NCHW',
            use_bias=False)

        mg_global_avgpool2d = functools.partial(
            nn.AvgPool2D,
            pool_size=Environment.N,
            strides=1,
            padding=0,
            layout='NCHW')

        mg_activation = functools.partial(
            nn.Activation,
            activation='relu'
        )

        def residual_inner():
            net = nn.Sequential()
            net.add(mg_conv2d())
            net.add(mg_batchn())
            net.add(mg_activation())
            net.add(mg_conv2d())
            net.add(mg_batchn())
            return net

        def mg_res_layer():
                class Res(nn.Block):
                    def __init__(self, **kwargs):
                        super(Res, self).__init__(**kwargs)
                        self.act = mg_activation()
                        self.r = residual_inner()

                    def forward(self, x):
                        return self.act(x + self.r(x))

                return Res()

        initial_block = nn.Sequential()
        with initial_block.name_scope():
            initial_block.add(mg_conv2d())
            initial_block.add(mg_batchn())
            initial_block.add(mg_activation())

        # the shared stack
        shared_output = initial_block
        for _ in range(params['trunk_layers']):
            shared_output.add(mg_res_layer())

        self.policy = nn.Sequential()
        self.policy.add(shared_output)
        self.policy.add(mg_conv2d(channels=params['policy_conv_width'], kernel_size=1))
        self.policy.add(mg_batchn(center=False, scale=False))
        self.policy.add(mg_activation())
        # shared_output.add(nd.Reshape())
        self.policy.add(nn.Dense(Environment.ACTIONS))

        self.value = nn.Sequential()
        self.value.add(shared_output)
        self.value.add(mg_conv2d(channels=params['value_conv_width'], kernel_size=1))
        self.value.add(mg_batchn(center=False, scale=False))
        self.value.add(mg_activation())
        # shared_output.add(nd.Reshape())
        self.value.add(nn.Dense(params['fc_width']))
        self.value.add(nn.Dense(1))

        self.embedding = nd.random.normal(0, 0.1, shape=[8, Environment.ACTIONS], ctx=self.ctx)


    def forward(self, x, action):
        emb = nd.Embedding(nd.array([action], self.ctx), self.embedding, input_dim=Environment.HISTORY,
                           output_dim=Environment.ACTIONS)
        emb = nd.Reshape(emb, [1,1,Environment.ACTIONS,1])
        xx = nd.ndarray.array(x, ctx=self.ctx)
        xxc = nd.concat(xx, emb, dim=0)
        return nd.tanh(self.value(xxc)), (self.policy(xxc))


class Network():

    def __init__(self, ctx=get_ctx()):
        self.ctx = ctx
        self.representation = Representation(ctx, FLAGS.flag_values_dict())
        self.dynamics = Dynamics(ctx, FLAGS.flag_values_dict())
        self.prediction = Prediction(ctx, FLAGS.flag_values_dict())
        self.representation.initialize(ctx=ctx)
        self.dynamics.initialize(ctx=ctx)
        self.prediction.initialize(ctx=ctx)
        self.train_steps = 0


    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        hidden_state = self.representation(image)
        # print('image', numpy.shape(image))
        # print('hidden', numpy.shape(hidden_state))
        hidden_state = nd.Reshape(hidden_state, [-1, 1, Environment.ACTIONS, 1])
        policy_output, value_output = self.prediction(hidden_state)
        # policy_output = policy_output.asnumpy()
        # value_output = value_output.asnumpy()
        # print('policy_output', numpy.shape(policy_output), 'value_output', numpy.shape(value_output))
        return NetworkOutput(value_output[0][0], nd.array([0], ctx=self.ctx), {a: policy_output[0][a.index] for a in Environment.get_actions()},
                             hidden_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        reward, hidden_state_new = self.dynamics(hidden_state, action)

        hidden_state_new = numpy.reshape(hidden_state_new.asnumpy(), [-1, 1, Environment.ACTIONS, 1])
        # print('hidden', numpy.shape(hidden_state_new))
        policy_output, value_output = self.prediction(hidden_state_new)
        # policy_output = policy_output.asnumpy()
        # value_output = value_output.asnumpy()
        # print('policy_output', numpy.shape(policy_output), 'value_output', numpy.shape(value_output), 'reward', numpy.shape(reward))
        return NetworkOutput(value_output[0][0], reward[0][0], {a: policy_output[0][a.index] for a in Environment.get_actions()},
                             hidden_state_new)

    def get_weights(self):
        # Returns the weights of this network.
        return [self.representation.collect_params(), self.dynamics.collect_params(), self.prediction.collect_params()]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.train_steps

    def save_network(self, file_name):
        self.representation.save_parameters('network/' + file_name + '.r.ckpt')
        self.dynamics.save_parameters('network/' + file_name + '.d.ckpt')
        self.prediction.save_parameters('network/' + file_name + '.p.ckpt')

    def load_network(self, file_name):
        self.representation.load_parameters('network/' + file_name + '.r.ckpt', ctx=self.ctx)
        self.dynamics.load_parameters('network/' + file_name + '.d.ckpt', ctx=self.ctx)
        self.prediction.load_parameters('network/' + file_name + '.p.ckpt', ctx=self.ctx)

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if not os.listdir('network'):
        return make_uniform_network()
    r = make_uniform_network()
    r.load_network('latest')
    return r

    # if self._networks:
    #   k = max(self._networks.keys())
    #   network = self._networks[k]
    #   network.load_network('latest')
    # else:
    #   # policy -> uniform, value -> 0, reward -> 0
    #   return make_uniform_network()

  def save_network(self, step: int, network: Network):
    # self._networks[step] = network
    network.save_network('latest')


##### End Helpers ########
##########################


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  p = []
  for i in range(config.num_actors):
    print('job ' , i)
    j = launch_job(run_selfplay, config, storage, replay_buffer, i)
    p.append(j)


  # for i in p:
  #     i.join()

  train_network(config, storage, replay_buffer, ctx=get_train_ctx())



  return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer, tid):
  numpy.random.seed(tid)
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
  game = config.new_game()

  count = 0
  while not game.terminal() and len(game.history) < config.max_moves:
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    current_observation = game.make_image(-1)
    expand_node(root, game.to_play(), game.legal_actions(),
                network.initial_inference(current_observation))
    add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    print('run_mcst ', count)
    count += 1
    run_mcts(config, root, game.action_history(), network)
    action = select_action(config, len(game.history), root, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
  min_max_stats = MinMaxStats(config.known_bounds)

  for sim in range(config.num_simulations):
    # print(sim)
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the dynamics function to obtain the next
    # hidden state given an action and the previous hidden state.
    parent = search_path[-2]
    network_output = network.recurrent_inference(parent.hidden_state,
                                                 history.last_action().index)
    expand_node(node, history.to_play(), history.action_space(), network_output)

    backpropagate(search_path, network_output.value, history.to_play(),
                  config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      num_moves=num_moves, training_steps=network.training_steps())
  _, action = softmax_sample(visit_counts, t)
  return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
  _, action, child = max(
      (ucb_score(config, node, child, min_max_stats), action,
       child) for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = min_max_stats.normalize(child.value())
  return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
  node.reward = network_output.reward
  policy = {a: math.exp(network_output.policy_logits[a].asnumpy()) for a in actions}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
  actions = list(node.children.keys())
  noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer, ctx):
  network = Network(ctx=ctx)
  schedule = lr_scheduler.FactorScheduler(step=int(config.lr_decay_steps), factor=config.lr_decay_rate)
  schedule.base_lr = config.lr_init

  optimizer = opt.SGD(config.lr_init, lr_scheduler=schedule)
  cross_entropy_loss = loss.SoftmaxCrossEntropyLoss(sparse_label=False)
  mse_loss = loss.L2Loss()
  r, d, p = network.get_weights()
  trainer = Trainer(r, optimizer), Trainer(d, optimizer), Trainer(p, optimizer)

  for i in range(config.training_steps):
    # print('training', i)
    if i != 0 and i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(trainer, cross_entropy_loss, mse_loss, network, batch, config.weight_decay, ctx=ctx)
    network.train_steps += 1
  storage.save_network(config.training_steps, network)

def update_weights(optimizer: opt.Optimizer, cross_entropy_loss, mse_loss, network: Network, batch,
                   weight_decay: float, ctx):

  for image, actions, targets in batch:
    with autograd.record():
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(
            image)
        predictions = [(1.0, value, reward, policy_logits)]

        # for prediction, target in zip(predictions, targets[0]):
        # gradient_scale, value, reward, policy_logits = prediction
        # target_value, target_reward, target_policy = targets[0]

        # logits = [policy_logits[a].asnumpy()[0] for a in Environment.get_actions()]
        # print(logits)
        # print(target_policy)
        # l = mse_loss(value, nd.array(target_value, ctx=ctx)) + mse_loss(reward, nd.array(target_reward, ctx=ctx)) + \
        #     cross_entropy_loss(
        #         nd.array(logits, ctx=ctx),
        #         nd.array(target_policy, ctx=ctx))
    #
    # l.backward(retain_graph=True)
    # optimizer[0].step(image.shape[0])

    # print(actions)

        # with autograd.record():
        for i, action in enumerate(actions):
        # Recurrent steps, from action and previous hidden state.
          value, reward, policy_logits, hidden_state = network.recurrent_inference(
              hidden_state, action)
          predictions.append((1.0 / len(actions), value, reward, policy_logits))

          # hidden_state = tf.scale_gradient(hidden_state, 0.5)

        values = None
        target_values = []
        rewards = None
        target_rewards = []
        logits = None
        target_policies = None
        for prediction, target in zip(predictions[:-1], targets[:-1]):
          gradient_scale, value, reward, policy_logits = prediction
          target_value, target_reward, target_policy = target

          if values is not None:
              values = nd.concat(values, value, dim=0)
          else:
              values = value
          target_values.append(target_value)

          if rewards is not None:
              rewards = nd.concat(rewards, reward, dim=0)
          else:
              rewards = reward
          target_rewards.append(target_reward)

          if logits is not None:
            pl = None
            for a in Environment.get_actions():
                if pl is not None:
                    pl = nd.concat(pl, policy_logits[a], dim=0)
                else:
                    pl = policy_logits[a]
            logits = nd.concat(logits, pl, dim=0)
          else:
              pl = None
              for a in Environment.get_actions():
                  if pl is not None:
                      pl = nd.concat(pl, policy_logits[a], dim=0)
                  else:
                      pl = policy_logits[a]
              logits = pl

          if target_policies is not None:
              target_policies = nd.concat(target_policies, nd.array(target_policy, ctx), dim=0)
          else:
              target_policies = nd.array(target_policy, ctx)

        # print('target_values', target_values)
        # print('values',values)
        target_values = nd.array(target_values, ctx)
        target_rewards = nd.array(target_rewards, ctx=ctx)
        l = mse_loss(values, target_values) + mse_loss(rewards, target_rewards) + \
            cross_entropy_loss(
                nd.array(logits, ctx=ctx),
                nd.array(target_policies, ctx=ctx))

    l.backward(retain_graph=True)
    for o in optimizer:
        o.step(image.shape[0], ignore_stale_grad    =True)


        # for weights in network.get_weights():
        #     loss_ += weight_decay * loss.L2Loss(weights)





def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  mse = numpy.mean((target - prediction)**2)

  # mse.update_dict(target, prediction)
  # mse.get()

  return mse

######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    s = nd.softmax(nd.array([d[0] for d in distribution], ctx=get_ctx()), temperature=temperature)
    a = numpy.random.choice(range(0,len(s)), p=s.asnumpy())
    return 0, a

import multiprocessing
def launch_job(f, *args):
    # f(*args)
  p = multiprocessing.Process(target=f, args=(args))
  p.start()
  return p


def make_uniform_network():
  return Network()


if __name__ == '__main__':
  muzero(make_c_config())