import numpy as np
#import tensorflow as tf
#from tensorflow.keras import layers as tfkl
#from tensorflow_probability import distributions as tfd
#from tensorflow.keras.mixed_precision import experimental as prec

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=nn.ELU,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='gru',
      num_actions=None, embed = None, device=None):
    super(RSSM, self).__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = embed
    self._device = device

    inp_layers = []
    if self._discrete:
      inp_dim = self._stoch * self._discrete + num_actions
    else:
      inp_dim = self._stoch + num_actions
    if self._shared:
      inp_dim += self._embed
    for i in range(self._layers_input):
      inp_layers.append(nn.Linear(inp_dim, self._hidden))
      inp_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._inp_layers = nn.Sequential(*inp_layers)

    if cell == 'gru':
      #self._cell = tfkl.GRUCell(self._deter)
      self._cell = nn.GRUCell(self._hidden, self._deter)
    elif cell == 'gru_layer_norm':
      #self._cell = GRUCell(self._deter, norm=True)
      self._cell = GRUCell(self._hidden, self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    img_out_layers = []
    inp_dim = self._deter
    for i in range(self._layers_output):
      img_out_layers.append(nn.Linear(inp_dim, self._hidden))
      img_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._img_out_layers = nn.Sequential(*img_out_layers)

    obs_out_layers = []
    if self._temp_post:
      inp_dim = self._deter + self._embed
    else:
      inp_dim = self._embed
    for i in range(self._layers_output):
      obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
      obs_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._obs_out_layers = nn.Sequential(*obs_out_layers)

    if self._discrete:
      self._ims_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
      self._obs_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
    else:
      self._ims_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
      self._obs_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
 
  def initial(self, batch_size):
    #dtype = prec.global_policy().compute_dtype
    #deter = self._cell.get_initial_state(None, batch_size, dtype)
    deter = torch.zeros(batch_size, self._deter).to(self._device)
    if self._discrete:
      state = dict(
          #logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          #stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          deter=deter)
    else:
      state = dict(
          #mean=tf.zeros([batch_size, self._stoch], dtype),
          mean=torch.zeros([batch_size, self._stoch]).to(self._device),
          #std=tf.zeros([batch_size, self._stoch], dtype),
          std=torch.zeros([batch_size, self._stoch]).to(self._device),
          #stoch=tf.zeros([batch_size, self._stoch], dtype),
          stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
          deter=deter)
    return state

  #@tf.function
  def observe(self, embed, action, state=None):
    #swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      #state = self.initial(tf.shape(action)[0])
      state = self.initial(action.shape[0])
    embed, action = swap(embed), swap(action)
    post, prior = tools.static_scan(
        lambda prev_state, prev_act, embed: self.obs_step(
            prev_state[0], prev_act, embed),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  #@tf.function
  def imagine(self, action, state=None):
    #swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = action
    action = swap(action)
    prior = tools.static_scan(self.img_step, [action], state)
    prior = prior[0]
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      #shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      #stoch = tf.reshape(stoch, shape)
      stoch = stoch.reshape(shape)
    #return tf.concat([stoch, state['deter']], -1)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      #logit = tf.cast(logit, tf.float32)
      #dist = tfd.Independent(tools.OneHotDist(logit), 1)
      dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
      #if dtype != tf.float32:
      #  dist = tools.DtypeDist(dist, dtype or state['logit'].dtype)
    else:
      mean, std = state['mean'], state['std']
      #if dtype:
      #  mean = tf.cast(mean, dtype)
      #  std = tf.cast(std, dtype)
      #dist = tfd.MultivariateNormalDiag(mean, std)
      dist = tools.ContDist(torchd.independent.Independent(
          torchd.normal.Normal(mean, std), 1))
    return dist

  #@tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    #if not self._embed:
    #  self._embed = embed.shape[-1]
    prior = self.img_step(prev_state, prev_action, None, sample)
    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      if self._temp_post:
        #x = tf.concat([prior['deter'], embed], -1)
        x = torch.cat([prior['deter'], embed], -1)
      else:
        x = embed
      #for i in range(self._layers_output):
      #  x = self.get(f'obi{i}', tfkl.Dense, self._hidden, self._act)(x)
      x = self._obs_out_layers(x)
      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  #@tf.function
  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      #shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      #prev_stoch = tf.reshape(prev_stoch, shape)
      prev_stoch = prev_stoch.reshape(shape)
    if self._shared:
      if embed is None:
        #shape = prev_action.shape[:-1] + [self._embed]
        shape = list(prev_action.shape[:-1]) + [self._embed]
        #embed = tf.zeros(shape, prev_action.dtype)
        embed = torch.zeros(shape)
      #x = tf.concat([prev_stoch, prev_action, embed], -1)
      x = torch.cat([prev_stoch, prev_action, embed], -1)
    else:
      #x = tf.concat([prev_stoch, prev_action], -1)
      x = torch.cat([prev_stoch, prev_action], -1)
    #for i in range(self._layers_input):
    #  x = self.get(f'ini{i}', tfkl.Dense, self._hidden, self._act)(x)
    x = self._inp_layers(x)
    for _ in range(self._rec_depth): # rec depth is not correctly implemented
      deter = prev_state['deter']
      #x, deter = self._cell(x, [deter])
      x, deter = self._cell(x, [deter])
      deter = deter[0]  # Keras wraps the state in a list.
    #for i in range(self._layers_output):
    #  x = self.get(f'imo{i}', tfkl.Dense, self._hidden, self._act)(x)
    x = self._img_out_layers(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      #x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      #logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      #x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      #mean, std = tf.split(x, 2, -1)
      mean, std = torch.split(x, [self._stoch]*2, -1)
      mean = {
          'none': lambda: mean,
          #'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
          'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          #'softplus': lambda: tf.nn.softplus(std),
          #'abs': lambda: tf.math.abs(std + 1),
          #'sigmoid': lambda: tf.nn.sigmoid(std),
          #'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
          'softplus': lambda: torch.softplus(std),
          'abs': lambda: torch.abs(std + 1),
          'sigmoid': lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale):
    #kld = tfd.kl_divergence
    kld = torchd.kl.kl_divergence
    #dist = lambda x: self.get_dist(x, tf.float32)
    dist = lambda x: self.get_dist(x)
    #sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    sg = lambda x: {k: v.detach() for k, v in x.items()}
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      #value = kld(dist(lhs), dist(rhs))
      value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                  dist(rhs) if self._discrete else dist(rhs)._dist)
      #loss = tf.reduce_mean(tf.maximum(value, free))
      loss = torch.mean(torch.maximum(value, free))
    else:
      #value_lhs = value = kld(dist(lhs), dist(sg(rhs)))
      value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                              dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
      #value_rhs = kld(dist(sg(lhs)), dist(rhs))
      value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                      dist(rhs) if self._discrete else dist(rhs)._dist)
      #loss_lhs = tf.maximum(tf.reduce_mean(value_lhs), free)
      loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
      #loss_rhs = tf.maximum(tf.reduce_mean(value_rhs), free)
      loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value


class ConvEncoder(nn.Module):

  def __init__(self, grayscale=False,
               depth=32, act=nn.ReLU, kernels=(4, 4, 4, 4)):
    super(ConvEncoder, self).__init__()
    self._act = act
    self._depth = depth
    self._kernels = kernels

    layers = []
    for i, kernel in enumerate(self._kernels):
      if i == 0:
        if grayscale:
          inp_dim = 1
        else:
          inp_dim = 3
      else:
        inp_dim = 2 ** (i-1) * self._depth
      depth = 2 ** i * self._depth
      layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
      layers.append(act())
    self.layers = nn.Sequential(*layers)

  def __call__(self, obs):
    #x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
    x = x.permute(0, 3, 1, 2)
    x = self.layers(x)
    #for i, kernel in enumerate(self._kernels):
    #  depth = 2 ** i * self._depth
    #  x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
    #x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    # print('Encoder output:', x.shape)
    #shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
    shape = list(obs['image'].shape[:-3]) + [x.shape[-1]]
    #return tf.reshape(x, shape)
    return x.reshape(shape)


#class ConvDecoder(tools.Module):

  #def __init__(
  #    self, depth=32, act=tf.nn.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6),
  #    thin=True):
class ConvDecoder(nn.Module):

  def __init__(
      self, inp_depth,
      depth=32, act=nn.ReLU, shape=(3, 64, 64), kernels=(5, 5, 6, 6),
      thin=True):
    super(ConvDecoder, self).__init__()
    self._inp_depth = inp_depth
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

    if self._thin:
      self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
    else:
      self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
    inp_dim = 32 * self._depth

    cnnt_layers = []
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        #depth = self._shape[-1]
        depth = self._shape[0]
        act = None
      if i != 0:
        inp_dim = 2 ** (len(self._kernels) - (i-1) - 2) * self._depth
      cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
      if act is not None:
        cnnt_layers.append(act())
    self._cnnt_layers = nn.Sequential(*cnnt_layers)

  def __call__(self, features, dtype=None):
    #ConvT = tfkl.Conv2DTranspose
    if self._thin:
      #x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
      #x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
      x = self._linear_layer(features)
      x = x.reshape([-1, 1, 1, 32 * self._depth])
      x = x.permute(0,3,1,2)
    else:
      #x = self.get('hin', tfkl.Dense, 128 * self._depth, None)(features)
      #x = tf.reshape(x, [-1, 2, 2, 32 * self._depth])
      x = self._linear_layer(features)
      x = x.reshape([-1, 2, 2, 32 * self._depth])
      x = x.permute(0,3,1,2)
    #for i, kernel in enumerate(self._kernels):
    #  depth = 2 ** (len(self._kernels) - i - 2) * self._depth
    #  act = self._act
    #  if i == len(self._kernels) - 1:
    #    depth = self._shape[-1]
    #    act = None
    #  x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    x = self._cnnt_layers(x)
    # print('Decoder output:', x.shape)
    #mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    mean = x.reshape(features.shape[:-1] + self._shape)
    mean = mean.permute(0, 1, 3, 4, 2)
    #if dtype:
    #  mean = tf.cast(mean, dtype)
    #return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
    return tools.ContDist(torchd.independent.Independent(
      torchd.normal.Normal(mean, 1), len(self._shape)))


class DenseHead(nn.Module):

  #def __init__(
  #    self, shape, layers, units, act=tf.nn.elu, dist='normal', std=1.0):
  def __init__(
      self, inp_dim,
      shape, layers, units, act=nn.ELU, dist='normal', std=1.0):
    super(DenseHead, self).__init__()
    self._shape = (shape,) if isinstance(shape, int) else shape
    if len(self._shape) == 0:
      self._shape = (1,)
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

    mean_layers = []
    for index in range(self._layers):
      mean_layers.append(nn.Linear(inp_dim, self._units))
      mean_layers.append(act())
      if index == 0:
        inp_dim = self._units
    mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
    self._mean_layers = nn.Sequential(*mean_layers)

    if self._std == 'learned':
      self._std_layer = nn.Linear(self._units, np.prod(self._shape))

  def __call__(self, features, dtype=None):
    x = features
    #for index in range(self._layers):
    #  x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    #mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
    mean = self._mean_layers(x)
    #mean = tf.reshape(mean, tf.concat(
    #    [tf.shape(features)[:-1], self._shape], 0))
    if self._std == 'learned':
      #std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
      std = self._std_layer(x)
      #std = tf.nn.softplus(std) + 0.01
      std = torch.softplus(std) + 0.01
      #std = tf.reshape(std, tf.concat(
      #    [tf.shape(features)[:-1], self._shape], 0))
    else:
      std = self._std
    #if dtype:
    #  mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
    if self._dist == 'normal':
      #return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
      return tools.ContDist(torchd.independent.Independent(
        torchd.normal.Normal(mean, std), len(self._shape)))
    if self._dist == 'huber':
      #return tfd.Independent(
      return tools.ContDist(torchd.independent.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
    if self._dist == 'binary':
      #return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
      return tools.Bernoulli(torchd.independent.Independent(
        torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
    raise NotImplementedError(self._dist)


#class ActionHead(tools.Module):
class ActionHead(nn.Module):

  def __init__(
      #self, size, layers, units, act=tf.nn.elu, dist='trunc_normal',
      self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    super(ActionHead, self).__init__()
    # assert min_std <= 2
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

    pre_layers = []
    for index in range(self._layers):
      pre_layers.append(nn.Linear(inp_dim, self._units))
      pre_layers.append(act())
      if index == 0:
        inp_dim = self._units
    self._pre_layers = nn.Sequential(*pre_layers)

    if self._dist in ['tanh_normal','tanh_normal_5','normal','trunc_normal']:
      self._dist_layer = nn.Linear(self._units, 2 * self._size)
    elif self._dist in ['normal_1','onehot','onehot_gumbel']:
      self._dist_layer = nn.Linear(self._units, self._size)

  def __call__(self, features, dtype=None):
    x = features
    #for index in range(self._layers):
    #  kw = {}
    #  if index == self._layers - 1 and self._outscale:
    #    kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
    #        self._outscale)
    #  x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, **kw)(x)
    x = self._pre_layers(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      #x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = self._dist_layer(x)
      #if dtype:
      #  x = tf.cast(x, dtype)
      #mean, std = tf.split(x, 2, -1)
      mean, std = torch.split(x, 2, -1)
      #mean = tf.tanh(mean)
      mean = torch.tanh(mean)
      #std = tf.nn.softplus(std + self._init_std) + self._min_std
      std = F.softplus(std + self._init_std) + self._min_std
      #dist = tfd.Normal(mean, std)
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      #dist = tfd.Independent(dist, 1)
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      #x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = self._dist_layer(x)
      #if dtype:
      #  x = tf.cast(x, dtype)
      #mean, std = tf.split(x, 2, -1)
      mean, std = torch.split(x, 2, -1)
      #mean = 5 * tf.tanh(mean / 5)
      mean = 5 * torch.tanh(mean / 5)
      #std = tf.nn.softplus(std + 5) + 5
      std = F.softplus(std + 5) + 5
      #dist = tfd.Normal(mean, std)
      dist = torchd.normal.Normal(mean, std)
      #dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      #dist = tfd.Independent(dist, 1)
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      #x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = self._dist_layer(x)
      #if dtype:
      #  x = tf.cast(x, dtype)
      #mean, std = tf.split(x, 2, -1)
      mean, std = torch.split(x, 2, -1)
      #std = tf.nn.softplus(std + self._init_std) + self._min_std
      std = F.softplus(std + self._init_std) + self._min_std
      #dist = tfd.Normal(mean, std)
      dist = torchd.normal.Normal(mean, std)
      #dist = tfd.Independent(dist, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'normal_1':
      #mean = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = self._dist_layer(x)
      #if dtype:
      #  mean = tf.cast(mean, dtype)
      #dist = tfd.Normal(mean, 1)
      dist = torchd.normal.Normal(mean, 1)
      #dist = tfd.Independent(dist, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'trunc_normal':
      # https://www.desmos.com/calculator/mmuvuhnyxo
      #x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = self._dist_layer(x)
      #x = tf.cast(x, tf.float32)
      #mean, std = tf.split(x, 2, -1)
      mean, std = torch.split(x, [self._size]*2, -1)
      #mean = tf.tanh(mean)
      mean = torch.tanh(mean)
      #std = 2 * tf.nn.sigmoid(std / 2) + self._min_std
      std = 2 * torch.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      #dist = tools.DtypeDist(dist, dtype)
      #dist = tfd.Independent(dist, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'onehot':
      #x = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = self._dist_layer(x)
      #x = tf.cast(x, tf.float32)
      #dist = tools.OneHotDist(x, dtype=dtype)
      dist = tools.OneHotDist(x)
      #dist = tools.DtypeDist(dist, dtype)
    elif self._dist == 'onehot_gumble':
      #x = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = self._dist_layer(x)
      #if dtype:
      #  x = tf.cast(x, dtype)
      temp = self._temp
      #dist = tools.GumbleDist(temp, x, dtype=dtype)
      dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1/temp))
    else:
      raise NotImplementedError(self._dist)
    return dist


#class GRUCell(tf.keras.layers.AbstractRNNCell):
class GRUCell(nn.Module):

  #def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
  #  super().__init__()
  def __init__(self, inp_size,
               size, norm=False, act=torch.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    #self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    self._layer = nn.Linear(inp_size+size, 3*size,
                            bias=norm is not None)
    if norm:
      #self._norm = tfkl.LayerNormalization(dtype=tf.float32)
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  #def call(self, inputs, state):
  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    #parts = self._layer(tf.concat([inputs, state], -1))
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      #dtype = parts.dtype
      #parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      #parts = tf.cast(parts, dtype)
    #reset, cand, update = tf.split(parts, 3, -1)
    reset, cand, update = torch.split(parts, [self._size]*3, -1)
    #reset = tf.nn.sigmoid(reset)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    #update = tf.nn.sigmoid(update + self._update_bias)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]
