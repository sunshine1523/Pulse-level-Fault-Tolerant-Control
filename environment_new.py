
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
# import tensorflow as tf
import numpy as np
import seaborn as sns
from qutip import *
from qutip.qip.operations import rx
from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
# from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
# from tf_agents.typing import types
# from tf_agents.utils import nest_utils
# import math
from math import log, sin, cos
from scipy.linalg import expm
# from typing import Sequence, Optional
# from multiprocessing import dummy as mp_threads
import pandas as pd
import re
import math
# tf.compat.v1.enable_v2_behavior()
prefix_path = ("D:/pycharmprojects/SSRL_10.19/environment_new")
"""
action_space = ControlSpace([.4, 10, 10], [.1, 1, 1])
environment = State(0.25,
                    sigmax(),
                    basis(2, 0),
                    [sigmaz(),   sigmax(),   sigmay(),   sigmam()])

control_agent = Agent(0.9, environment, action_space)
"""


class SpinQubitEnv(py_environment.PyEnvironment):

    """
    Divide the total time frame of operation into n segments of equal duration. 
    There are two possibilities of going about this. 
       1. Use a heuristic to find the time taken for the whole operation. Assume that 
          this time is our worst case time. ie our algorithm will defnitely improve the 
           time. So divide the time required by heuristic into n segments

       2. Take the interval_width as a tunable parameter. Start with a random value. And 
           find the number of time steps such that, after n steps the state obtained will have 
           the maximum fidelity

        For computational reasons we go by method 2 [it goes better with RL approach]
    TODO [SCALE]: CUrrently the implemention is limited to single qubits. Make it generic using numpy array for n-qubits

    # Set the threshold to start fine tuning [with respect to fidelity] 
    pick the user defined intial control (default 0 0 0). Based on the intial control we find the k controls from knn or kfn for the fist time segment

    Create control list which can be appended after each segment duration. 

    detuning, Omega_x and Omega_y are the control parameters (or the actions) which can be varied
    for each time segment. Based on these controls the system will evolve into a new state. We start
    with controlling only the amplitude  and not the wave type (ie its always square wave)
    TODO [SCALE] : Have a functionality for changing the wave type

    TODO [STANDARDIZE] : Ensure all the values are float for consistancy. numpy mgrid stores the value aS 0. while a float is 0.0
    # This may create inconsitancy while converting to string. This is a minor issue as the coordinates will be always 
    # used from the coordinate matrix. (Since we are restricting any other control coordinate
    """

    def __init__(self, interval_width,
                 target_unitary,
                 # initial_state,
                 rabi_freq,
                 relative_detuning,
                 # gamma,
                 noise,
                 alpha_anharmonicity):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=[0.6, 0], maximum=[1.5, 0], name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=-50, name='observation')

        self._state = 0
        self._episode_ended = False
        self.fidelity = 0
        self.time_stamp = 0
        self.interval_width = interval_width
        self.max_stamp = 20  #*5  # 10 -> relative = 5 for omega = 10
        self.max_fidelity = 0

        self.target_unitary = target_unitary
        # self.initial_state = initial_state
        self.rabi_freq = rabi_freq
        self.relative_detuning = relative_detuning
        self.gamma = [0, 0]
        self.alpha_anharmonicity = alpha_anharmonicity
        # self.target_state = target_unitary*initial_state
        self.noise_ = noise  # noise is of the form [operator, fn]
        # self.evolution_unitary

        """
        Example for defining noise
        
        def noise(self, t, args):
            # TODO [ALGORITHM]:  make this staochastic using random variable for each segment
            h = 1
            return np.sin(10*t)*10*h/2
        noise_op = sigmaz()
        H_noise = [noise_op, noise]
        
        """

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def generate_state(self):
        dic = []
        index = []
        i = 0
        j = 0
        A = np.mat(np.zeros((2, 1)))

        dic = {i: str(A) for i in range(self.max_stamp + 1)}  # timesteps + target_state


        for i in range(25):  # number of state
            index.append(i)
        df = pd.DataFrame(dic, index)

        file = prefix_path + "/stste_" + ".pkl"
        df.to_pickle(file)

        np.random.seed(0)
        for x in np.random.uniform(0, np.pi, (5,)):
            np.random.seed(0)
            for y in np.random.uniform(0, 2 * np.pi, (5,)):

                initial_state = cos(x / 2) * basis(2, 0) + sin(x / 2) * basis(2, 1) * (cos(y) + 1j * sin(y))
                initial_state = initial_state.unit()
                target_state = self.target_unitary * initial_state
                a = str(initial_state.full())
                a = re.sub('\\[','', a)
                a = re.sub('\\]','', a)
                a = re.sub(' ', '', a)
                df.at[j, 0] = a
                b = str(target_state.full())
                b = re.sub('\\[','', b)
                b = re.sub('\\]','', b)
                b = re.sub(' ', '', b)
                df.at[j, 20] = b
                j += 1

        df.to_pickle(file)
        # print(df)




    def _reset(self):
        # self._state = 0
        # self.generate_state()
        self._episode_ended = False
        self.fidelity = 0
        self.time_stamp = 0
        self.max_fidelity = 0
        #self.noise = None
        #self.gamma = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
        return ts.restart(np.array([0, 0, 0], dtype=np.float32))

    def render(self):

        return (self._state, self.fidelity)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def _step(self, action):
        self.time_stamp += 1

        # else:
        #     self._episode_ended = False

        if self.time_stamp >= self.max_stamp:
            self._episode_ended = True

        if self._episode_ended:
            #self.max_fidelity = 0
            return ts.termination(np.array([0, 0, 0], dtype=np.float32), 0)
        # if self._episode_ended:
        #     self.max_fidelity = 0
        #     return ts.termination(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 0)

        H_control, c_ops = self.get_hamiltonian(action)

        # newly added

        c_ops = []
        # print(self.gamma)
        if self.gamma[0] > 0.0:
            H_control += np.sqrt(self.gamma[0]) * sigmam()

        if self.gamma[1] > 0.0:
            H_control += np.sqrt(self.gamma[1]) * sigmaz()
        #print(self.gamma[1])

        ######
        # U = np.matrix(np.identity(2, dtype=complex))  # initial Evolution operator
        # U = expm(-1j * H * self.dt)  # Evolution operator
        # gate_infidelity = 1 - self.gate_transition_fidelity(self, U, H)
        # U = expm(-1j * H * self.interval_width)  # Evolution operator
        # gate_infidelity = 1 - (U.dag() * self.target_unitary).norm()
        # list of times for which the solver should store the state vector
        t = np.arange(self.get_interval_width() * (self.time_stamp - 1), self.get_interval_width() * self.time_stamp, 0.001)
        # t = np.arange(0, self.get_interval_width(), 1e-3)
        ### Transmon
        Nstates = 2
        a = destroy(Nstates)
        adag = a.dag()
        alpha = self.alpha_anharmonicity * 2 * np.pi  # 非谐项 GHz
        Hnonlin = 0.5 * adag * adag * a * a  # 非谐项哈密顿量
        H = [self.H0(self.relative_detuning), [Hnonlin, self.anharmonicity_coeff], [a, self.H1_coeff_gauss], [adag, self.H2_coeff_gauss]]

        # noise
        # H_noise = self.noise(t, None)
        # H = [H_control, H_noise]
        # if self.noise == None:
        #     H = H_control
        # else:
        #     args_test = {}
        #     args_test['Gaussian time'] = 30
        #     args_test['amp'] = 1
        #     args_test['drive frequency'] = 1
        #     H_noise = self.drive_pulse(t, args_test)
        #     H = [H_control, H_noise]
        #     #print(H_noise)
        fidelity_all = []
        # self.generate_state()

        df1 = pd.read_pickle(prefix_path + "/stste_" + ".pkl")
        # a = df.at[0, self.time_stamp - 1]
        # b = np.fromstring(df.at[0, self.time_stamp - 1], dtype=complex).reshape(2, 1)
        # print(b)

        for i in range(25):
            # df.at[i, self.time_stamp - 1] = re.sub(' ', '', df.at[i, self.time_stamp - 1])
            # print(df1.at[i, self.time_stamp - 1])
            # print(self.time_stamp)
            # print(df1.at[i, 0])
            tem = np.array(df1.at[i, self.time_stamp - 1].split('\n')).reshape(2, 1)
            # tem = np.fromstring(df.at[i, self.time_stamp - 1], dtype=complex).reshape(2, 1)
            _state = Qobj(tem)
            # print(self._state)
            transition_state = mesolve(H, _state, t, [],
                                             args={'w_drive': 2 * np.pi * self.rabi_freq, 'alpha': alpha, 'omega': self.omega_gauss(4 + self.noise_, action[0]),
                                                   'variance': 4 + self.noise_, 'gate time': self.max_stamp * self.interval_width})
            # transition_state = mesolve(H, _state, t, c_ops=c_ops)
            tem_tran_str = str(transition_state.states[-1].unit().full())

            tem_tran_str = re.sub('\\[', '', tem_tran_str)
            tem_tran_str = re.sub('\\]', '', tem_tran_str)
            tem_tran_str = re.sub(' ', '', tem_tran_str)

            df1.at[i, self.time_stamp] = tem_tran_str
            # print(df.at[i, self.time_stamp])
            # print(df.at[i, self.time_stamp-1])

            tem_finall = np.array(df1.at[i, 20].split('\n')).reshape(2, 1)
            # print(tem_finall)
            target_ = Qobj(tem_finall)
            # new_fidelity = self.get_transition_fidelity(transition_state.states[-1], target_)
            new_fidelity = fidelity(transition_state.states[-1], target_)
            fidelity_all.append(new_fidelity)

            df1.to_pickle(prefix_path + "/stste_" + ".pkl")


        new_fidelity = np.mean(fidelity_all, dtype=float)

        #reward = 2*new_fidelity - self.fidelity - self.max_fidelity
        #reward = reward if reward > 0 else 0
        #reward = new_fidelity - self.fidelity
        #reward = -math.log(1-new_fidelity)
        # reward_ = new_fidelity - self.fidelity
        # if (new_fidelity > 0.9995 or self.time_stamp >= self.max_stamp):
        #     reward = -math.log(1-new_fidelity) * 1000
        # else:
        #     reward = -self.time_stamp

        self.fidelity = new_fidelity
        self.max_fidelity = new_fidelity if new_fidelity > self.max_fidelity else self.max_fidelity

        observation = [#self.fidelity,
                       # expect(sigmax(), self._state),
                       # expect(sigmay(), self._state),
                       # expect(sigmaz(), self._state),
                       self.time_stamp,
                       action[0],
                       action[1],
                       # H_control_total
                       ]

        if (self.fidelity >= 0.9999990):
            self._episode_ended = True
            # reward = -math.log(1-new_fidelity) * 100
            reward = new_fidelity * 1000
            #self.max_fidelity = 0
            #self.fidelity = 0
            #self.gamma = [random.uniform(0, 1), random.uniform(0, 1)]
            return ts.termination(np.array(observation, dtype=np.float32), reward=reward)

        if (self.fidelity < 0.9999990 and self.time_stamp == self.max_stamp - 1):
            # reward = -math.log(1-new_fidelity) * 10
            reward = -math.log(1-new_fidelity) * 10
        else:
            reward = 0

        return ts.transition(
            np.array(observation, dtype=np.float32), reward=reward, discount=0.98)

    def get_target_state(self, transition_state):
        return self.target_unitary*transition_state

    def get_current_state(self):
        return self.current_state

    def get_interval_width(self):
        return self.interval_width

    def get_fidelity(self):
        return (self.get_target_state().dag() * self.get_current_state()).norm()

    def get_transition_fidelity(self, transition_state, target_state):
        return (target_state.dag()*transition_state).norm()

    def gate_transition_fidelity(self, U, H):
        U = expm(-1j * H * self.interval_width)  # Evolution operator
        return (U.dag()*self.target_unitary).norm()

    def get_hamiltonian_operator_z(self):
        return sigmaz()

    def get_hamiltonian_operator_x(self):
        return sigmax()

    def get_hamiltonian_operator_y(self):
        return sigmay()

    def get_hamiltonian_operator_noise(self):
        return self.noise

    def noise(self, t, args):
        # TODO [ALGORITHM]:  make this staochastic using random variable for each segment
        h = 1
        return np.sin(10*t)*10*h/2

    def update_state(self, next_state):
        self.current_state = next_state

    def get_hamiltonian_control(self, controls):
        h = 1
        I_tmp = controls[0] * 2 * np.pi
        Q_tmp = controls[1] * 2 * np.pi
        tau = self.interval_width * self.max_stamp * 0.5 # 均值
        sigma = 10 # 标准差
        # omegax = 2*np.pi*self.rabi_freq*controls[0]
        # omegay = 2*np.pi*self.rabi_freq*controls[1]
        omegay = Q_tmp * np.exp(
            -(self.interval_width - tau) ** 2 / (sigma ** 2))
        omegax = I_tmp * np.exp(
            -(self.interval_width - tau) ** 2 / (sigma ** 2))

        detuning = self.relative_detuning * 2 * np.pi * self.rabi_freq
        H_z = self.get_hamiltonian_operator_z()
        H_x = self.get_hamiltonian_operator_x()
        H_y = self.get_hamiltonian_operator_y()
        H_control = (detuning*H_z
                     + 0.5 * omegax*H_x
                     + 0.5 * omegay*H_y)
        return H_control

    def get_hamiltonian(self, controls):
        H_control = self.get_hamiltonian_control(controls)
        c_ops = []
        # H_control_total = []
        # H_control_total.append(H_control)
        if self.gamma[0] > 0.0:
            c_ops.append(np.sqrt(self.gamma[0]) * sigmam())

        if self.gamma[1] > 0.0:
            c_ops.append(np.sqrt(self.gamma[1]) * sigmaz())

        # H_noise = [self.get_hamiltonian_operator_noise(), self.noise]
        # H_control = [H_control,H_noise]
        return [H_control, c_ops]

    def drive_pulse(self, t, args):
        t_drive = args['Gaussian time']
        amp = args['amp']
        w_d = args['drive frequency']
        tau = 0
        sigma = t_drive/4.0
        return amp * np.exp(-(t-tau)**2/(sigma**2))*np.cos(w_d * t)

    def H0(self, deta):  # 哈密顿量
        Nstates = 2
        a = destroy(Nstates)
        adag = a.dag()
        H0 = (self.rabi_freq + deta) * 2 * np.pi * adag * a
        return H0

    def anharmonicity_coeff(self, t, args):  # 用来返回非谐项
        return args['alpha']

    ## Pulse
    def H1_coeff_gauss(self, t, args):
        t_g = args['gate time']
        omega = args['omega']
        wd = args['w_drive']
        tau = t_g / 2
        sigma_ = args['variance']
        amp = 0.5 * omega * np.exp(-(t-tau) ** 2 / (sigma_ ** 2)) * np.exp(1j * wd * t)
        return amp

    def H2_coeff_gauss(self, t, args):
        t_g = args['gate time']
        omega = args['omega']
        wd = args['w_drive']
        tau = t_g / 2
        sigma_ = args['variance']
        amp = 0.5 * omega * np.exp(-(t-tau) ** 2 / (sigma_ ** 2)) * np.exp(-1j * wd * t)
        return amp

    def omega(self, sigma_):
        return np.pi / (2 * sigma_)

    def omega_gauss(self, sigma_, strength):
        return self.omega(sigma_) * strength



def validate_evironment():
    # state = []
    # for i in range(25):
    #     state = basis(2,0)
    #     state.append(state)

    validate_env = SpinQubitEnv(1,
                                qip.operations.rx(np.pi),
                                # state,
                                6,
                                0.001,
                                0,
                                -0.3)
                                # None)
    utils.validate_py_environment(validate_env, episodes=5)
    return validate_env


def get_tf_environment(interval_width, target_unitary, rabi_freq, relative_detuning, noise, alpha_anharmonicity):
    # initial_state_all = []
    # for i in range(len(x)):
    #
    #     initial_state = cos(x[i]/2)*basis(2, 0) + sin(x[i]/2)*basis(2, 1)*(cos(y[i]) + 1j*sin(y[i]))
    #     initial_state = initial_state.unit()
    #     initial_state_all.append(initial_state)

    py_env = SpinQubitEnv(interval_width,
                          target_unitary,
                          # initial_state_all,
                          rabi_freq,
                          relative_detuning,
                          # gamma,
                          noise,
                          alpha_anharmonicity)

    # tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return py_env


# %%


# %%


# %%
