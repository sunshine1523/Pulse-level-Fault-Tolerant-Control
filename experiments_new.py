
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from qutip.qip.operations import rx
import pandas as pd
from tf_agents.policies.policy_saver import PolicySaver

import matplotlib.pyplot as plt
from qutip import *

from agent_tf_new import *
from environment_new import SpinQubitEnv, get_tf_environment, validate_evironment
import os


# import abc
# import tensorflow as tf
import numpy as np
import math
import pickle
# import tf_agents
# import math
# from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
# from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
# from tf_agents.trajectories import time_step as ts
#
# from tf_agents.agents.reinforce import reinforce_agent
# from tf_agents.drivers import dynamic_step_driver
# from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import tf_metrics
# from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import seaborn as sns

"""
Sample run with intial state Qobj = [[1]  and target = [[0]
                                    [0]]                [1]]

The measure of accuracy is fidelity. 
looking for a target accuracy of .99965 
The slighest of variations matters here given the growth of error 
with increase in qubits
"""


# action_space = ControlSpace([.4,10,10], [.1, 1, 1])
# environment = State(0.25,
#                     q.sigmax(),
#                     q.basis(2,0),
#                     [q.sigmaz(), q.sigmax(), q.sigmay(), q.sigmam()])

# control_agent = Agent(0.9, environment, action_space)
# control_agent.learn_rollout()


# Learning Parameters
num_iterations = 3  # @param {type:"integer"}
collect_episodes_per_iteration = 250  # @param {type:"integer"}
# for ddpg max repaly = 2
replay_buffer_capacity = 12500  # @param {type:"integer"}

fc_layer_params = (100, 2)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 2  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}


# Quantum Environment Parameters
interval_width = 1
target_unitary_list = qip.operations.rx(np.pi)
# phi = np.pi
# target_unitary_list = Qobj([[np.cos(phi / 2), -1j * np.sin(phi / 2), 0], [-1j * np.sin(phi / 2), np.cos(phi / 2), 0], [0, 0, 0]])
initial_state = basis(2, 0)
rabi_freq = 6
relative_detuning = np.random.uniform(-5e-3, 0.01, (1,))
t_width = 1

prefix_path = ("D:/pycharmprojects/SSRL_10.19/environment_new")

def train(tf_dumm, tf_agent):

    #parallel_env = ParallelPyEnvironment([SpinQubitEnv(en_configs[i]) for i in range(6)])
    #train_env = tf_py_environment.TFPyEnvironment(parallel_env)
    #gammavals = np.linspace(.01,.99, 5000)
    dummy_env = tf_py_environment.TFPyEnvironment(tf_dumm)
    # dummy_env.generate_state()


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=dummy_env.batch_size,
        max_length=replay_buffer_capacity)

    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(dummy_env, tf_agent.policy, 10)
    returns = [avg_return]
    gate_fidelity = []
    state_x = []
    state_y = []
    max_gate_avg_return = 0



    for _ in range(num_iterations):
        gate_avg_return = []
        experience = []

        # for x in np.linspace(0, np.pi, 5):
        #     for y in np.linspace(0, 2*np.pi, 5):

        for i in range(1):
            for target_unitary in [target_unitary_list]:
                for relative_detuning in np.random.uniform(-5e-3, 0.01, (5,)):

                    # for _ in range(num_iterations):
                    # num_iterations
                    # Collect a few episodes using collect_policy and save to the replay buffer.
                    #train_env,eval_env = get_environments(x, y)

                    train_env = get_tf_environment(t_width, target_unitary, rabi_freq, relative_detuning)

                    #np.random.seed(1)
                    #train_env.set_gamma([np.random.uniform(0, 1), np.random.uniform(0, 1)])
                    #train_env.set_gamma([.1, .3])
                    train_env = tf_py_environment.TFPyEnvironment(train_env)
                    collect_episode(
                        train_env, replay_buffer, tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.

        for _ in range(1):
            experience = replay_buffer.gather_all()
            train_loss = tf_agent.train(experience)

            step = tf_agent.train_step_counter.numpy()

            if step % 1 == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

                if True or step % eval_interval == 0:

                    # eval_py_env = SpinQubitEnv(environment_configs)
                    ##eval_gamma = [random.uniform(0,0.5),random.uniform(0,0.5)]
                    # eval_py_env.set_gamma(eval_gamma)
                    # eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
                    # for x in np.linspace(0, np.pi, 5):
                    #     for y in np.linspace(0, 2 * np.pi, 5):

                    eval_env = get_tf_environment(t_width, target_unitary, rabi_freq,
                                                                          relative_detuning)
                    # np.random.seed(1)
                    # eval_env.set_gamma([np.random.uniform(1e-4, 1e-5), np.random.uniform(1e-4, 1e-5)])
                    # eval_env.set_gamma([.1, .3])
                    eval_env_Batched = tf_py_environment.TFPyEnvironment(eval_env)
                    avg_return = compute_avg_return(eval_env_Batched, tf_agent.policy,
                                                                            num_eval_episodes)
                    # print('step = {0}: Average Return = {1}'.format(step, avg_return))
                    # gate_avg_return.append(avg_return)
                    # gate_avg_return = sum(gate_avg_return) / 25
                    print('step = {0}: Average Return = {1}'.format(step, avg_return))
                    returns.append(avg_return)
                    eval = evaluate(tf_agent, eval_env)
                    # ave_fide_without = []
                    # np.random.seed(0)
                    # for x in np.random.uniform(0, np.pi, (5,)):
                    #     np.random.seed(0)
                    #     for y in np.random.uniform(0, 2 * np.pi, (5,)):
                    #         initial_state = math.cos(x / 2) * basis(2, 0) + math.sin(x / 2) * basis(2, 1) * (
                    #                 math.cos(y) + 1j * math.sin(y))
                    #         initial_state = initial_state.unit()
                    #         fide = evaluate_policy(eval[2], initial_state, eval_env)
                    #         ave_fide_without.append(fide)
                    # print(np.mean(ave_fide_without))
                    if max_gate_avg_return < avg_return:
                        max_gate_avg_return = avg_return
                        tf_agent_max = tf_agent
                        eval_max = eval
                        # eval = evaluate(tf_agent_max, eval_env)
                        save_evals(
                            prefix_path+'/Visualizations_max',
                            eval_max, 999)

                    # gate_avg_return = []
                    replay_buffer.clear()




    return (step, train_loss, returns, eval_max, tf_agent_max)


def plot_training_return():
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=2)


def evaluate(tf_agent, eval_py_env):
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    data = []
    num_episodes = 1
    fidelity = []
    actions = []
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        U, fidel = eval_py_env.render()
        data.append(U)
        fidelity.append(fidel)
        while not time_step.is_last():
            action_step = tf_agent.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            actions.append(action_step.action)
            U, fidel = eval_py_env.render()
            data.append(U)
            fidelity.append(fidel)
    return data, fidelity, actions


def evaluate_policy(actions, initial_state, eval_py_env):
    controls = []
    H_control_total = []
    alpha = eval_py_env.alpha_anharmonicity * 2 * np.pi  # 非谐项 GHz
    print(alpha)
    Nstates = 2
    a = destroy(Nstates)
    adag = a.dag()
    Hnonlin = 0.5 * adag * adag * a * a  # 非谐项哈密顿量

    U, fidel = eval_py_env.render()
    target_state = target_unitary_list*initial_state
    state = initial_state
    # eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    actions = np.array(actions)

    # actions = np.array(actions).reshape(len(actions), 2).transpose()
    for i in range(len(actions)):
        controls = actions[i].reshape(2)
        t = np.arange(t_width * (i+1 - 1), t_width * (i+1),
                      0.001)
        # H_control = eval_py_env.get_hamiltonian_control(controls)
        H = [eval_py_env.H0(eval_py_env.relative_detuning), [Hnonlin, eval_py_env.anharmonicity_coeff], [a, eval_py_env.H1_coeff_gauss],
             [adag, eval_py_env.H2_coeff_gauss]]
        # t = np.arange(0, t_width, 1e-5)
        transition_state = mesolve(H, state, t, [],
                                   args={'w_drive': 2 * np.pi * eval_py_env.rabi_freq, 'alpha': alpha,
                                         'omega': eval_py_env.omega_gauss(4 + eval_py_env.noise_, controls[0]),
                                         'variance': 4 + eval_py_env.noise_,
                                         'gate time': len(actions) * interval_width})
        _state = transition_state.states[-1]
        fidel = (target_state.dag() * _state).norm()
        state = _state
        # U = np.exp(-1j * H_control * t)
        # fidel = eval_py_env.gate_fidelity(eval_py_env, U)
        # H_control_total.append(H_control)
    fidelity = fidel
    return fidelity


def save_policy(tf_agent, fname='/content/reinforce_cz'):
    my_policy = tf_agent.collect_policy
    # save policy
    PolicySaver(my_policy).save(fname)


def create_anim(fname, data):
    sphere = Bloch()
    for i in range(len(data)):
        sphere.clear()
        sphere.add_states(data[:i])
        sphere.make_sphere()
        sphere.save(fname + str(i))


environment_configs = validate_evironment()


def save_controls(control_name, fname, action, num):
    steps = [i*environment_configs.get_interval_width() for i in range(action.shape[0])]
    plt.clf()
    plt.step(steps, action)
    plt.ylabel(control_name)
    plt.xlabel('t')
    num = str(num)
    plt.savefig(fname+'_'+control_name+num)

def save_H_controls(fname, actions):
    actions_tran = np.array(actions).reshape(len(actions), 2).transpose()
    x = actions_tran[0]
    y = actions_tran[1]
    pulse =[]
    tau = len(x) * t_width * 0.5
    sigma = 4
    t = np.arange(0, t_width * len(x), 0.01)
    for i in range(len(x)):
        t_tmp = t[(i * 100):((i+1) * 100)]
        # t_tmp = np.arange(0, t_width, t_width * 0.01)
        I_tmp = x[i] * np.ones(len(t_tmp)) * np.pi / (2 * sigma)
        Q_tmp = x[i] * np.ones(len(t_tmp)) * np.pi / (2 * sigma)
        wav_tmp = I_tmp * np.exp(1j * 2 * np.pi * rabi_freq * t_tmp) * np.exp(
            -(t_tmp - tau) ** 2 / (sigma ** 2)) + Q_tmp * np.exp(-1j * 2 * np.pi * rabi_freq * t_tmp) * np.exp(
            -(t_tmp - tau) ** 2 / (sigma ** 2))
        pulse.append(wav_tmp)
    plt.clf()
    plt.plot(t, np.array(pulse).flatten())
    plt.ylabel('pulse')
    plt.xlabel('t')
    plt.savefig(fname+'_'+ 'pulse')


def save_fidelity(fname, fidelity, num):
    steps = [i*environment_configs.get_interval_width() for i in range(len(fidelity))]
    print(fidelity[len(fidelity)-1])

    plt.clf()
    plt.plot(steps, fidelity)
    plt.ylabel('fidelity')
    plt.xlabel('step')
    num = str(num)
    plt.savefig(fname + 'fidelity'+'_'+num)


def save_all_controls(fname, actions, num):
    actions_tran = np.array(actions).reshape(len(actions), 2).transpose()
    save_controls('Omega_x', fname, actions_tran[0], num)
    save_controls('Omega_y', fname, actions_tran[1], num)


def save_evals(fname, result, num):
    save_all_controls(fname, result[2], num)
    save_fidelity(fname, result[1], num)
    #create_anim(fname, result[0] )


def gate_fidelity(avg_return):
    ave_fidelity = sum(avg_return)/len(avg_return)
    return ave_fidelity

def picture_dict():
    file = prefix_path+"/ave_reward_cloning_" + ".pkl"
    with open(file, "rb") as f:
        data = pickle.load(f)
    x1 = data["mean"]

    sns.set(style="whitegrid", font_scale=1.5)

    c = {'Iteration Number': list(range(num_iterations+1)), 'avg_return': x1}
    cost = pd.DataFrame(c)
    ax = sns.regplot(x='Iteration Number', y='avg_return', data=cost, order=1, scatter_kws={"s": 2})
    # ax.set_titles("SSRL")
    # ax.savefig(prefix_path + '/new_avg_return' + '_')
    fig = plt.gcf()
    fig.savefig(prefix_path + '/new_avg_return' + '_')
    plt.show()


# %%
# for ddpg max repaly = 2
# replay_buffer_capacity = 2000

dummy_env = SpinQubitEnv(interval_width=t_width,
                         target_unitary=target_unitary_list,
                         rabi_freq=6,
                         relative_detuning=relative_detuning,
                         noise=0,
                         alpha_anharmonicity=-0.3)

# tf_dumm = tf_py_environment.TFPyEnvironment(dummy_env)


# %%

# agent_reinforce = get_agent(dummy_env, 'ppo', "noise_trained")
# train_results = train(dummy_env, agent_reinforce)
# ac_max = {"action": train_results[3][2]}
# with open(os.path.join(prefix_path, "action_cloning_max_" + ".pkl"), "wb") as f1:
#     pickle.dump(ac_max, f1, pickle.HIGHEST_PROTOCOL)
# path = prefix_path + "/Visualizations_max"
# save_H_controls(path, train_results[3][2])
# print(train_results)
# d = {"mean": train_results[2]}
#
#
# with open(os.path.join(prefix_path, "ave_reward_cloning_" + ".pkl"), "wb") as f:
#     pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
# # print('ave_fidelity: ' + str(gate_fidelity(train_results[4])))
#
#
# plt.clf()
# plt.plot(np.linspace(0, train_results[0] * 2, num_iterations + 1), train_results[2])
# plt.ylabel('avg_return')
# plt.xlabel('step')
# plt.savefig(prefix_path + '/avg_return' + '_')
# plt.show()
#
# picture_dict()
#
# num = 0
#
# PolicySaver(agent_reinforce.collect_policy).save(prefix_path+'/Visualizations')
#
#
# # qip.operations.rx(np.pi/2)
# # %%
# # noisy_configs
#
# env_noisy = SpinQubitEnv(interval_width=t_width,
#                          target_unitary=target_unitary_list,
#                          rabi_freq=6,
#                          relative_detuning=relative_detuning,
#                          noise=sigmaz())
# # env_noisy.set_gamma([.1, .3])
#
# env_without_noisy = SpinQubitEnv(interval_width=t_width,
#                                 target_unitary=target_unitary_list,
#                                 rabi_freq=6,
#                                 relative_detuning=relative_detuning)
#
# eval_without_noisy = evaluate(agent_reinforce, env_without_noisy)
# path = prefix_path + "/Visualizations"
# save_evals(path, eval_without_noisy, num=num+1)
# save_H_controls(path, eval_without_noisy[2])
# ac = {"action": eval_without_noisy[2]}
# with open(os.path.join(prefix_path, "action_cloning_" + ".pkl"), "wb") as f1:
#     pickle.dump(ac, f1, pickle.HIGHEST_PROTOCOL)
#
# # eval_noisy[1]
# eval_noisy = evaluate(agent_reinforce, env_noisy)
# path = prefix_path + "/Visualizations_noisy"
# save_evals(path, eval_noisy, num=num+1)


###########   test
def test_tau_drive():
    x_ = []
    y_ = []
    num_one = 0
    num_two = 0
    num_three = 0
    num_four = 0
    fide_change = []
    file = open('D:/PycharmProjects/SSRL_10.19/environment_new/action_cloning_max_.pkl', 'rb')
    data = pickle.load(file)
    strength = data['action']
    alpha = -0.3
    # strength = strength.numpy()[0]
    f_drive_list = np.linspace(- 0.005, 0.01, num=25)  # 5MHz
    for f_drive in f_drive_list:
        tau_list = np.linspace(-2, 2, num=25)
        for tau_ in tau_list:
            env_noisy = SpinQubitEnv(interval_width=t_width,
                                 target_unitary=target_unitary_list,
                                 rabi_freq=rabi_freq,
                                 relative_detuning=f_drive,
                                 noise=tau_,
                                 alpha_anharmonicity=alpha)
            ave_fide_without = []
            np.random.seed(0)
            for x in np.random.uniform(0, np.pi, (1,)):
                np.random.seed(0)
                for y in np.random.uniform(0, 2*np.pi, (1,)):
                    initial_state = math.cos(x / 2) * basis(2, 0) + math.sin(x / 2) * basis(2, 1) * (math.cos(y) + 1j * math.sin(y))
                    initial_state = initial_state.unit()
                    fide = evaluate_policy(strength, initial_state, env_noisy)
                    ave_fide_without.append(fide)
            ave_fide = np.mean(ave_fide_without)
            ave_fide_without = 0
            x_.append(f_drive * 1000)
            y_.append(tau_)
            fide_change.append(ave_fide)
            if ave_fide >= 0.95:
                num_one += 1
            if ave_fide >= 0.99:
                num_two += 1
            if ave_fide >= 0.999:
                num_three += 1
            if ave_fide >= 0.9999:
                num_four += 1
            # print(ave_fide)
    print(num_one)
    print(num_two)
    print(num_three)
    print(num_four)
            # print(eval_without_noisy[2])
    data = {"Detuning": x_, "Omega_t": y_, "fide": fide_change}

    with open("d:/PycharmProjects/SSRL_10.19/SSRL_3D_data_1.pkl", "wb") as f:
            pickle.dump(data, f)


    # f_list = open("d:/PycharmProjects/SSRL_10.19/SSRL_3D_data_1.pkl", "rb")
    # data = pickle.load(f_list)
    z = np.array(data['fide'])
    z = z.reshape(25, 25)
    fig1 = plt.figure()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    f_drive_list_ = np.linspace(- 0.005 * 1000, 0.01 * 1000, num=25)
    tau_list_ = np.linspace(-2, 2, num=25)
    surf_ = ax1.contourf(f_drive_list_, tau_list_, z, 100, cmap=plt.cm.Spectral, linewidth=0.2)
    ax1.set_xlabel('Detuning(MHz)')
    ax1.set_ylabel('Omega_t(ns)')
    fig1.colorbar(surf_, shrink=1)
    fname = 'SSRL_generalization_result_2D_1' + '.png'
    plt.savefig(fname, dpi=1080)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_trisurf(data['Detuning'], data['Omega_t'], data['fide'], cmap=plt.cm.Spectral, linewidth=0.2)
    ax.view_init(10, 50)
    ax.legend()
    # ax.set_zlim3d(0.9, 1)
    ax.set_xlabel('Detuning(MHz)')
    ax.set_ylabel('Omega_t(MHz)')
    ax.set_zlabel('Fidelity')
    # ax.set_title('Generalization')
    fig.colorbar(surf, shrink=0.6, ticks=[0.95, 0.99, 0.999])
    fname = 'SSRL_generalization_result_3D_1' + '.png'
    plt.savefig(fname, dpi=1080)
    f.close()


def test_drive_data():
    x_ = []

    fide_change = []
    file = open('D:/PycharmProjects/SSRL_10.19/environment_new/action_cloning_max_.pkl', 'rb')
    data = pickle.load(file)
    strength = data['action']
    # strength = strength.numpy()[0]
    f_drive_list = np.linspace(- 0.005, 0.01, num=25)  # 5MHz
    for f_drive in f_drive_list:
        tau_list = 0
        for tau_ in [tau_list]:
            env_noisy = SpinQubitEnv(interval_width=t_width,
                                     target_unitary=target_unitary_list,
                                     rabi_freq=rabi_freq,
                                     relative_detuning=f_drive,
                                     noise=tau_)
            ave_fide_without = []
            np.random.seed(0)
            for x in np.random.uniform(0, np.pi, (1,)):
                np.random.seed(0)
                for y in np.random.uniform(0, 2 * np.pi, (1,)):
                    initial_state = math.cos(x / 2) * basis(2, 0) + math.sin(x / 2) * basis(2, 1) * (
                                math.cos(y) + 1j * math.sin(y))
                    initial_state = initial_state.unit()
                    fide = evaluate_policy(strength, initial_state, env_noisy)
                    ave_fide_without.append(fide)
            ave_fide = np.mean(ave_fide_without)
            ave_fide_without = 0
            x_.append(f_drive * 1000)
            # if max_ave_fide >

            fide_change.append(ave_fide)
            # print(ave_fide)
    data = {"Detuning": x_, "fide": fide_change}

    with open("d:/PycharmProjects/SSRL_10.19/SSRL_drive_data_1.pkl", "wb") as f:
        pickle.dump(data, f)

    # f_list = open("d:/PycharmProjects/SSRL_10.19/SSRL_3D_data_1.pkl", "rb")
    # data = pickle.load(f_list)

def test_tau_data():
    x_ = []

    fide_change = []
    file = open('D:/PycharmProjects/SSRL_10.19/environment_new/action_cloning_max_.pkl', 'rb')
    data = pickle.load(file)
    strength = data['action']
    # strength = strength.numpy()[0]
    f_drive_list = 0.005
    for f_drive in [f_drive_list]:
        tau_list = np.linspace(-2, 2, num=25)
        for tau_ in tau_list:
            env_noisy = SpinQubitEnv(interval_width=t_width,
                                     target_unitary=target_unitary_list,
                                     rabi_freq=rabi_freq,
                                     relative_detuning=f_drive,
                                     noise=tau_)
            ave_fide_without = []
            np.random.seed(0)
            for x in np.random.uniform(0, np.pi, (1,)):
                np.random.seed(0)
                for y in np.random.uniform(0, 2 * np.pi, (1,)):
                    initial_state = math.cos(x / 2) * basis(2, 0) + math.sin(x / 2) * basis(2, 1) * (
                                math.cos(y) + 1j * math.sin(y))
                    initial_state = initial_state.unit()
                    fide = evaluate_policy(strength, initial_state, env_noisy)
                    ave_fide_without.append(fide)
            ave_fide = np.mean(ave_fide_without)
            ave_fide_without = 0
            x_.append(tau_)

            fide_change.append(ave_fide)
            # print(ave_fide)
    data = {"Omega_t": x_, "fide": fide_change}

    with open("d:/PycharmProjects/SSRL_10.19/SSRL_tau_data_1.pkl", "wb") as f:
        pickle.dump(data, f)

    # f_list = open("d:/PycharmProjects/SSRL_10.19/SSRL_3D_data_1.pkl", "rb")
    # data = pickle.load(f_list)

def test_anharmoricity_data():
    x_ = []

    fide_change = []
    file = open('D:/PycharmProjects/SSRL_10.19/environment_new/action_cloning_max_.pkl', 'rb')
    data = pickle.load(file)
    strength = data['action']
    # strength = strength.numpy()[0]
    f_drive_list = 0.005
    tau_ = 0
    for f_drive in [f_drive_list]:
        alpha_list = np.linspace(-0.4, -0.2, num=25)
        for alpha_ in alpha_list:
            env_noisy = SpinQubitEnv(interval_width=t_width,
                                     target_unitary=target_unitary_list,
                                     rabi_freq=rabi_freq,
                                     relative_detuning=f_drive,
                                     noise=tau_,
                                     alpha_anharmonicity=alpha_)
            ave_fide_without = []
            np.random.seed(0)
            for x in np.random.uniform(0, np.pi, (1,)):
                np.random.seed(0)
                for y in np.random.uniform(0, 2 * np.pi, (1,)):
                    initial_state = math.cos(x / 2) * basis(2, 0) + math.sin(x / 2) * basis(2, 1) * (
                            math.cos(y) + 1j * math.sin(y))
                    initial_state = initial_state.unit()
                    fide = evaluate_policy(strength, initial_state, env_noisy)
                    ave_fide_without.append(fide)
            ave_fide = np.mean(ave_fide_without)

            x_.append(alpha_)

            fide_change.append(ave_fide)
            print(ave_fide)
    data = {"Omega_t": x_, "fide": fide_change}

    with open("d:/PycharmProjects/SSRL_10.19/SSRL_alpha_data_1.pkl", "wb") as f:
        pickle.dump(data, f)

# ave_fide_noisy = []
# np.random.seed(0)
# for x in np.random.uniform(0, np.pi, (5,)):
#     np.random.seed(0)
#     for y in np.random.uniform(0, 2*np.pi, (5,)):
#         initial_state = math.cos(x / 2) * basis(2, 0) + math.sin(x / 2) * basis(2, 1) * (math.cos(y) + 1j * math.sin(y))
#         initial_state = initial_state.unit()
#         fide = evaluate_policy(eval_noisy[2], initial_state, env_noisy)
#         ave_fide_noisy.append(fide)
# print(np.mean(ave_fide_noisy))
# print(eval_noisy[2])
# for x in np.random.uniform(0, np.pi, 2):
#     for y in np.random.uniform(0, 2 * np.pi, 2):
#         eval_env = get_tf_environment_test(2, target_unitary, 1, relative_detuning)
#         eval_result = evaluate(agent_reinforce, eval_env)

if __name__ == '__main__':
    # test_drive_data()
    # test_tau_data()
    test_anharmoricity_data()