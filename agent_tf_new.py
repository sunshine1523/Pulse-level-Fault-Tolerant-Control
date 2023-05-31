import tf_agents
# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import tf_metrics
# from tf_agents.networks import q_network
# from itertools import combinations
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import trajectory
# import tensorflow as tf
from tensorflow import compat
# Package tensorflow._api.v2.compat
from tf_agents.agents.reinforce import reinforce_agent
# import numpy

# TODO: Encapsulate agents into a class.
# of replay for ddgp
num_iterations = 1000  # @param {type:"integer"}
collect_episodes_per_iteration = 100  # @param {type:"integer"}
# for ddpg max repaly = 2
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (100, 2)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 2  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}




def get_ddpg_agent(spin_py_enviroment, name):
    train_env = tf_py_environment.TFPyEnvironment(spin_py_enviroment)
    actor_network = tf_agents.agents.ddpg.actor_network.ActorNetwork(
        train_env.observation_spec(),
        train_env.action_spec())

    critic_network = tf_agents.agents.ddpg.critic_network.CriticNetwork(
        (train_env.observation_spec(),
         train_env.action_spec()), name='CriticNetwork'
    )

    optimizer = compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = compat.v2.Variable(0)

    name = name + '_ddpg'
    tf_agent = tf_agents.agents.DdpgAgent(
        train_env.time_step_spec(), train_env.action_spec(), actor_network, critic_network,
        actor_optimizer=optimizer, critic_optimizer=optimizer, ou_stddev=1.0, ou_damping=1.0,
        target_actor_network=None, target_critic_network=None, target_update_tau=1.0,
        target_update_period=1, dqda_clipping=None, td_errors_loss_fn=None, gamma=1.0,
        reward_scale_factor=1.0, gradient_clipping=None, debug_summaries=False,
        summarize_grads_and_vars=False, train_step_counter=None, name=name
    )

    return tf_agent

# Creates a PPO Agent implementing the clipped probability ratios
def get_ppo_agent(spin_py_enviroment, name):
    train_env = tf_py_environment.TFPyEnvironment(spin_py_enviroment)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec())
    value_net = tf_agents.networks.value_network.ValueNetwork(
        (
         train_env.observation_spec()), preprocessing_layers=None, preprocessing_combiner=None
    )
    optimizer = compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    tf_agent = tf_agents.agents.PPOClipAgent(
        train_env.time_step_spec(), train_env.action_spec(), optimizer=optimizer, actor_net=actor_net,
        value_net=value_net, greedy_eval=True, importance_ratio_clipping=0.3, lambda_value=0.98, discount_factor=0.96,
        entropy_regularization=0.0, policy_l2_reg=0.0, value_function_l2_reg=0.0, shared_vars_l2_reg=0.0,
        value_pred_loss_coef=0.5, num_epochs=2, use_gae=True, use_td_lambda_return=True, normalize_rewards=True,
        reward_norm_clipping=50, normalize_observations=True, log_prob_clipping=0.0, gradient_clipping=0.0,
        value_clipping=0.0, check_numerics=True, compute_value_and_advantage_in_train=True,
        update_normalizers_in_train=True, debug_summaries=False,
        summarize_grads_and_vars=False, name=name
    )
    return tf_agent

def get_reinforce_agent(spin_py_environment, name):

    train_env = tf_py_environment.TFPyEnvironment(spin_py_environment)
    actor_net = actor_distribution_network.ActoistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec())

    optimizer = compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = compat.v2.Variable(0)
    name = name + '_reinforce'
    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter,
        name=name)

    tf_agent.initialize()

    return tf_agent


def get_agent(env, agent_type, name):
    return get_reinforce_agent(env, name) if agent_type == 'reinforce' else get_ppo_agent(env, name)


# def compute_avg_return(environment, policy, num_episodes=10):
#
#     total_return = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             time_step = environment.step(action_step.action)
#             fide = time_step.observation[0].numpy()[0]
#             # print(fide)
#             if (fide > 0):
#                 episode_return = fide
#             #episode_return += time_step.reward
#             # print(_, time_step.observation[0])
#             # print(num_episodes)
#         #episode_return = time_step.observation[0]
#         total_return += episode_return
#
#     avg_return = total_return / num_episodes
#     return avg_return

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            # fide = time_step.observation[0].numpy()[0]
            # # print(fide)
            # if (fide > 0):
            #     episode_return = fide
            episode_return += time_step.reward.numpy()[0]
            # print(_, time_step.observation[0])
            # print(num_episodes)
        #episode_return = time_step.observation[0]
        total_return += episode_return

    avg_return = total_return / num_episodes

    return avg_return

def collect_episode(environment, replay_buffer, policy, num_episodes):

    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


