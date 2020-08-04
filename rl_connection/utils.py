import numpy as np
import torch

from torch.distributions import Categorical


class ActorCritic(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_hidden_units):
        """
        Todo: Research to see if Actor and Critic should share some layers
        Todo: Look into initialization
        :param n_features:
        :param n_outputs:
        :param n_hidden_units:
        """
        super(ActorCritic, self).__init__()

        # Actor
        self.actor_layer = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, n_outputs),
            torch.nn.Softmax(dim=-1)
        )

        # Critic
        self.critic_layer = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, 1)
        )

    def forward(self, state):
        actor_output = self.actor_layer(state)
        value = self.critic_layer(state)
        dist = Categorical(actor_output)

        return dist, value

    def evaluate(self, state, action_index):
        actor_output = self.actor_layer(state)
        dist = Categorical(actor_output)
        entropy = dist.entropy().mean()
        log_probs = dist.log_prob(action_index.view(-1))
        value = self.critic_layer(state)

        return log_probs, value, entropy


class PPO:
    def __init__(self, action_to_index, n_states, n_actions, alpha=2e-5, gamma=0.99, batch_size=128):
        # Set attributes
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_to_index = action_to_index
        self.index_to_action = {index: action for action, index in action_to_index.items()}

        # Hyper parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize weights
        self.n_hidden_units = n_actions
        self.policy_net = ActorCritic(
            n_features=self.n_states,
            n_hidden_units=self.n_hidden_units,
            n_outputs=self.n_actions
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def sample_action(self, state):
        """
        Returns an action or an action_index

        :param state:
        :return: action_index
        """
        dist, value = self.policy_net(state)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        action = self.index_to_action[action_index.data.numpy()[0]]

        # Return action
        return action, log_prob, value

    def evaluate_state_and_action(self, state, action):
        """
        Obtain:
           - the probability of selection `action_index` when in input state 'state'
           - the value of the being in input state `state`
        :param action:
        :param state:
        :return:
        """
        # Obtain state and action index:
        action_index = self.action_to_index.get(action)

        # Check if creature hadn't taken any actions.
        if action_index is None:
            return

        # Convert to tensor
        action_index = torch.tensor(action_index)

        dist, value = self.policy_net(state)
        log_prob = dist.log_prob(action_index)
        return log_prob, value

    def get_gae(self, trajectory, lmbda=0.95):
        """
        :param trajectory:
        :param lmbda:
        :return:
        """
        # Todo: replace this codeblock
        rewards = [t[2] for t in trajectory]
        values = [t[-1] for t in trajectory]
        dummy_next_value = 0  # should get masked out
        values = values + [dummy_next_value]
        masks = [t[3] is not None for t in trajectory]

        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * lmbda * masks[step] * gae
            returns.insert(0, gae + values[step])

        returns = torch.cat(returns)
        return returns

    def get_returns(self, trajectory):
        rewards = [t[2] for t in trajectory]
        is_terminals = [t[3] is None for t in trajectory]
        discounted_rewards = list()

        for reward, is_terminal in reversed(list(zip(rewards, is_terminals))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, [discounted_reward])

        discounted_rewards = torch.tensor(discounted_rewards)

        return discounted_rewards

    @staticmethod
    def select_random_batch(current_states, actions, log_probs, returns, advantages, mini_batch_size):
        random_indicies = np.random.randint(0, len(current_states), mini_batch_size)

        batch_current_states = current_states[random_indicies]
        batch_actions = actions[random_indicies]
        batch_log_probs = log_probs[random_indicies]
        batch_returns = returns[random_indicies]
        batch_advantages = advantages[random_indicies]

        return batch_current_states, batch_actions, batch_log_probs, batch_returns, batch_advantages

    def update(self, trajectory, clip_val=0.2):
        """
        :param trajectory:
        :param clip_val:
        :return:
        """
        # Extract from trajectory
        current_states, actions, rewards, next_states, old_log_probs, old_values = list(zip(*trajectory))
        current_states = torch.cat(current_states)
        old_log_probs = torch.cat(old_log_probs)
        old_values = torch.cat(old_values)
        returns = self.get_gae(trajectory)

        # Obtain advantages
        advantages = returns - old_values
        action_indicies = torch.tensor([[self.action_to_index[action]] for action in actions])

        # Learn for each step in trajectory
        for _ in range(len(trajectory)):
            # Get random sample of experiences
            batch_current_state, batch_action_indicies, batch_old_log_probs, batch_returns, batch_advantages = \
                self.select_random_batch(
                    current_states=current_states,
                    actions=action_indicies,
                    log_probs=old_log_probs,
                    returns=returns,
                    advantages=advantages,
                    mini_batch_size=16
                )
            batch_old_log_probs = batch_old_log_probs.detach()
            batch_current_state = batch_current_state.detach()
            batch_action_indicies = batch_action_indicies.detach()

            new_log_probs, new_values, entropy = self.policy_net.evaluate(
                batch_current_state,
                batch_action_indicies
            )

            # Calculate loss for actor
            ratio = (new_log_probs - batch_old_log_probs.detach()).exp().view(-1, 1)
            loss1 = ratio * batch_advantages.detach()
            loss2 = torch.clamp(ratio, 1 - clip_val, 1 + clip_val) * batch_advantages.detach()
            actor_loss = -torch.min(loss1, loss2).mean()

            # Calculate loss for critic
            sampled_returns = batch_returns.detach()
            critic_loss = (new_values - sampled_returns).pow(2).mean()

            # Credit: https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
            overall_loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            self.optimizer.zero_grad()
            overall_loss.backward()
            self.optimizer.step()

    def determine_reward(self, output_sentence, target_sentence):
        """
        Uses ROUGE to calculate reward
        :param output_sentence:
        :param target_sentence:
        :return:
        """
        pass
