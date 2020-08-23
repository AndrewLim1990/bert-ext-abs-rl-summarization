from abstractor.train import nll_loss
from bert.utils import START_OF_SENTENCE_TOKEN
from bert.utils import obtain_word_embeddings
from extractor.utils import BERT_OUTPUT_SIZE
from rouge import Rouge
from torch.distributions import Categorical

import numpy as np
import torch


class ActorCritic(torch.nn.Module):
    def __init__(self, extraction_model):
        """
        :param extraction_model:
        """
        super(ActorCritic, self).__init__()

        # Actor
        self.extraction_model = extraction_model  # returns prob of extraction per sentence
        self.convert_to_dist = torch.nn.Sequential(
            Logit(),
            torch.nn.Softmax(dim=1)
        )

        self.softmax = torch.nn.Softmax(dim=2)
        self.sigmoid = torch.nn.Sigmoid()

        # Convert bert embeddings to custom embeddings
        self.tune_dim = 8
        self.bert_fine_tune = torch.nn.Linear(BERT_OUTPUT_SIZE, self.tune_dim, bias=False)

        self.bert_tokenizer = extraction_model.bert_tokenizer
        self.bert_model = extraction_model.bert_model
        self.init_hidden = torch.nn.Parameter(torch.rand(1, self.tune_dim), requires_grad=True)
        self.stop_embedding = torch.nn.Parameter(torch.rand(extraction_model.input_dim), requires_grad=True)

        self.gru = torch.nn.GRU(self.tune_dim * 2, self.tune_dim, batch_first=True)
        self.value_layer = torch.nn.Linear(self.tune_dim, 1)
        self.max_n_ext_sents = 4

    def add_stop_action(self, state, mask=None):
        # Add stop embedding:
        batch_size = state.shape[0]
        stop_embeddings = torch.cat([self.stop_embedding] * batch_size)
        stop_embeddings = stop_embeddings.view(batch_size, 1, -1)
        state = torch.cat([state, stop_embeddings], dim=1)

        # Don't mask out stop embedding
        if mask is not None:
            mask = torch.cat([mask, torch.ones(batch_size, 1)], dim=1)

        return state, mask

    def forward(self, state, mask, n_label_sents=None):
        action_dists, action_indicies, ext_sents = self.actor_layer(state, mask, n_label_sents)
        values = self.critic_layer(state, mask, ext_sents)
        return action_dists, action_indicies, values

    def actor_layer(self, batch_state, mask, n_label_sents=None):
        """
        :param batch_state:
        :param mask:
        :param n_label_sents: list containing number of extracted sentences in summary labels (oracle)
        :return:
        """
        # Obtain distribution amongst actions
        batch_state, mask = self.add_stop_action(batch_state, mask)
        batch_action_probs = self.extraction_model.forward(batch_state)  # prob of extraction per sentence (binary)
        batch_action_probs = batch_action_probs * mask

        # Obtain number of samples in batch
        batch_size = batch_action_probs.shape[0]

        # Obtain maximum number of sentences to extract
        if n_label_sents is None:
            n_doc_sents = mask.sum(dim=1)
            batch_max_n_ext_sents = torch.tensor([self.max_n_ext_sents] * batch_size)
            batch_max_n_ext_sents = torch.min(batch_max_n_ext_sents.float(), n_doc_sents)
        else:
            batch_max_n_ext_sents = n_label_sents

        # Iterate through samples
        batch_action_dists, batch_action_indicies, batch_ext_sents = list(), list(), list()
        for sample_idx in range(batch_size):
            max_n_ext_sents = batch_max_n_ext_sents[sample_idx]
            action_probs = batch_action_probs[sample_idx:sample_idx+1]  # (1, n_actions)
            n_actions = action_probs.shape[1]
            stop_action_idx = n_actions - 1  # Previously appended stop_action embedding
            action_indicies, ext_sents, action_dists = list(), list(), list()

            # Extract sentences one at a time
            n_ext_sents = 0
            while True:
                # Obtain distribution amongst sentences to extract
                action_dist = self.convert_to_dist(action_probs)
                action_dist = Categorical(action_dist)

                # Sample sentence to extract
                action_idx = action_dist.sample()  # index of sentence to extract
                ext_sent = batch_state[sample_idx, action_idx:action_idx+1, :]  # embedding of sentence to extract

                # Collect
                action_dists.append(action_dist)
                ext_sents.append(ext_sent)
                action_indicies.append(action_idx)

                # Don't select sentence again in future
                extraction_mask = torch.ones(action_probs.shape)
                for extracted_idx in action_indicies:
                    extraction_mask[0, extracted_idx] = 0
                action_probs = action_probs * extraction_mask

                # Track number of sentences extracted from article
                n_ext_sents = n_ext_sents + 1

                # Check to see if should stop extracting sentences
                is_stop_action = action_idx >= stop_action_idx
                is_long_enough = n_ext_sents >= max_n_ext_sents
                if is_stop_action or is_long_enough:
                    break

            # Collect
            batch_action_dists.append(action_dists)  # Sentence extraction distribution at each time step
            batch_action_indicies.append(action_indicies)  # Extracted sentence index at each time step
            batch_ext_sents.append(ext_sents)  # Emebddings of extracted sentences at each time step

        batch_action_indicies = [torch.tensor(article_actions) for article_actions in batch_action_indicies]

        return batch_action_dists, batch_action_indicies, batch_ext_sents

    def critic_layer(self, batch_state, batch_mask, batch_extracted_sents):
        """
        :param batch_state: torch.tensor shape (batch_size, n_doc_sents, bert_dim)
        :param batch_mask: torch.tensor shape (batch_size, n_doc_sents, bert_dim)
        :param batch_extracted_sents: list(list(extracted_sent_embedding))
            extracted_sent_embedding: torch.tensor shape ()
        :return: torch.tensor of size (1, n_batch_ext_sents)
        """
        n_articles = len(batch_extracted_sents)
        batch_values = list()

        # Iterate over each article
        for j in range(n_articles):
            # Obtain single article in batch and its corresponding extracted sentences:
            state = batch_state[j:j+1]
            mask = batch_mask[j:j+1]
            state, mask = self.add_stop_action(state, mask)
            mask = (1 - mask).bool().unsqueeze(0)
            state = self.bert_fine_tune(state)  # (1, n_doc_sents, tune_dim)
            extracted_sents = batch_extracted_sents[j]

            # Obtain initial input_embedding: "[CLS]"
            input_embedding = torch.tensor(self.bert_tokenizer.encode(
                START_OF_SENTENCE_TOKEN
            )).unsqueeze(0)
            input_embedding = self.bert_model(input_embedding)[0]
            input_embedding = self.bert_fine_tune(input_embedding)  # (1, 1, tune_dim)

            # Initialize hidden state:
            hidden = self.init_hidden.unsqueeze(0)  # (1, 1, tune_dim)

            # Iterate over each extracted sent: Use PREVIOUSLY extracted sentence to calculate value
            values = list()
            n_ext_sents = len(extracted_sents)
            for i in range(n_ext_sents):
                # Obtain context (attention weighted document embeddings)
                attention = torch.bmm(hidden, state.transpose(1, 2))  # (1, 1, n_doc_sents)
                attention[mask] = -1e6  # assign low attention to things to mask
                attention_weight = self.softmax(attention)  # (1, 1, n_doc_sents)
                context = torch.bmm(attention_weight, state)  # (1, 1, tune_dim)

                # Obtain new hidden state
                rnn_input = torch.cat([input_embedding, context], dim=2)  # (1, 1, tune_dim * 2)
                __, hidden = self.gru(rnn_input, hidden)

                # Calculate value
                value = self.value_layer(hidden)  # (1, 1, 1)
                values.append(value)

                # Obtain PREVIOUSLY extracted sentence
                input_embedding = extracted_sents[i].unsqueeze(0)
                input_embedding = self.bert_fine_tune(input_embedding)  # (1, 1, 8)

            batch_values = batch_values + values

        batch_values = torch.cat(batch_values).squeeze()  # (1, n_batch_ext_sents)
        return batch_values


class RLModel(torch.nn.Module):
    def __init__(
            self, extractor_model, abstractor_model, alpha=1e-4, gamma=0.99, batch_size=128
    ):
        super(RLModel, self).__init__()
        # Set attributes
        self.extractor_model = extractor_model
        self.abstractor_model = abstractor_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.rouge = Rouge()
        self.n_features = BERT_OUTPUT_SIZE

        # Hyper parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize weights
        self.n_hidden_units = 8
        self.policy_net = ActorCritic(
            extraction_model=self.extractor_model
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)

    def sample_actions(self, state, mask):
        """
        Returns an actions for an entire trajectory

        :param state:
        :param mask:
        :return:
        """
        dists, actions, values = self.policy_net.forward(state, mask)

        log_probs = list()
        entropys = list()
        for article_dists, article_actions in zip(dists, actions):
            log_prob, entropy = zip(*[
                (dist.log_prob(action), dist.entropy()) for dist, action in zip(article_dists, article_actions)
            ])

            log_prob = torch.cat(log_prob)
            entropy = torch.cat(entropy)

            log_probs.append(log_prob)
            entropys.append(entropy)

        log_probs = torch.cat(log_probs)
        entropys = torch.cat(entropys)

        # Return action
        return actions, log_probs, entropys, values

    def get_gae(self, rewards, values, last_action_masks, lmbda=0.95):
        """
        Calculates "generalized advantage estimate" (GAE).
        :param rewards:
        :param values:
        :param last_action_masks:
        :param lmbda:
        :return:
        """
        dummy_next_value = 0  # should get masked out
        values = torch.cat([values, torch.tensor([dummy_next_value]).float()])
        last_action_masks = ~last_action_masks  # is not terminal

        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * last_action_masks[step] - values[step]
            gae = delta + self.gamma * lmbda * last_action_masks[step] * gae
            returns.insert(0, (gae + values[step]).view(-1))

        returns = torch.cat(returns)
        return returns

    @staticmethod
    def select_random_batch(actions, log_probs, entropys, returns, advantages, mini_batch_size):
        random_indicies = np.random.randint(0, len(actions), mini_batch_size)

        batch_actions = actions[random_indicies]
        batch_log_probs = log_probs[random_indicies]
        batch_entropys = entropys[random_indicies]
        batch_returns = returns[random_indicies]
        batch_advantages = advantages[random_indicies]

        return batch_actions, batch_log_probs, batch_entropys, batch_returns, batch_advantages

    def create_abstracted_sentences(
            self,
            batch_actions,
            source_documents,
            stop_action_index,
            teacher_forcing_pct=0.0,
            target_summary_embeddings=None
    ):
        """
        :param batch_actions:
        :param source_documents:
        :param stop_action_index:
        :param teacher_forcing_pct:
        :param target_summary_embeddings:
        :return:
        """
        # Obtain embeddings
        actions = list()
        for trajectory_actions in batch_actions:
            if trajectory_actions[-1] == stop_action_index:
                actions.append(trajectory_actions[:-1])
            else:
                actions.append(trajectory_actions)

        extracted_sentences = [
            np.array(source_doc)[a].tolist() for source_doc, a in zip(source_documents, actions)
        ]
        source_document_embeddings, __, __ = obtain_word_embeddings(
            self.extractor_model.bert_model,
            self.extractor_model.bert_tokenizer,
            extracted_sentences,
            static_embeddings=False
        )
        # Obtain extraction probability for each word in vocabulary
        word_probabilities = self.abstractor_model.forward(
            source_document_embeddings,
            target_summary_embeddings,
            teacher_forcing_pct=teacher_forcing_pct
        )[0]  # (batch_size, n_target_words, vocab_size)

        # Get words with highest probability per time step
        chosen_words = torch.argmax(word_probabilities, dim=2)

        return chosen_words, word_probabilities

    def determine_rewards(self, actions, output_sentence, target_sentence, target_mask):
        """
        Uses ROUGE to calculate reward
        :param actions:
        :param output_sentence:
        :param target_sentence:
        :param target_mask:
        :return:
        Todo: Reward for each action, not just a the end
        Todo: Reward should require_grad
        """
        rewards = [torch.zeros(ext_sents.shape) for ext_sents in actions]
        n_target_words = target_mask[:, :, 0].sum(dim=1)

        # Convert target sentences indicies into words
        target_sentence = self.convert_to_words(target_sentence, n_target_words)

        # Convert output sentence indicies into words
        output_sentence = self.convert_to_words(output_sentence, n_target_words)

        # Compute ROUGE-L score
        scores = self.rouge.get_scores(output_sentence, target_sentence)
        scores = [score['rouge-l']['f'] for score in scores]

        for doc_idx in range(len(rewards)):
            rewards[doc_idx][len(actions[doc_idx]) - 1] = scores[doc_idx]

        return rewards

    def convert_to_words(self, sentence_indicies, n_target_words):
        sentence_indicies = torch.roll(sentence_indicies, dims=1, shifts=-1)  # shift left
        bert_tokenizer = self.abstractor_model.bert_tokenizer

        sentence_words = [
            bert_tokenizer.convert_ids_to_tokens(idx) for idx in sentence_indicies.tolist()
        ]

        n_target_words = n_target_words.int()
        sentence_words = [" ".join(s[:n_words]) for s, n_words in zip(sentence_words, n_target_words)]

        return sentence_words

    @staticmethod
    def last_action_mask(actions):
        action_mask = [torch.zeros(ext_sents.shape) for ext_sents in actions]

        for doc_idx in range(len(action_mask)):
            action_mask[doc_idx][len(actions[doc_idx]) - 1] = 1

        return torch.cat(action_mask).bool()

    def calc_abstractor_loss(self, word_probabilities, target_tokens, target_mask):
        """
        :param word_probabilities:
        :param target_tokens:
        :param target_mask:
        :return:
        """
        # Shift target tokens and format masks
        target_mask = torch.flatten(target_mask[:, :, 0])

        target_tokens = torch.roll(target_tokens, dims=1, shifts=-1)  # shift left
        target_tokens[:, -1] = 0
        target_tokens = torch.flatten(target_tokens)

        loss = nll_loss(word_probabilities.view(-1, self.abstractor_model.vocab_size), target_tokens)
        loss = loss * target_mask
        loss = loss.sum()

        return loss

    def update(self, trajectory, word_probabilities, target_tokens, target_mask):
        """
        :param trajectory:
        :param word_probabilities:
        :param target_tokens:
        :param target_mask:
        :return:
        """
        # Extract from trajectory
        actions, rewards, log_probs, entropys, values, last_action_masks = trajectory
        returns = self.get_gae(
            rewards=rewards,
            values=values,
            last_action_masks=last_action_masks
        ).detach()

        # Obtain advantages
        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        abstractor_loss = self.calc_abstractor_loss(word_probabilities, target_tokens, target_mask)
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropys.mean() + abstractor_loss
        print(f"RL Loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return


class Logit(torch.nn.Module):
    def __init__(self):
        super(Logit, self).__init__()

    @staticmethod
    def forward(x):
        x = torch.max(torch.tensor(1e-6), x)
        z = torch.log(x / (1 - x))
        return z
