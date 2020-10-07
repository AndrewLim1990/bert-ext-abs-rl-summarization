from abstractor.train import nll_loss
from bert.utils import START_OF_SENTENCE_TOKEN
from bert.utils import obtain_word_embeddings
from extractor.utils import BERT_OUTPUT_SIZE
from rouge import Rouge
from torch.distributions import Categorical
from utils import batched_index_select
from utils import logit

import numpy as np
import torch


class ActorCritic(torch.nn.Module):
    def __init__(self, extraction_model):
        """
        A pytorch torch.nn.Module used to determine which sentences to extract from a source document to be used for
        summarization

        :param extraction_model: torch.nn.Module used to extract sentences from documents
        """
        super(ActorCritic, self).__init__()
        # Bert embeddings:
        self.bert_tokenizer = extraction_model.bert_tokenizer
        self.bert_model = extraction_model.bert_model

        # Convert Bert embeddings to custom embeddings
        self.tune_dim = 8
        self.bert_fine_tune = torch.nn.Linear(BERT_OUTPUT_SIZE, self.tune_dim, bias=False)

        # Embeddings to fine tune:
        self.init_hidden = torch.nn.Parameter(torch.rand(1, self.tune_dim), requires_grad=True)
        self.stop_embedding = torch.nn.Parameter(torch.rand(extraction_model.input_dim), requires_grad=True)

        # Actor Layer
        self.extraction_model = extraction_model  # returns prob of extraction per sentence
        self.convert_to_dist = torch.nn.Sequential(
            Logit(),
            torch.nn.Softmax(dim=-1)
        )
        self.softmax = torch.nn.Softmax(dim=2)

        # Critic Layer
        self.gru = torch.nn.GRU(self.tune_dim * 2, self.tune_dim, batch_first=True)
        self.value_layer = torch.nn.Linear(self.tune_dim, 1)
        self.max_n_ext_sents = 4

    def add_stop_action(self, state, mask=None):
        """
        Extends the input 'state' by placing self.stop_embedding as an additional state/action. Also adjusts 'mask'
        appropriately such that the additional stop_embedding will not be masked out.

        :param state:   A torch.tensor of sentence embeddings from source documents.
                        Shape: (batch_size, n_doc_sentences, embedding_dim)
        :param mask:    A torch.tensor of booleans indicating whether or not the document within the batch actually
                        has the sentence. This is necessary because we've batched multiple documents together of
                        various lengths.
        :return:        Returns a tuple of state and mask with the additional embedding and mask boolean values
        """
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
        """
        :param state:           A torch.tensor() containing sentence embeddings for each acticle.
                                Shape: (batch_size, n_doc_sentences, embedding_dim)
        :param mask:            A torch.tensor of booleans indicating whether or not the document within the batch
                                actually has the sentence. This is necessary because we've batched multiple documents
                                together of various lengths.
        :param n_label_sents:   An optional list containing number of extracted sentences in summary labels (oracle)
        :return:                A tuple containing
                                 - action_dists:    list(Categorical()) containing categorical distributions. Each entry
                                                    represents the distribution amongst sentences to extract at a given
                                                    step for all batches.
                                 - action_indicies: torch.tensor() containing the indicies of extracted sentences
                                                    Shape: (batch_size, n_extracted_sentences, embedding_dim)
                                 - values:          A torch.tensor() where each value represents the predicted "value"
                                                    of being in the input "state". Shape: (
                                 - n_ext_sents:     A torch.tensor() where entries show # of sentences extracted per
                                                    sample
        """
        action_dists, action_indicies, ext_sents, n_ext_sents = self.actor_layer(state, mask, n_label_sents)
        values = self.critic_layer(state, mask, ext_sents)
        return action_dists, action_indicies, values, n_ext_sents

    def actor_layer(self, batch_state, mask, n_label_sents=None):
        """
        Determines which sentences to extract for each of the documents represented by batch_state

        :param batch_state:     A torch.tensor representing sentence embeddings of each document within the batch.
                                Shape: (batch_size, n_doc_sentences, embedding_dim)
        :param mask:            A torch.tensor of booleans indicating whether or not the document within the batch
                                actually has the sentence. This is necessary because we've batched multiple documents
                                together of various lengths.
        :param n_label_sents:   An optional list containing number of extracted sentences in summary labels (oracle)
        :return:                A tuple containing:
                                 - action_dists: list(Categorical()) containing categorical distributions. Each entry
                                                 represents the distribution amongst sentences to extract at a given
                                                 step for all batches.
                                 - action_indices: torch.tensor() containing the indicies of extracted sentences
                                                    Shape: (batch_size, n_extracted_sentences, embedding_dim)
                                 - ext_sents: A torch.tensor() containing extracted sentence embeddings.
                                              Shape: (batch_size, n_extracted_sentences, embedding_dim)
                                 - n_ext_sents: A torch.tensor() where entries show # of sentences extracted per sample
        """
        # Obtain distribution amongst actions
        batch_state, mask = self.add_stop_action(batch_state, mask)

        # Obtain number of samples in batch
        batch_size = batch_state.shape[0]
        max_n_doc_sents = batch_state.shape[1]
        embedding_dim = batch_state.shape[2]

        # Obtain maximum number of sentences to extract
        if n_label_sents is None:
            n_doc_sents = mask.sum(dim=1)
            batch_max_n_ext_sents = torch.tensor([self.max_n_ext_sents] * batch_size)
            batch_max_n_ext_sents = torch.min(batch_max_n_ext_sents.float(), n_doc_sents)
        else:
            batch_max_n_ext_sents = n_label_sents

        # Create variables to stop extraction loop
        max_n_ext_sents = batch_max_n_ext_sents.max()  # Maximum number of sentences to extract
        stop_action_idx = max_n_doc_sents - 1  # Previously appended stop_action embedding
        is_stop_action = torch.zeros(batch_size).bool()

        src_doc_lengths = torch.sum(mask, dim=1)
        batch_state = torch.nn.utils.rnn.pack_padded_sequence(
            batch_state,
            lengths=src_doc_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # Extraction loop
        action_indices, ext_sents, action_dists, stop_action_list = list(), list(), list(), list()
        n_ext_sents = 0
        is_first_sent = True
        extraction_labels = None
        while True:
            # Obtain distribution amongst sentences to extract
            if is_first_sent:
                action_probs, __ = self.extraction_model.forward(batch_state, mask)
                is_first_sent = False
            else:
                action_probs, __ = self.extraction_model.forward(
                    sent_embeddings=batch_state,
                    sent_mask=mask,
                    extraction_indicator=extraction_labels,
                    use_init_embedding=True
                )
            action_probs = action_probs[:, -1:, :]
            action_dist = Categorical(action_probs)

            # Sample sentence to extract
            ext_sent_indices = action_dist.sample().T

            # Embeddings of sentences to extract
            ext_sent_embeddings = batched_index_select(batch_state, 1, ext_sent_indices)

            # Collect
            action_dists.append(action_dist)
            ext_sents.append(ext_sent_embeddings)
            action_indices.append(ext_sent_indices)

            # Form extraction_labels
            extraction_labels = torch.zeros(batch_size, max_n_doc_sents)
            already_ext_indices = torch.cat(action_indices)
            extraction_labels[torch.arange(batch_size), already_ext_indices] = 1

            # Track number of sentences extracted from article
            n_ext_sents = n_ext_sents + 1

            # Check to see if should stop extracting sentences
            # Todo: Fix this, the mask ALWAYS masks out the stop action...
            is_stop_action = is_stop_action | (ext_sent_indices >= stop_action_idx)
            stop_action_list.append(is_stop_action)
            all_samples_stop = torch.sum(is_stop_action) >= batch_size
            is_long_enough = n_ext_sents >= max_n_ext_sents
            if all_samples_stop or is_long_enough:
                break

        action_indices = torch.stack(action_indices).T.squeeze(1)
        n_ext_sents = (~torch.stack(stop_action_list).squeeze(1).T).sum(dim=1)
        ext_sents = torch.stack(ext_sents).transpose(0, 1).squeeze()
        return action_dists, action_indices, ext_sents, n_ext_sents

    def critic_layer(self, batch_state, batch_mask, extracted_sents):
        """
        Predicts the 'value' of the input state and action

        :param batch_state:     A torch.tensor representing sentence embeddings of each document within the batch.
                                Shape: (batch_size, n_doc_sentences, embedding_dim)
        :param batch_mask:      A torch.tensor of booleans indicating whether or not the document within the batch
                                actually has the sentence. This is necessary because we've batched multiple documents
                                together of various lengths. Shape: (batch_size, n_doc_sentences)
        :param extracted_sents: A torch.tensor containing extracted sentence embeddings.
                                Shape: (batch_size, n_extracted_sentences, embedding_dim)
        :return: A torch.tensor containing estimated values. Shape (batch_size, n_extracted_sentences)
        """
        is_missing_batch_dim = extracted_sents.dim() < 3
        if is_missing_batch_dim:
            extracted_sents = extracted_sents.unsqueeze(0)

        batch_size = len(extracted_sents)

        # Iterate over each article
        state, mask = self.add_stop_action(batch_state, batch_mask)
        mask = (1 - mask).bool().unsqueeze(1)
        state = self.bert_fine_tune(state)  # (batch_size, n_doc_sents + 1, tune_dim)

        # Obtain initial input_embedding: "[CLS]"
        input_embedding = torch.tensor(
            self.bert_tokenizer.encode(START_OF_SENTENCE_TOKEN) * batch_size
        ).unsqueeze(1)
        input_embedding = self.bert_model(input_embedding)[0]
        input_embedding = self.bert_fine_tune(input_embedding)  # (batch_size, 1, tune_dim)

        # Initialize hidden state:
        hidden = self.init_hidden.unsqueeze(0)
        hidden = hidden.repeat(batch_size, 1, 1)  # (batch_size, 1, tune_dim)

        # Obtain input_embeddings:
        input_embeddings = self.bert_fine_tune(extracted_sents)

        # Iterate over each extracted sent: Use PREVIOUSLY extracted sentence to calculate value
        values = list()
        max_ext_sents = input_embeddings.shape[1]
        for i in range(max_ext_sents):
            # Obtain context (attention weighted document embeddings)
            attention = torch.bmm(hidden, state.transpose(1, 2))  # (batch_size, 1, n_doc_sents)
            attention = attention + (mask * -1e16)  # assign low attention to things to mask
            attention_weight = self.softmax(attention)  # (batch_size, 1, n_doc_sents)
            context = torch.bmm(attention_weight, state)  # (batch_size, 1, tune_dim)

            # Obtain new hidden state
            rnn_input = torch.cat([input_embedding, context], dim=2)  # (batch_size, 1, tune_dim * 2)
            __, hidden = self.gru(rnn_input, hidden.transpose(0, 1))
            hidden = hidden.transpose(0, 1)  # (batch_size, 1, tune_dim)

            # Calculate value
            value = self.value_layer(hidden)  # (batch_size, 1, 1)
            values.append(value)

            # Obtain PREVIOUSLY extracted sentence
            input_embedding = input_embeddings[:, i:i+1, :]

        # Clean shape of values
        values = torch.cat(values, dim=2).squeeze()
        if is_missing_batch_dim:
            values = values.unsqueeze(0)

        return values


class RLModel(torch.nn.Module):
    def __init__(
            self, extractor_model, abstractor_model, alpha=1e-4, gamma=0.99, batch_size=128
    ):
        """
        A torch.nn.Module meant to output which sentences to extract in order to form high quality abstract summaries

        :param extractor_model:     A pretrained torch.nn.Module to select sentences to extract from source documents
        :param abstractor_model:    A pretrained torch.nn.Module to convert extracted sentences into abstract summaries
        :param alpha:               Learning rate
        :param gamma:               Reinforcement learning discount factor
        :param batch_size:          Number of documents to include in each training batch
        """
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
        Returns actions (extracted sentence indicies)

        :param state: A torch.tensor of sentence embeddings for each document.
                      Shape: (batch_size, n_doc_sentences, embedding_dim)
        :param mask:  A torch.tensor of booleans indicating the existance of a sentence within the batch.
                      Shape: (batch_size, n_doc_sentences)
        :return:      A tuple consisting of:
                      - actions:     torch.tensor containing indicies of sentence to extract.
                                     Shape: (batch_size, n_ext_sentences)
                      - log_probs:   torch.tensor containing log probability of extracting the extracted sentences
                                     Shape: (batch_size, n_ext_sentences)
                      - entropys:    torch.tensor containing entropy of the distribution used to obtain actions
                                     Shape: (batch_size, n_ext_sentences)
                      - values:      A torch.tensor containing estimated values of being in each state and taking
                                     returned actions. Shape: (batch_size, n_extracted_sentences)
                      - n_ext_sents: A torch.tensor containing the # of sentences extracted per source_document
                                     Shape: batch_size
        """
        dists, actions, values, n_ext_sents = self.policy_net.forward(state, mask)

        log_probs = list()
        entropys = list()
        for article_dists, article_actions in zip(dists, actions.T):
            log_prob = article_dists.log_prob(article_actions)
            entropy = article_dists.entropy()

            log_probs.append(log_prob)
            entropys.append(entropy)

        log_probs = torch.stack(log_probs).T
        entropys = torch.stack(entropys).T

        # Return action
        return actions, log_probs, entropys, values, n_ext_sents

    def create_abstracted_sentences(
            self,
            batch_actions,
            source_documents,
            n_ext_sents,
            teacher_forcing_pct=0.0,
            target_summary_embeddings=None,
    ):
        """
        Creates a summary from extracted sentences indicated by batch_actions

        :param batch_actions: A torch.tensor containing the indicies of sentences to extract
        :param source_documents: A list(list(document_sentences))
        :param n_ext_sents: A torch.tensor containing the # of sentences extracted per document
        :param teacher_forcing_pct: Percentage of the time to use directly use target_summary_embeddings
        :param target_summary_embeddings: A torch.tensor containing embeddings of each word within the target summary
                                          (oracle). Shape: (batch_size, n_summary_label_words, embedding_dim)
        :return: A tuple containing:
                - chosen_words: torch.tensor containing corpus indicies of words to use in summary.
                                Shape: (batch_size, n_summary_predicted_words)
                - word_probabilities: torch.tensor containing the probability of extracting each word in corpus.
                                      Shape: (batch_size, n_summary_predicted_words, n_words_in_corpus)
        """
        # Obtain embeddings
        actions = [action_indicies[:n_ext_sent] for action_indicies, n_ext_sent in zip(batch_actions, n_ext_sents)]

        # Obtain actual string sentences that were exctracted
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

    def determine_rewards(self, n_ext_sents, n_actions, output_sentence, target_sentence, target_mask):
        """
        Uses ROUGE to calculate reward

        :param n_ext_sents:     A torch.tensor with # of sentences extracted
        :param n_actions:       An int indicating the number of sentences extracted
        :param output_sentence: A torch.tensor containing corpus word indicies that have been chosen
        :param target_sentence: A torch.tensor containing corpur word indicies of label (oracle)
        :param target_mask:     A torch.tensor of bools indicating if word is present in summary
                                (required because working in batches)
        :return:                A torch.tensor containing the rewards for each extracted sentence
                                Shape: (batch_size, n_extracted_sentences)
        """
        n_target_words = target_mask[:, :, 0].sum(dim=1)

        # Convert target sentences indicies into words
        target_sentence = self.convert_to_words(target_sentence, n_target_words)

        # Convert output sentence indicies into words
        output_sentence = self.convert_to_words(output_sentence, n_target_words)

        # Compute ROUGE-L score
        scores = self.rouge.get_scores(output_sentence, target_sentence)
        scores = [score['rouge-l']['f'] for score in scores]

        # Populate rewards tensor
        n_batches = len(n_ext_sents)
        rewards = torch.zeros(n_batches, n_actions)
        for doc_idx in range(n_batches):
            if n_ext_sents[doc_idx] > 0:
                last_ext_sentence_idx = n_ext_sents[doc_idx] - 1
                rewards[doc_idx][last_ext_sentence_idx] = scores[doc_idx]

        return rewards

    def convert_to_words(self, word_indicies, n_target_words):
        """
        Converts a batch of word indicies into a list of lists containing the string associated with each index

        :param word_indicies: A torch.tensor containing corpus word indicies. Shape: (batch_size, n_words)
        :param n_target_words: A torch.tensor indicating the number of words within each label summary (oracle)
        :return: A list(list(str)) containing the words
        """
        word_indicies = torch.roll(word_indicies, dims=1, shifts=-1)  # shift left
        bert_tokenizer = self.abstractor_model.bert_tokenizer

        sentence_words = [
            bert_tokenizer.convert_ids_to_tokens(idx) for idx in word_indicies.tolist()
        ]

        n_target_words = n_target_words.int()
        sentence_words = [" ".join(s[:n_words]) for s, n_words in zip(sentence_words, n_target_words)]

        return sentence_words

    def get_gae(self, rewards, values, n_ext_sents, lmbda=0.95):
        """
        Calculates "generalized advantage estimate" (GAE).

        :param rewards: A torch.tensor containing rewards obtained for each extracted sentence
        :param values: A torch.tensor containing predicted values of each state
        :param n_ext_sents: A torch.tensor containing number of sentences extracted per document within batcn
        :param lmbda: Hyper-parameter to adjust GAE calculation
        :return: A torch.tensor containing the GAE values for each sentence extracted
        """
        batch_size = values.shape[0]
        dummy_next_value = torch.zeros(batch_size, 1)  # should get masked out
        values = torch.cat([values, dummy_next_value], dim=1)

        gae = 0
        n_steps = rewards.shape[1]
        returns = []

        for step in reversed(range(n_steps)):
            not_last_action_mask = step != (n_ext_sents - 1)
            not_past_last_action_mask = ~(step > (n_ext_sents - 1))

            delta = rewards[:, step] + not_last_action_mask * self.gamma * values[:, step + 1] - values[:, step]
            gae = delta + not_last_action_mask * self.gamma * lmbda * gae
            gae = not_past_last_action_mask * (gae + values[:, step])
            gae = gae.view(1, -1)
            returns.insert(0, gae)

        returns = torch.cat(returns).T

        return returns

    def calc_abstractor_loss(self, word_probabilities, target_tokens, target_mask):
        """
        Calculates the loss associated with suboptimal words chosen during abstraction

        :param word_probabilities: torch.tensor containing probs of words being selected
        :param target_tokens: torch.tensor containing the corpus word index of each word in label summary (oracle)
        :param target_mask: torch.tensor indicating whether the word was present in summary (required because batches)
        :return: loss associated with with input word_probabilities
        """
        # Shift target tokens and format masks
        target_mask = torch.flatten(target_mask[:, :, 0])

        target_tokens = torch.roll(target_tokens, dims=1, shifts=-1)  # shift left
        target_tokens[:, -1] = 0
        target_tokens = torch.flatten(target_tokens)

        loss = nll_loss(word_probabilities.view(-1, self.abstractor_model.vocab_size), target_tokens)
        loss = loss * target_mask
        loss = loss.mean()

        return loss

    def update(
        self,
        rewards,
        log_probs,
        entropys,
        values,
        n_ext_sents,
        word_probabilities,
        target_tokens,
        target_mask
    ):
        """
        Updates the extraction (actor + critic) and abstraction layers iteratively

        :param rewards: torch.tensor containing rewards for each extracted sentence
        :param log_probs: torch.tensor containing log prob of extracting each word at each time step
        :param entropys: torch.tensor containing the entropy of the distribution at each time step of extraction
        :param values: torch.tensor containing predicted values of being in the associated state
        :param n_ext_sents: torch.tensor containing the # of sentence chosen for extraction
        :param word_probabilities: torch.tensor containing the probability of a word being chosen for abstraction
        :param target_tokens: torch.tensor containing indicies of corpus words in label summary (oracle)
        :param target_mask: torch.tensor indicating whether or not the word exists in the label summary
                            (required because working in batches)
        """
        returns = self.get_gae(
            rewards=rewards,
            values=values,
            n_ext_sents=n_ext_sents
        ).detach()

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        abstractor_loss = self.calc_abstractor_loss(word_probabilities, target_tokens, target_mask)
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropys.mean() + abstractor_loss
        print(f"RL Loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Logit(torch.nn.Module):
    def __init__(self):
        """
        Applies a logit function robust to x=0
        """
        super(Logit, self).__init__()

    @staticmethod
    def forward(x):
        """
        Applies a logit function robust to x=0
        :param x: float value satisfying: 0 <= x < 1
        :return: the logit of input x
        """
        return logit(x)
