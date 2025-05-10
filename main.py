import numpy as np
from collections import defaultdict

class HMM:
    def __init__(self):
        self.states = set()
        self.vocab = set()
        self.start_probs = defaultdict(float)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        self.emit_probs = defaultdict(lambda: defaultdict(float))

    def train(self, tagged_sentences):
        tag_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        start_counts = defaultdict(int)

        for sentence in tagged_sentences:
            prev_tag = None
            for i, (word, tag) in enumerate(sentence):
                self.states.add(tag)
                self.vocab.add(word)
                tag_counts[tag] += 1
                emission_counts[tag][word] += 1

                if i == 0:
                    start_counts[tag] += 1
                if prev_tag is not None:
                    transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        total_starts = sum(start_counts.values())
        for tag in self.states:
            self.start_probs[tag] = start_counts[tag] / total_starts if total_starts != 0 else 0.0
            total_transitions = sum(transition_counts[tag].values())
            for next_tag in self.states:
                if total_transitions == 0:
                    self.trans_probs[tag][next_tag] = 0.0
                else:
                    self.trans_probs[tag][next_tag] = transition_counts[tag][next_tag] / total_transitions
            total_emissions = sum(emission_counts[tag].values())
            for word in self.vocab:
                if total_emissions == 0:
                    self.emit_probs[tag][word] = 0.0
                else:
                    self.emit_probs[tag][word] = emission_counts[tag][word] / total_emissions

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        for tag in self.states:
            emit_prob = self.emit_probs[tag].get(sentence[0], 1e-6)
            V[0][tag] = self.start_probs[tag] * emit_prob
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in self.states:
                max_prob = -1
                best_prev_tag = None
                for prev_tag in self.states:
                    trans_prob = self.trans_probs[prev_tag].get(curr_tag, 1e-6)
                    prob = V[t-1][prev_tag] * trans_prob
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag
                emit_prob = self.emit_probs[curr_tag].get(sentence[t], 1e-6)
                V[t][curr_tag] = max_prob * emit_prob
                new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

            path = new_path

        max_prob = -1
        best_tag = None
        for tag in self.states:
            if V[-1][tag] > max_prob:
                max_prob = V[-1][tag]
                best_tag = tag

        return path.get(best_tag, [])

def parse_tagged_sentence(sentence):
    tokens = sentence.split()
    tagged = []
    for token in tokens:
        word, tag = token.split('_')
        tagged.append((word, tag))
    return tagged

# Training data
training_data = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

# Prepare tagged sentences
tagged_sentences = [parse_tagged_sentence(sent) for sent in training_data]

# Train HMM model
model = HMM()
model.train(tagged_sentences)

# Test sentences
test_sentences = [
    ['The', 'cat', 'meows'],
    ['My', 'dog', 'barks', 'loudly']
]

# Predict POS tags
for idx, sentence in enumerate(test_sentences):
    predicted_tags = model.viterbi(sentence)
    print(f"Test Sentence {idx + 1}: {' '.join(sentence)}")
    print("Predicted POS Tags:", predicted_tags)
    print()
