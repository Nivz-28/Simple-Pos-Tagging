# Simple-Pos-Tagging

Developed a BILSTM model for part-of-speech (POS) tagging, using a modified Brown corpus as training data. To simplify the problem, we will use a tagset which is composed of 11 tags. i.e. Noun, Pronoun, Verb, Adjective, Adverb, Conjunction, Preposition, Determiner, Number, Punctuation and Other. User can enter their own sentences to get its POS tags.

# Description

The dataset used is a modified brown corpus, the full folder is available here: https://github.com/Nivz-28/Simple-Pos-Tagging/tree/main/brown_corpus_modified

These are the following example test sentences:

1.The Secretariat is expected to race tomorrow .

2.People continue to enquire the reason for the race for outer space .

3.the planet jupiter and its moons are in effect a mini solar system .

(You can also use your own.)

# How to run
To execute type this in terminal:

-> python main.py

Let the epochs run, after which these will pop up, enter these as well:

Enter the number of test sentences:

Enter a test sentence:


# Output
1.output for first sentence : ['DETERMINER', 'NOUN', 'VERB', 'VERB', 'PREPOSITION', 'NOUN', 'NOUN', 'PUNCT']


2.output for second sentence : ['NOUN', 'VERB', 'X', 'VERB', 'DETERMINER', 'NOUN', 'PREPOSITION', 'DETERMINER', 'NOUN', 'PREPOSITION', 'NOUN', 'NOUN', 'PUNCT']

3.output for third sentence : ['DETERMINER', 'NOUN', 'NOUN', 'CONJUNCTION', 'PRONOUN', 'NOUN', 'VERB', 'PREPOSITION', 'NOUN', 'DETERMINER', 'NOUN', 'DETERMINER', 'NOUN', 'PUNCT']
