# Simple-Pos-Tagging

Developed a BILSTM model for part-of-speech (POS) tagging, using a modified Brown corpus as training data. To simplify the problem, we will use a tagset which is composed of 11 tags. i.e. Noun, Pronoun, Verb, Adjective, Adverb, Conjunction, Preposition, Determiner, Number, Punctuation and Other. User can enter their own sentences to get its POS tags.

# Description

These are the following example test sentences:

1.The Secretariat is expected to race tomorrow .

2.People continue to enquire the reason for the race for outer space .

3.the planet jupiter and its moons are in effect a mini solar system .

(You can also use your own.)

# How to run

# Output
output for first sentence : ['DETERMINER', 'NOUN', 'VERB', 'VERB', 'PREPOSITION', 'NOUN', 'NOUN', 'PUNCT']
output for second sentence : ['NOUN', 'VERB', 'X', 'VERB', 'DETERMINER', 'NOUN', 'PREPOSITION', 'DETERMINER', 'NOUN', 'PREPOSITION', 'NOUN', 'NOUN', 'PUNCT']
output for third sentence : ['DETERMINER', 'NOUN', 'NOUN', 'CONJUNCTION', 'PRONOUN', 'NOUN', 'VERB', 'PREPOSITION', 'NOUN', 'DETERMINER', 'NOUN', 'DETERMINER', 'NOUN', 'PUNCT']
