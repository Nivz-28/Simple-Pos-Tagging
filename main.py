import pos_tagger

num_test_sentences = int(input("Enter the number of test sentences: "))
test_samples = []
for _ in range(num_test_sentences):
    test_sentence = input("Enter a test sentence: ").split()
    test_samples.append(test_sentence)

# Preprocess user-input sentences
test_samples_X = []
for s in test_samples:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')

# Use the model to predict tags for user-input sentences
predictions = model.predict(test_samples_X)

def logits_to_tokens(predictions, tag_index_mapping):
    token_predictions = []
    for pred_sequence in predictions:
        token_sequence = []
        for pred_logits in pred_sequence:
            pred_tag_index = int(np.argmax(pred_logits))
            pred_tag = tag_index_mapping[pred_tag_index]
            token_sequence.append(pred_tag)
        token_predictions.append(token_sequence)
    return token_predictions

# Convert predictions to token sequences
tag_index_mapping = {i: t for t, i in tag2index.items()}
token_predictions = logits_to_tokens(predictions, tag_index_mapping)

# Print token predictions
for i, tokens in enumerate(token_predictions):
    print(f"Predicted tags for test sentence {i + 1}:")
    print(tokens)

