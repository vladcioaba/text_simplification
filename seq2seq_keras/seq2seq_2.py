from __future__ import print_function

from keras.models import Model
from keras.models import load_model
from keras.layers import Input, LSTM, Dense
import numpy as np

def initCorpus(num_samples = 10000, data_path = 'fra.txt'):
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_words = set()
    target_words = set()
    max_encoder_seq_length = 0
    max_decoder_seq_length = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for i in range(0, len(lines)-1, 3):
        num_samples = num_samples - 1;
        if (num_samples <= 0):
            break;

        input_text = lines[i].split(' ')
        target_text = lines[i+1].split(' ')
        target_text = ['\t'] + target_text + ['\n']
        input_texts.append(input_text)
        target_texts.append(target_text)
        max_encoder_seq_length = max(len(input_text), max_encoder_seq_length)
        for word in input_text:
            if word not in input_words:
                input_words.add(word)
        max_decoder_seq_length = max(len(target_text), max_decoder_seq_length)
        for word in target_text:
            if word not in target_words:
                target_words.add(word)
                
    input_words = sorted(list(input_words))
    target_words = sorted(list(target_words))
    num_encoder_tokens = len(input_words)
    num_decoder_tokens = len(target_words)

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(word, i) for i, word in enumerate(input_words)])
    target_token_index = dict(
        [(word, i) for i, word in enumerate(target_words)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    print("encoder_input_data",encoder_input_data.shape)
    print("decoder_input_data",decoder_input_data.shape)
    print("decoder_target_data",decoder_target_data.shape)
    print("**************")

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, word in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[word]] = 1.    # earlier we have made a numpy array of zeros so here we are one hot encoding by puttong 1 on places words are found.
        #encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
        for t, word in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[word]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start word.
                decoder_target_data[i, t - 1, target_token_index[word]] = 1.
        #decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
        #decoder_target_data[i, t:, target_token_index[' ']] = 1.

    return input_texts, target_texts, input_words, target_words, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data

def trainAndSave(input_texts = [],
          target_texts = [],
          batch_size = 64,
          epochs = 100,
          latent_dim = 512,
          save_file = 's2s.h5',
          num_encoder_tokens = 0,
          num_decoder_tokens = 0,
          max_encoder_seq_length = 0,
          max_decoder_seq_length = 0,
          encoder_input_data = None, 
          decoder_input_data = None,
          decoder_target_data = None):

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
    # Save model
    model.save(save_file)

def load(save_file = 's2s.h5',
         latent_dim = 512,
         num_encoder_tokens = 0,
         num_decoder_tokens = 0):
    model = load_model(save_file)

    encoder_inputs = model.input[0]
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = model.layers[2].output
    encoder_states = [state_h, state_c]
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs = model.input[1]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def decodeSequence(encoder_model,
                    decoder_model,
                    num_encoder_tokens,
                    num_decoder_tokens,
                    input_token_index,
                    target_token_index,
                    reverse_input_word_index,
                    reverse_target_word_index,
                    max_encoder_seq_length,
                    max_decoder_seq_length,
                    input_text):
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, word in enumerate(input_text):
        input_seq[0, t, input_token_index[word]] = 1.
    input_seq[0, t + 1:, input_token_index[' ']] = 1.

    states_value = encoder_model.predict(input_seq)    
    decoded_sentence_in = ''
    for token_index in input_seq[0]:
        result = np.where(token_index == 1)
        decoded_sentence_in += reverse_input_word_index[result[0][0]]
    print(decoded_sentence_in)
    print("******")

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop word.
        if (sampled_word == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    print(decoded_sentence)

    return decoded_sentence
