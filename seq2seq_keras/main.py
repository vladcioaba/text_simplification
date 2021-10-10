from seq2seq_2 import initCorpus
from seq2seq_2 import trainAndSave
from seq2seq_2 import load
from seq2seq_2 import decodeSequence
import sys
import os.path


def main():
    input_texts, target_texts, input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data = initCorpus(num_samples = 10000, data_path = sys.argv[1])
    saved_model = 's2s.h5'
    if not os.path.isfile(saved_model):
        trainAndSave(
              input_texts = input_texts,
              target_texts = target_texts,
              batch_size = 64,
              epochs = (int(sys.argv[3]) if len(sys.argv) > 2 else 100),
              latent_dim = 256,
              save_file = saved_model,
              num_encoder_tokens = num_encoder_tokens,
              num_decoder_tokens = num_decoder_tokens,
              max_encoder_seq_length = max_encoder_seq_length,
              max_decoder_seq_length = max_decoder_seq_length,
              encoder_input_data = encoder_input_data,
              decoder_input_data = decoder_input_data,
              decoder_target_data = decoder_target_data)
    enc, dec = load(
                    save_file = saved_model,
                    latent_dim = 512,
                    num_encoder_tokens = num_encoder_tokens,
                    num_decoder_tokens = num_decoder_tokens)

    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    sen_out = decodeSequence(enc, dec,
                            num_encoder_tokens, num_decoder_tokens,
                            input_token_index, target_token_index,
                            reverse_input_char_index, reverse_target_char_index, 
                            max_encoder_seq_length, max_decoder_seq_length,
                            sys.argv[2])

if __name__ == "__main__":
    main()