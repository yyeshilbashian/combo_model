from keras.layers import Embedding, regularizers, Conv1D, Input, GlobalMaxPooling1D, Dense
from keras.models import Model
import numpy as np
import keras

char_max_len = 30
char_embed_size = 64

def create(vocab_size, words_count):
    char_embed = Embedding(
        input_dim=vocab_size,
        output_dim=char_embed_size,
        mask_zero=False,
        weights=None,
        embeddings_regularizer=regularizers.l2(0.00001)
    )

    conv1 = Conv1D(
        filters=char_embed_size * 8,
        kernel_size=3,
        strides=1,
        dilation_rate=1,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(0.000001),
        bias_regularizer=regularizers.l2(0.000001),
        name="char_conv1"
    )

    conv2 = Conv1D(
        filters=char_embed_size * 4,
        kernel_size=3,
        strides=1,
        dilation_rate=2,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(0.000001),
        bias_regularizer=regularizers.l2(0.000001),
        name="char_conv2"
    )

    conv3 = Conv1D(
        filters=char_embed_size,
        kernel_size=3,
        strides=1,
        dilation_rate=4,
        activation=None,
        padding='same',
        kernel_regularizer=regularizers.l2(0.000001),
        bias_regularizer=regularizers.l2(0.000001),
        name="char_conv3"
    )

    single_input_char = Input(shape = (char_max_len + 2, ))
    char_emb = char_embed(single_input_char)
    char_emb = conv3(conv2(conv1(char_emb)))
    char_emb = GlobalMaxPooling1D()(char_emb)
    dense_softmax_emb = Dense(1, activation="softmax", name="output")(char_emb)
    char_model = Model(inputs=[single_input_char], outputs=dense_softmax_emb)
    return char_model


def char2id(words, vocabulary):
    new_words = []
    for word in words:
        new_word = [3]
        for c in word:
            new_word.append(vocabulary[c])
        new_word.append(4)
        new_words.append(new_word)
    return new_words

def pad(words):
    for word in words:
        if len(word) < char_max_len + 2:
            for _ in range(char_max_len + 2 - len(word)):
                word.append(0)

def word2id(words, words1):
    word2id = {}
    id = 3
    for word in words+words1:
        if word not in word2id:
            word2id[word] = id
            id = id + 1
    ids = []
    for word in words:
        ids.append(word2id[word])
    ids1 = []
    for word in words1:
        ids1.append(word2id[word])
    return ids, ids1

def main():
    file = '/home/lab5/COMBO/uk.txt'
    file1 = '/home/lab5/COMBO/uk1.txt'
    with open(file) as f:
        line = f.readline()
        words = line.split()
    with open(file1) as f:
        line = f.readline()
        words1 = line.split()
    vocabulary = {}
    id = 5
    for word in words+words1:
        for c in word:
            if not c in vocabulary:
                vocabulary[c] = id
                id = id + 1

    word_char_id = char2id(words, vocabulary)
    word_char_id1 = char2id(words1, vocabulary)
    pad(word_char_id)
    pad(word_char_id1)
    word_ids, word_ids1 = word2id(words, words1)
    model = create(len(vocabulary), len(words+words1))
    model.compile(keras.optimizers.RMSprop(lr=0.00001), loss=keras.losses.mean_absolute_percentage_error, metrics=["accuracy"])
    x = np.array(word_char_id)
    x1 = np.array(word_char_id1)
    y = word_ids
    y1 = word_ids1
    hist = model.fit(x, y, batch_size=4, epochs=5, validation_data=(x1,y1), shuffle=True)
    print(hist.history)



if __name__ == '__main__':
    main()
