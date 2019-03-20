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
    dense_softmax_emb = Dense(words_count, activation="softmax", name="output")(char_emb)
    char_model = Model(inputs=[single_input_char], outputs=dense_softmax_emb)
    return char_model


def char2id(words, vocabulary):
    new_words = []
    for word in words:
        new_word = [2]
        for c in word:
            if c in vocabulary:
                new_word.append(vocabulary[c])
            else:
                new_word.append(1)
        new_word.append(3)
        new_words.append(new_word)
    return new_words


def pad(words):
    for i in range(len(words)):
        if len(words[i]) < char_max_len + 2:
            for _ in range(char_max_len + 2 - len(words[i])):
                words[i].append(0)
        elif len(words[i]) > char_max_len + 2:
            words[i] = words[i][:char_max_len + 1]
            words[i].append(3)


def word2id(words, words1):
    word2id = {
        '__PADDING__': 0,
        '__UNKNOWN__': 1
    }
    id = 2
    for word in words:
        if word not in word2id:
            word2id[word] = id
            id = id + 1
    ids = []
    for word in words:
        ids.append(word2id[word])
    ids1 = []
    for word in words1:
        if word in word2id:
            ids1.append(word2id[word])
        else:
            ids1.append(1)
    return ids, ids1, id


def one_hot(ids, length):
    all_enc = []
    for id in ids:
        enc = [0] * length
        if id != 1:
            enc[id] = 1
        else:
            enc[1] = 1
        all_enc.append(enc)
    return all_enc


def main():
    file = '/home/lab5/combo_model/uk_train.txt'
    file1 = '/home/lab5/combo_model/uk_dev.txt'
    with open(file) as f:
        line = f.readline()
        words_train = line.split()
    with open(file1) as f:
        line = f.readline()
        words_dev = line.split()
    vocabulary = {
        '__PADDING__': 0,
        '__UNKNOWN__': 1,
        '__START__': 2,
        '__END__': 3
    }
    id = 4
    for word in words_train:
        for c in word:
            if not c in vocabulary:
                vocabulary[c] = id
                id = id + 1

    word_char_id_train = char2id(words_train, vocabulary)
    word_char_id_dev = char2id(words_dev, vocabulary)
    pad(word_char_id_train)
    pad(word_char_id_dev)
    for word in word_char_id_dev:
        if len(word) > 32:
            print(word, len(word))
    word_ids_train, word_ids_dev, length = word2id(words_train, words_dev)
    # one_hot_enc_train = one_hot(word_ids_train, length)
    # one_hot_enc_dev = one_hot(word_ids_dev, length)
    one_hot_enc_train = keras.utils.to_categorical(word_ids_train, length, int)
    one_hot_enc_dev = keras.utils.to_categorical(word_ids_dev, length, int)
    model = create(len(vocabulary), length)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    x = np.array(word_char_id_train)
    x1 = np.array(word_char_id_dev)
    y = np.array(one_hot_enc_train)
    y1 = np.array(one_hot_enc_dev)
    hist = model.fit(x, y, batch_size=8, epochs=10, validation_data=(x1,y1), shuffle=True)
    print(hist.history)

if __name__ == '__main__':
    main()
