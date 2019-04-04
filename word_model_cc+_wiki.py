from keras.layers import Embedding, regularizers, Conv1D, Input, GlobalMaxPooling1D, Dense
from keras.models import Model
import numpy as np
import keras
import types
import pickle

train_set1 = '/home/lab5/COMBO/uk_crawl_train.txt'
train_set2 = '/home/lab5/COMBO/uk_wiki_train.txt'

char_max_len = 30
char_embed_size = 64
batch_size = 512
steps_count = 128
word_id = 2
char_id = 5

word2id = {
    '__PADDING__': 0,
    '__UNKNOWN__': 1
}


char2id = {
        '__PADDING__': 0,
        '__UNKNOWN__': 1,
        '__ROOT__': 2,
        '__START__': 3,
        '__END__': 4
}


id2char = {}

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

def generate_batches(file):
    with open(file) as f:
        checker = batch_size
        corrector = ''
        while checker == batch_size:
            ready_char_batch = f.read(batch_size)
            checker = len(ready_char_batch)
            ready_char_batch = corrector + ready_char_batch
            ready_char_batch = ready_char_batch.split()
            corrector = ready_char_batch[-1]
            ready_char_batch.remove(ready_char_batch[-1])
            yield ready_char_batch


def generator(file):
    global word_id
    global char_id
    while True:
        gen = generate_batches(file)
        for batch in gen:
            enc = []
            list_of_emb = []
            for i in range(len(batch)):
                e = []
                if batch[i] not in word2id:
                    word2id[batch[i]] = word_id
                    word_id = word_id + 1
                list_of_emb.append(word2id[batch[i]])
                for char in batch[i]:
                    if char not in char2id:
                        char2id[char] = char_id
                        char_id = char_id + 1
                    e.append(char2id[char])
                enc.append(e)
            for i in range(len(enc)):
                enc[i] = enc[i][:char_max_len]
                enc[i].insert(0, char2id['__START__'])
                n = char_max_len - len(enc[i])
                padding = [char2id['__PADDING__']] * n
                enc[i].append(char2id['__END__'])
                enc[i].extend(padding)
            n = steps_count - len(list_of_emb)
            list_of_emb.extend(n*[word2id['__PADDING__']])
            enc.extend([char_max_len*[char2id['__PADDING__']]]*n)
            if n < 0:
                list_of_emb = list_of_emb[:steps_count]
                enc = enc[:steps_count]
            for c in char2id.keys():
                id2char.update({char2id[c] : c})
            yield (enc, list_of_emb)


def generator_final(gen):

    while True:
        for (words, enc) in gen:
            x = np.array(words)
            y = np.array(keras.utils.to_categorical(enc, word_id, int))
            yield x, y


def main():
    gen = generator(train_set1)
    try:
        while True:
            next(gen)
    except:
        pass


    print(char2id, id2char)
    with open('id2char.pickle', 'wb') as handle:
        pickle.dump(id2char, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('char2id.pickle', 'wb') as handle:
        pickle.dump(char2id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = create(len(char2id), len(word2id))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.002, clipvalue=5.0, beta_1=0.9, beta_2=0.9, decay=1e-4),
                  loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

    hist = model.fit_generator(generator_final(gen),epochs=200, steps_per_epoch=steps_count)

    model.save_weights("model.txt")
    print(hist.history)


if __name__ == '__main__':
    main()