import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Lendo a base de dados e transformando em um dataframe de pandas
data = pd.read_csv('./data/20191226-reviews.csv')
df = pd.DataFrame(data, columns=['rating','body'])

# Acrescentando os valores de sentimento para a análise
df['sentiments'] = df.rating.apply(lambda x: 0 if x > 0 and x <= 3 else 1)

# Separando o dataframe para treino
split = round(len(df)*0.8)
train_reviews = df['body'][:split]
train_label = df['sentiments'][:split]
test_reviews = df['body'][split:]
test_label = df['sentiments'][split:]

# Deixando todas as reviews como strings
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for row in train_reviews:
    training_sentences.append(str(row))
for row in train_label:
    training_labels.append(row)
for row in test_reviews:
    testing_sentences.append(str(row))
for row in test_label:
    testing_labels.append(row)

# Configs para o treino
vocab_size = 40000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'

# Tratando o texto, tranformando ele em uma sequencia de inteiros e criando o vocabulario
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Deixando as sequencias de texto com o mesmo tamanho
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)
testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length)

# Criando o modelo keras
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
num_epochs = 20
history = model.fit(training_padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

# Plotando o gráfico de precisão de aprendizado
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and validation accuracy')
plt.savefig('training_acc.jpeg')
plt.show()


# Testando uma review aleatoria
while True:
    review = str(input("Write a ramdom review:\n"))
    txt = [review]
    txt = tokenizer.texts_to_sequences(txt)
    txt = pad_sequences(txt, maxlen=max_length)
    sentiment = model.predict(txt,batch_size=1,verbose = 2)[0]
    print(f"This review has a {sentiment.max()} chance of being good.")