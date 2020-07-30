import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import os

from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM

ZEMBEREK_PATH: str = os.path.join('bin', 'zemberek-full.jar')

startJVM(
    getDefaultJVMPath(),
    '-ea',
    f'-Djava.class.path={ZEMBEREK_PATH}',
    convertStrings=False
)

TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
TurkishSentenceNormalizer: JClass = JClass('zemberek.normalization.TurkishSentenceNormalizer')
Paths: JClass = JClass('java.nio.file.Paths')
normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(
        os.path.join('data', 'normalization')
    ),
    Paths.get(
        os.path.join('data', 'lm', 'lm.2gram.slm')
    )
)

TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
Token: JClass = JClass('zemberek.tokenization.Token')
tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT

WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')
morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

words = []
words_temp = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = normalizer.normalize(JString(pattern))
        w = []
        token_iterator = tokenizer.getTokenIterator(JString(pattern))
        for token in token_iterator:
            w.append(token.content)
        words_temp.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


for w in words_temp:
    if w not in ignore_words:
        results: WordAnalysis = morphology.analyze(JString(w))
        for result in results:
            for r in result.getLemmas():
                words.append(str(r))

words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words_temp = doc[0]
    pattern_words = []

    for w in pattern_words_temp:
        results: WordAnalysis = morphology.analyze(JString(w))
        for result in results:
            for r in result.getLemmas():
                pattern_words.append(str(r))
        
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model created")

shutdownJVM()
