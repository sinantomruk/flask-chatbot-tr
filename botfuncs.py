import os
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

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

def clean_up_sentence(sentence):
    sentence_words = normalizer.normalize(JString(sentence))
    token_iterator = tokenizer.getTokenIterator(JString(sentence_words))
    words_temp = []
    for token in token_iterator:
        words_temp.append(str(token.content))
    
    sentence_words = []
    for w in words_temp:
        results: WordAnalysis = morphology.analyze(JString(w))
        for result in results:
            for r in result.getLemmas():
                sentence_words.append(str(r))

    sentence_words = list(set(sentence_words))

    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return (np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:
        res = getResponse(ints, intents)
    else:
        res = "Özür dilerim ne demek istediğini tam olarak anlayamadım. Lütfen tekrar deneyin."
    return res


#shutdownJVM()
