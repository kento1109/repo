# maximum entropy model
import numpy as np

class MEM(object):
    def __init__(self, eta, c, stop_num):
        self.eta = eta
        self.c = c
        self.stop_num = stop_num
        self.L = -999

    def updateW(self):
        sum_f = np.zeros(self.fm.shape[2])
        for i, (sent, label) in enumerate(zip(self.docs, self.labels)):
            sum_f = sum_f + self.fm[i][label] - np.dot(self.prob[i], self.fm[i])
        return sum_f - self.c * np.sum(self.w)

    def negative_log_likelihood(self):
        return np.sum(np.log(self.prob[np.arange(len(self.labels)), self.labels])) \
               - (self.c / 2) * np.linalg.norm(self.w)

    def fit(self, docs, labels):
        self.docs = docs
        self.labels = labels
        self.classes = set(labels)
        # build feature
        self.features = [f for f in self.build_feature()]
        # docs to feature vector
        self.fm = np.array([fm for fm in self.doc2fv(docs)])
        self.w = np.zeros(self.fm.shape[2])
        dif_ = 1
        while self.stop_num < dif_:
            self.prob = self.calc_prob(self.fm)
            w_ = self.updateW()
            self.w = self.w + self.eta * w_
            current_L = self.negative_log_likelihood()
            dif_ = current_L - self.L
            self.L = current_L

    def predict(self, docs):
        # docs to feature vector
        fm = np.array([fm for fm in self.doc2fv(docs)])
        prob = self.calc_prob(fm)
        return np.argmax(prob, axis=0)

    def calc_prob(self, fm):
        dot_ = np.dot(fm, self.w)
        return self.softmax(dot_)

    def softmax(self, f):
        e = np.exp(f)
        return e / np.sum(e, axis=1, keepdims=True)

    def build_feature(self):
        words = set([word for sent in self.docs for word in sent.split()])
        added = set()
        for word in words:
            for label in self.labels:
                if word + str(label) in added:
                    pass
                else:
                    added.add(word + str(label))
                    yield [word, label]

    def doc2fv(self, docs):
        for i, doc in enumerate(docs):
            label_list = []
            for label in self.classes:
                feature_list = []
                for feature in self.features:
                    if feature[0] in doc.split() and feature[1] == label:
                        feature_list.append(1)
                    else:
                        feature_list.append(0)
                label_list.append(feature_list)
            yield label_list


docs = ['good bad good good',
        'exciting exciting',
        'bad boring boring boring',
        'bad exciting bad']
labels = [0, 0, 1, 1]

mem = MEM(eta=0.1, c=0.1, stop_num=1.0e-5)
mem.fit(docs, labels)
print mem.predict(['bad exciting good', 'bad boring'])
