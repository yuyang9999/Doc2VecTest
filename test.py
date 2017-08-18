import sys
import numpy as np
import gensim

from gensim.models.doc2vec import Doc2Vec, LabeledSentence, TaggedDocument
from sklearn.cross_validation import train_test_split



def get_dataset(pos_file, neg_file, unsup_file):
    with open(pos_file, 'r') as infile:
        pos_reviews = infile.readlines()
    with open(neg_file, 'r') as infile:
        neg_reviews = infile.readlines()
    with open(unsup_file, 'r') as infile:
        unsup_reviews = infile.readlines()

    #使用1表示正面情感，0为负面
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    #将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    #对英文做简单的数据清洗预处理，中文根据需要进行修改
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n', '') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s ' % c) for z in corpus]
        corpus = [z.split() for z in corpus]

        return np.array(corpus)

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)

    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s' % (label_type, i)
            labelized.append(TaggedDocument(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    return x_train, x_test, unsup_reviews, y_train, y_test

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape(1, size) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, x_test, unsup_reviews, size=400, epoch_num=10):
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    # model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    #使用所有的数据建立字典

    all_docs = []
    all_docs.extend(x_train)
    all_docs.extend(x_test)
    all_docs.extend((unsup_reviews))

    # all_docs = np.concatenate((x_train, x_test, unsup_reviews))


    model_dm.build_vocab(all_docs)
    # model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

    all_train_reviews = []
    all_train_reviews.extend(x_train)
    all_train_reviews.extend(unsup_reviews)
    # all_train_reviews = np.concatenate((x_train, unsup_reviews))

    for epoch in range(epoch_num):
        print('train ', epoch)
        perm = np.random.permutation(len(all_train_reviews))
        perm_train_reviews = [all_train_reviews[idx] for idx, line in enumerate(perm)]
        model_dm.train(perm_train_reviews, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        # model_dbow.train(all_train_reviews[perm])

    #训练测试数据
    # x_test = np.array(x_test)
    for epoch in range(epoch_num):
        print('train ', epoch)
        perm = np.random.permutation(len(x_test))
        perm_test_reviews = [x_test[idx] for idx, line in enumerate(perm)]
        model_dm.train(x_test[perm], total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        # model_dbow.train(x_test[perm])

    return model_dm

def get_vectors(model_dm, model_dbow, x_train, x_test, size):
    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test,size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs, test_vecs_dbow

def Classifier(train_vecs, y_train, test_vecs, y_test):
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))

    return lr

def ROC_curve(lr, y_test, test_vecs):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:,1]

    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lable='area = %.2f' % roc_auc)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()


if __name__ == '__main__':
    size, epoch_num = 400, 1
    x_train, x_test, unsup_reviews, y_train, y_test = get_dataset('temp/pos.txt', 'temp/neg.txt', 'temp/unsup.txt')
    model_dm = train(x_train, x_test, unsup_reviews, size, epoch_num)
    model_dm.save('output/dm.model')
    # train_vecs, test_vecs = get_vectors(model_dm, model_dbow)
    # lr = Classifier(train_vecs, y_train, test_vecs, y_test)
    # ROC_curve(lr, y_test)


