
import gensim
import os
import collections
import smart_open
import random
import gensim.utils
import gensim.models.doc2vec

import jieba

def seg_text(txt):
    pass



def read_corpus(fname):

    idx = -1
    with open(fname) as f:
        for i, line in enumerate(f):
            line_parts = gensim.utils.simple_preprocess(line)
            if len(line_parts) == 0:
                continue
            parts = []
            for line_part in line_parts:
                parts.extend(list(jieba.cut(line_part)))
            idx += 1


            yield gensim.models.doc2vec.TaggedDocument(parts, [idx])


train_corpus = list(read_corpus('input/questions.csv'))



model = gensim.models.doc2vec.Doc2Vec(size=600, min_count=1, iter=55)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

ranks = []

second_ranks = []

correct = 0
wrong = 0

for words, doc_id in train_corpus:
    doc_id = doc_id[0]
    # try:
    inferred_vector = model.infer_vector(words)
    # except Exception as e:
        # print('check')
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    # print('most similar index', sims[0][0])
    print('pairs')
    print(words)
    print('most similar docs', train_corpus[sims[0][0]])

    if doc_id == sims[0][0]:
        correct += 1
    else:
        wrong += 1

print('precision is ', correct / (correct + wrong))


