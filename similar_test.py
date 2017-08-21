
import gensim
import os
import collections
import smart_open
import random
import gensim.utils
import gensim.models.doc2vec
import jieba
import sys


def read_corpus(fname, start_idx = -1):

    idx = start_idx
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


train_corpus_1 = list(read_corpus('input/questions.csv'))
train_corpus_2 = list(read_corpus('input/sougou.csv', start_idx=len(train_corpus_1)))

total_corpus = []
total_corpus.extend(train_corpus_1)
total_corpus.extend(train_corpus_2)
model = gensim.models.doc2vec.Doc2Vec(size=600, min_count=1, iter=55)
model.build_vocab(total_corpus)

model.train(total_corpus, total_examples=model.corpus_count, epochs=model.iter)

ranks = []

second_ranks = []

correct = 0
wrong = 0

#build the doc dictionary
doc_dic = {}
for words, doc_ids in train_corpus_1:
    doc_dic[doc_ids[0]] = words

for words, doc_ids in train_corpus_2:
    doc_dic[doc_ids[0]] = words

for words, doc_id in train_corpus_1:
    doc_id = doc_id[0]
    # try:
    inferred_vector = model.infer_vector(words)
    # except Exception as e:
        # print('check')
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    # print('most similar index', sims[0][0])
    print('pairs')
    print(words)

    print('most similar docs', doc_dic[sims[0][0]])
    print('second similar docs', doc_dic[sims[1][0]])

    if doc_id == sims[0][0]:
        correct += 1
    else:
        wrong += 1

print('precision is ', correct / (correct + wrong))


