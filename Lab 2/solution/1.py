from nltk.corpus import brown, nps_chat
import nltk
from collections import Counter, defaultdict

sents = list(brown.tagged_sents())


def a(sents):
    tags = []
    for sent in sents:
        tags.extend([tag for word, tag in sent])
    return Counter(tags).most_common(1)


def b(sents):
    words = defaultdict(set)
    for sent in sents:
        for word, tag in sent:
            words[word].add(tag)

    return len([word for word in words if len(words[word]) > 1])


def c(sents):
    words_len = len(set(brown.words()))
    amount_of_amb = b(sents)
    return f"The percentage of ambigious words is {amount_of_amb/words_len*100}%"


def d(sents):
    words = defaultdict(list)
    for sent in sents:
        for word, tag in sent:
            words[word].append(tag)
    counter = Counter({key: len(lst) for key, lst in words.items()})
    counter = sorted(counter, key=lambda pair: pair[1], reverse=True)
    most_common_words = [entry for entry in counter.most_common(10)]

    for word in most_common_words:
        for sent in brown.sents():
            if word in sent:
                print(sent)
                break


print(sents[0])

print(a(sents))
print(b(sents))
print(c(sents))
print(d(sents))
