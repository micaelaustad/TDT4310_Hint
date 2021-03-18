from sklearn.model_selection import train_test_split
from nltk.corpus import brown, nps_chat
import nltk
from collections import Counter
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, RegexpTagger, TrigramTagger


brown_test_50, brown_train_50 = train_test_split(
    brown.tagged_sents(), test_size=0.5)
nps_test_50, nps_train_50 = train_test_split(
    nps_chat.tagged_posts(), test_size=0.5)
brown_90, brown_10 = train_test_split(
    brown.tagged_sents(),  test_size=0.1)
nps_90, nps_10 = train_test_split(
    nps_chat.tagged_posts(), test_size=0.1)

most_common_nps = Counter(
    [tag for word, tag in nps_chat.tagged_words()]).most_common(1)[0][0]
most_common_brown = Counter(
    [tag for word, tag in brown.tagged_words()]).most_common(1)[0][0]

nps_tagger = DefaultTagger(most_common_nps)
brown_tagger = DefaultTagger(most_common_brown)

br_50_acc = brown_tagger.evaluate(brown_test_50)
br_10_acc = brown_tagger.evaluate(brown_10)
nps_50_acc = nps_tagger.evaluate(nps_test_50)
nps_10_acc = nps_tagger.evaluate(nps_10)

print(f"Brown 50 : {br_50_acc}")
print(f"Brown 10 : {br_10_acc}")
print(f"Nps 50 : {nps_50_acc}")
print(f"Nps 10 : {nps_10_acc}")

# The comment is that the default tagger is in general not a good option, however what this exercise does not give any information about is if the Default tagger performs well as a back-off.
# Say that there are only 20% of tags that are not handled by Bigram or Unigram, how many of the labels that are left will be accuractly tagged with a default tagger?
# That percentage might be considerably higher or lower than 13% percent.


# B

patterns = [
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # simple past
    (r'.*es$', 'VBZ'),                 # 3rd singular present
    (r'.*ould$', 'MD'),                # modals
    (r'.*\'s$', 'NN$'),                # possessive nouns
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    # I removed the catch all ;)
]


def train_n_print(dataset, org_backoff):
    regex = RegexpTagger(patterns, backoff=org_backoff)
    uni = UnigramTagger(train=dataset[0], backoff=org_backoff)
    bi = BigramTagger(train=dataset[0], backoff=uni)
    tri = TrigramTagger(train=dataset[0], backoff=uni)

    default_acc = org_backoff.evaluate(dataset[1])
    regex_acc = regex.evaluate(dataset[1])
    uni_acc = uni.evaluate(dataset[1])
    bi_acc = bi.evaluate(dataset[1])
    tri_acc = tri.evaluate(dataset[1])

    print("Accuracy \n==============================")
    print("Default Tagger/Lookup:", default_acc)
    print("Regex Tagger: (backoff=Default)", regex_acc)
    print("Unigram (backoff=Regex):", uni_acc)
    print("Bigram Tagger (backoff=Unigram):", bi_acc)
    print("Trigram Tagger (backoff=Bigram):", tri_acc)


print("\n\nBrown 50/50")
train_n_print([brown_90, brown_10], brown_tagger)
print("\n\nBrown 90/10")
train_n_print([brown_train_50, brown_test_50], brown_tagger)
print("\n\nNPS 50/50")
train_n_print([nps_train_50, nps_test_50], nps_tagger)
print("\n\nNps 90/10")
train_n_print([nps_90, nps_10], nps_tagger)

# 3


def lookup_tagger(non_tagged, tagged, amount):
    fd = nltk.FreqDist(non_tagged())
    cfd = nltk.ConditionalFreqDist(tagged())
    most_freq_words = fd.most_common(amount)
    likely_tags = dict((word, cfd[word].max())
                       for (word, _) in most_freq_words)
    baseline_tagger = nltk.UnigramTagger(model=likely_tags)
    return baseline_tagger


brown_lookup = lookup_tagger(brown.words, brown.tagged_words, 200)
print("\n\nBrown 50/50")
train_n_print([brown_90, brown_10], brown_lookup)
print("\n\nBrown 90/10")
train_n_print([brown_train_50, brown_test_50], brown_lookup)
print("\n\nNPS 50/50")
train_n_print([nps_train_50, nps_test_50], brown_lookup)
print("\n\nNps 90/10")
train_n_print([nps_90, nps_10], brown_lookup)

# 3b)
reductions = [5000, 1000, 500, 100]
for reduction in reductions:
    brown_test_50, brown_train_50 = train_test_split(
        brown.tagged_sents()[:reduction], test_size=0.5)
    nps_test_50, nps_train_50 = train_test_split(
        nps_chat.tagged_posts()[:reduction], test_size=0.5)
    brown_90, brown_10 = train_test_split(
        brown.tagged_sents()[:reduction],  test_size=0.1)
    nps_90, nps_10 = train_test_split(
        nps_chat.tagged_posts()[:reduction], test_size=0.1)
    brown_lookup = lookup_tagger(brown.words, brown.tagged_words, 200)
    print(f"\n\n Reduced to {reduction} Brown 50/50")
    train_n_print([brown_90, brown_10], brown_lookup)
    train_n_print([brown_90, brown_10], brown_tagger)
    print(f"\n\n Reduced to {reduction} Brown 90/10")
    train_n_print([brown_train_50, brown_test_50], brown_lookup)
    train_n_print([brown_train_50, brown_test_50], brown_tagger)
    print(f"\n\n Reduced to {reduction} NPS 50/50")
    train_n_print([nps_train_50, nps_test_50], brown_lookup)
    train_n_print([nps_train_50, nps_test_50], nps_tagger)
    print(f"\n\n Reduced to {reduction} Nps 90/10")
    train_n_print([nps_90, nps_10], brown_lookup)
    train_n_print([nps_90, nps_10], nps_tagger)


# The general notion here is that the smaller you get the dataset the more inconsistent the results become.
# In some cases, unintuitivly the 50/50 split will perform better than the 90/10 split.
# For a larger dataset one could say that it means that the amount of data has gotten to the point where it "saturates" the needs of the algorithm to get to optimal performance, so 90% vs 50% for training is insignificate.
# In this case however with such small dataset there are several possible reasons. Overfitting due to small datasets, too little test data to get accurate feedback from the algorithm, outlier, inate errors in the data, and possibly other issues.
# What is the spesific cause of the unexpected results in each case of training an algoritm is hard to pin down. Doing data exploration in advance (cleaning, removing outlier, missing values, errors, noise etc), using simpler models and regulerazing the data are some of the most common ways to handle these issues.
# An intersting note is that for the NPS dataset we see the expected results for all sizes, 50% performing worse that 90%, however for the Brown 50% consistenly beats 90%. This does not hold true down to the 100 values range, but it suggests that there is something about the brown datasets that is special.
