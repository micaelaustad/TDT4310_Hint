import nltk


tagged_sents = nltk.corpus.brown.tagged_sents()


def chunk(sentence):
    grammar = r"""
NP: {<DT>? <JJ>* <NN>*} # NP
P: {<IN>}           # Preposition
V: {<V.*>}          # Verb
PP: {<P> <NP>}      # PP -> P NP
CLAUSE: {<V> <NP>}
"""
    parser = nltk.RegexpParser(grammar)
    result = parser.parse(sentence)
    return result


tuples = set([])
for sent in tagged_sents:
    # Get the tree
    tree = chunk(sent)
    # Look for subtrees
    for subtree in tree.subtrees():
        # If a subtree is the CLASUE we defined as VB -> NP,
        if subtree.label() == 'CLAUSE':
            # Add the verb and the first word of the NP that triggered the match
            tuples.add((subtree[0][0][0], subtree[1][0][0]))
[print(tuplx) for tuplx in tuples]

"""
('Draw', 'each')
('make', 'available')
('find', 'meaning')
('buy', 'copybooks')
('make', 'good')
('increase', 'CDC')
('lessen', 'this')
('speed', 'diagnosis')
('avoid', 'suspicion')
('see', 'objects')
('make', 'definite')
('buy', 'meat')
('eat', 'Western')
('limit', 'delays')
('use', 'wood')
('bring', 'long')
('sell', 'dozens')
('put', 'life')
"""
