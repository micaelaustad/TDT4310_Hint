import nltk
your_grammar = nltk.CFG.fromstring("""
S -> V NP
V -> 'describe' | 'present'
NP -> PRP N 
PRP -> 'your' 
N -> 'work'
""")

parser = nltk.ChartParser(your_grammar)
sent = 'describe your work'.split()

print(list(parser.parse(sent)))
