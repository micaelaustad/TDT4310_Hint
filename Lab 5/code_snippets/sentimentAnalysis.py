from collections import Counter
import wikipedia
from textblob import TextBlob
ma = wikipedia.WikipediaPage(title='Marcus Aurelius').summary


def analize_sentiment(sentence):
    analysis = TextBlob(sentence)
    if analysis.sentiment.polarity > 0:
        return 'pos'
    elif analysis.sentiment.polarity == 0:
        return 'neu'
    else:
        return 'neg'

for sentence in ma:
    print(analize_sentiment(sentence))
