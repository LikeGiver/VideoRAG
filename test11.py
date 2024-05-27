import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
hypothesis=set(hypothesis)
reference=set(reference)
#there may be several references
merteor_score = nltk.translate.meteor_score.single_meteor_score(reference, hypothesis)
print(merteor_score)