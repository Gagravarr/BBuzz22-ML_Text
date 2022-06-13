#!/usr/bin/python3

#   Licensed to the Apache Software Foundation (ASF) under one
#   or more contributor license agreements.  See the NOTICE file
#   distributed with this work for additional information
#   regarding copyright ownership.  The ASF licenses this file
#   to you under the Apache License, Version 2.0 (the
#   "License"); you may not use this file except in compliance
#   with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing,
#   software distributed under the License is distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied.  See the License for the
#   specific language governing permissions and limitations
#   under the License.


# ----------------------------------------------------------------------------

# Setup step - load all our libraries
# These are chosen for speed of development and understanding, not performance!

import json
import pickle
import csv
import sys

from collections import namedtuple, defaultdict, Counter, OrderedDict
from random import randrange

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------

# Try to import things for Notebook display/rendering
try:
    from IPython.display import display, HTML
    notebook = 'ipykernel' in sys.modules
except ImportError:
    notebook = False

# Pretty-print our DataFrames
def render(df):
    if notebook:
        display(HTML(df.to_html()))
    else:
        print("")
        print(df)

# ----------------------------------------------------------------------------

result_green  = "\U0001F7E9"
result_yellow = "\U0001F7E8"
result_white  = "\u2B1C"
result_black  = "\u25A0"

# Generates squares for guesses like the real thing
def calculate_squares(actual, guess):
   res = []
   for i in range (0,5):
      if guess[i] == actual[i]:
         res += result_green
      elif guess[i] in actual:
         res += result_yellow
      else:
         res += result_white
   return "".join(res)

# Generates the squares, then next to it the letters
# Letters wrapped in unicode combining boxes to look nicer
def calc_with_squares(actual, guess):
   return calculate_squares(actual, guess) + "  " + \
          " ".join( [ "%s\u20e3" % x for x in guess ] )

# Return if you have one, then the squares
def calc_squares_win(actual, guess):
   squares = calculate_squares(actual, guess)
   return [actual==guess, squares]

# Let's see it in action!
print("")
print(calc_with_squares("bbuzz","guess"))
print(calc_with_squares("bbuzz","uzbek"))
print(calc_with_squares("bbuzz","soyuz"))
print(calc_with_squares("bbuzz","bbuzz"))

# ----------------------------------------------------------------------------

# Removes any duplicate letters, may also scramble order
remove_duplicate_letters = lambda word: "".join(set(word))

# ----------------------------------------------------------------------------


# Read in the 5 letter words, in English and French
def load_words(language):
   words = pd.read_csv("wordle/%s"%language, header=0, names=["word"])
   print("")
   print("Loaded %d words of %s" % (len(words), language))
   return words

words    = load_words("british-english")
words_fr = load_words("french")

# ----------------------------------------------------------------------------

# Have a look at our first few words
render(words.head(5))
render(words_fr.head(5))

# ----------------------------------------------------------------------------

# Identify the most common letters
# Should be similar to the distribution for the language, but may be slightly
#  off due to the shorter words only that we're working with
as_letters = words["word"].str.split('',n=5,expand=True).drop(0, axis=1)
render(as_letters.head(5))

letter_counts = Counter(as_letters.values.flatten())

print("")
print(letter_counts.most_common(10))

max_letter_count = letter_counts.most_common(1)[0][1]
total_letters_count = sum( [letter_counts[x] for x in letter_counts] )
print("Maximum letter count was %d, from %d" % (max_letter_count, total_letters_count))

# ----------------------------------------------------------------------------

# Graph the letter distributions

# Order the bars so the highest frequency ones go on the left
ordered_letters = OrderedDict(letter_counts.most_common())
# Pick a different colour for each letter, from a rainbow-like range
letter_colors = plt.cm.get_cmap("rainbow")(np.linspace(0,1,len(letter_counts)))
# Plot as a Bar Chart
plt.bar(ordered_letters.keys(), ordered_letters.values(), color=letter_colors)
plt.title("Letter Frequencies - 5 Letter Words - English")
# Save as a PNG plus display
plt.savefig("letter-freqs.png", dpi=150)
plt.show()
print(ordered_letters)

# ----------------------------------------------------------------------------


# What is a good starting word?
# We want ones with as many popular letters as possible

# What ratio of the most popular letter does the word use?
# Optionally score words with duplicate letters as zero
def score_by_letter_counts(wordrow, skip_duplicates=False):
   word = wordrow["word"]
   if skip_duplicates and len(remove_duplicate_letters(word)) != 5:
      return 0
   return np.product( 
              [letter_counts[l]*1.0/max_letter_count for l in word] )

# Calculate the scores, with and without repeats
words_letterscore = words.copy(deep=True)
words_letterscore["score_all"] = words_letterscore.apply(
                           lambda x: score_by_letter_counts(x,False), axis=1)
words_letterscore["score_nodup"] = words_letterscore.apply(
                           lambda x: score_by_letter_counts(x,True), axis=1)

# Look at the first few
render(words_letterscore.head(5))

# What are the best, the two different ways?
render(words_letterscore.sort_values("score_all",ascending=False).head(5))
render(words_letterscore.sort_values("score_nodup",ascending=False).head(5))

# How do a few words compare?
render(words_letterscore.query("word == 'soyuz'"))
render(words_letterscore.query("word == 'audio'"))

# ----------------------------------------------------------------------------


# Build a per-character TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0, stop_words=None,
                        analyzer="char", ngram_range=(1,1))
tfidf_matrix = tfidf.fit_transform(words["word"])

print("")
print("TF-IDF build, shape is:")
print(tfidf_matrix.shape)
print("Features (letters) are:")
print(tfidf.get_feature_names())
print("")

# See how a few words map with the TF-IDF
for i in range(10):
   idx = randrange(len(words))
   print(words["word"][idx])
   print(tfidf_matrix[idx])

# ----------------------------------------------------------------------------

# Calculate an average TF-IDF score for each word
words_letterscore["score_tfidf"] = pd.Series([
   tfidf_matrix[idx].sum()/5 for idx in range(len(words)) ])

render(words_letterscore.head(5))

# What are the best words based on the TF-IDF?
render(words_letterscore.sort_values("score_tfidf",ascending=False).head(5))

# TF-IDF isn't great....
# We want the most common letters to start with, not the least!
# More like TF-DF not TF-IDF

# ----------------------------------------------------------------------------

# Build a "scorer" for a given Word
# Will return 6 scores for a given Word - Overall + Per-Letter
# The "hyper-parameters" of the weighting for "right letter wrong place"
#  can be tuned to give you control over the scoring
def score(actual, guess, weight_yellow_single=0.5, weight_yellow_overall=0.8,
                         weight_white_single=0.0, weight_white_overall=0.2):
   res = [1.0]*6
   for i in range (0,5):
      if guess[i] == actual[i]:
         # Everything already set
         pass
      elif guess[i] in actual:
         res[0] = res[0] * weight_yellow_overall
         res[i+1] = weight_yellow_single
      else:
         res[0] = res[0] * weight_white_overall
         res[i+1] = weight_white_single
   return res

print("")
print(score("bbuzz","uzbek"))
print(score("bbuzz","soyuz"))

# ----------------------------------------------------------------------------

# Do we even need AI / ML?
# No... but stats or regexp approaches are out of scope!

# ----------------------------------------------------------------------------

# Multinomial Naive Bayes
# Classier which can predict from learning what's similar

# Train one on the TF-IDF scores of the words
classifier = MultinomialNB()
model = make_pipeline(tfidf, classifier)
learn_text = words["word"]
model.fit( list(learn_text), list(learn_text.index) )

# Ask it for similar words
def recommend(query, model):
   # Ask for a prediction, predicts the index of a word
   # Note there's no way to tell it not to keep recommending the same one...
   predictions = model.predict([query])
   if len(predictions) > 0:
      # Find the word for that
      pred_idx = predictions[0]
      word = learn_text[pred_idx]
      return word
   return None
def print_recommend(query, model):
   guess = recommend(query, model)
   print("Model guessed from %s - %s" % (query, guess))

print("")
print_recommend("arise", model)
print_recommend("raise", model)
print_recommend("..ise", model)
print_recommend("r..se", model)

# ----------------------------------------------------------------------------

# Ask our MNB to play Wordle!
# Our TF-IDF has lost the information on letter position, it is
#  a "bag of letters"
# That means we can consider greens and yellows together, not great...
# Any letter not in our A-Z is ignored, so use . for letters we don't know

def recommend_play(model):
   # Pick a random word to guess
   picked_idx = randrange(len(words))
   actual = words["word"][picked_idx]
   # We don't know any words, model will pick something
   guess = "....."

   for guessnum in range(6):
      # Ask the model to recommend a word based on what we have
      guess = recommend(guess, model)
      # How was it?
      won, squares = calc_squares_win(actual, guess)
      print(squares + "   " + guess)
      if won:
         break
      else:
         # Keep greens and yellows
         guess = "".join([x if x in actual else '.' for x in guess])
   print("The model didn't get it right! Word was %s" % actual)

print("")
recommend_play(model)

print("")
recommend_play(model)

# ----------------------------------------------------------------------------

# Unfortunately, this is pretty bad... 

# Clustering? Classifiers?
# No... We want to get away from eg break / creak / freak / wreak
# No... Double letters don't give us as much information

# ----------------------------------------------------------------------------

# How about you?

# Play the game, using the earlier squares code
picked_idx = randrange(len(words))
picked = words["word"][picked_idx]
solved = False
print("\nLet's play Wordle! Game number %d" % picked_idx)

for i in range(6):
   guess = input("What is your guess #%d? " % (i+1))
   won, squares = calc_squares_win(picked, guess)
   print(squares)
   if won:
      print("Well done! You solved game %d" % picked_idx)
      solved = True
      break
if not solved:
   print("Sorry, you failed to guess the word in 6 tries, it was %s" % picked)
