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
from collections import namedtuple, defaultdict, Counter

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import linear_kernel
from sklearn import metrics

# ----------------------------------------------------------------------------

result_green  = "\U0001F7E9"
result_yellow = "\U0001F7E8"
result_white  = "\u2B1C"
result_black  = "\u25A0"

# Prints squares for guesses like the real thing
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

# Prints the squares, then next to it the letters
# Letters wrapped in unicode combining boxes to look nicer
def calc_with_squares(actual, guess):
   return calculate_squares(actual, guess) + "  " + \
          " ".join( [ "%s\u20e3" % x for x in guess ] )

# Let's see it in action!
print(calc_with_squares("bbuzz","guess"))
print(calc_with_squares("bbuzz","uzbek"))
print(calc_with_squares("bbuzz","soyuz"))
print(calc_with_squares("bbuzz","bbuzz"))

# ----------------------------------------------------------------------------

# Read in the 5 letter words
language = "british-english"
words = pd.read_csv("wordle/%s"%language, header=0, names=["word"])
print("Loaded %d words of %s" % (len(words), language))

# ----------------------------------------------------------------------------

# Have a look at our first few words
print(words.head(5))

# ----------------------------------------------------------------------------

# Identify the most common letters
# Should be similar to the distribution for the language, but may be slightly
#  off due to the shorter words only that we're working with
as_letters = words["word"].str.split('',n=5,expand=True).drop(0, axis=1)
letter_counts = Counter(as_letters.values.flatten())
print(letter_counts.most_common(10))

# ----------------------------------------------------------------------------

# What is a good starting word?
# We want ones with as many popular letters as possible

# If we include repeats?
# TODO

# Or if we want as many different popular letters as possible?
# TODO

# ----------------------------------------------------------------------------

# Build a "scorer" for a given Word
# Will return 6 scores for a given Word - Overall + Per-Letter
# - Per-letter, give 1.0 if correct, 0.5 if wrong place, 0.0 if not used
# - Overall, similar but with 0.8 for wrong place, and 0.2 if not used
def score(answer, guess):
   # TODO
   return [0]

# ----------------------------------------------------------------------------

# TODO Something on TF-IDF
# TODO Something on TF-DF which is more what we need

# ----------------------------------------------------------------------------

# Do we even need AI / ML?
# Try with just....

# ----------------------------------------------------------------------------
