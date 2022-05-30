#!/bin/bash
# Builds up wordle-like word lists from the system dictionaries
# To make it more worde-like, it:
#  * Lowercases everything
#  * Removes accents
#  * Skips words with punctuation
# Unlike wordle, does include plurals, past tenses etc

DICTDIR="/usr/share/dict/"
DICTS="british-english french spanish"

for LANGUAGE in $DICTS; do 
  DICT=${DICTDIR}${LANGUAGE}
  cat $DICT | \
    iconv -f utf8 -t ascii//TRANSLIT - | \
    tr '[:upper:]' '[:lower:]' | \
    grep '^[a-z][a-z][a-z][a-z][a-z]$' > wordle/$LANGUAGE
done
