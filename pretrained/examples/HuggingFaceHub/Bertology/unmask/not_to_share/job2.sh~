#!/bin/sh

cat wikitext-103-raw-v1.test | 
cut -f2- -d'|' | 
awk 'NF > 3' | 
python unmask.py > wikitext-103-raw-v1.test.unmasked
