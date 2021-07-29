# unmask, based on HugginfFaceHub pipelines

```shell 
  pip install -r requirements.txt

  # unmask: replace each input word with [MASK], and output k best fillers with scores
  echo 'This is a test of the emergency broadcast system .' | python unmask.py

  # output
  # this	this|0.595, it|0.375, there|0.004, that|0.002, here|0.001
  # is	was|0.628, is|0.334, included|0.007, includes|0.004, involved|0.003
  # a	a|0.935, another|0.029, the|0.023, one|0.005, first|0.000
  # test	part|0.253, subset|0.125, variation|0.082, feature|0.072, component|0.051
  # of	of|0.886, for|0.070, on|0.017, in|0.010, against|0.004
  # the	the|0.883, an|0.088, our|0.005, your|0.003, their|0.002
  # emergency	radio|0.257, television|0.048, satellite|0.031, digital|0.031, microwave|0.029
  # broadcast	braking|0.620, response|0.070, management|0.024, medical|0.016, lighting|0.014
  # system	system|0.244, technique|0.077, capability|0.059, technology|0.033, method|0.028
  # .	.|0.969, ;|0.029, !|0.001, ?|0.000, ||0.000
```

<h1>Calibration</h1>

Can we interpret these scores as probabilities?

The plots <a href="https://github.com/kwchurch/deepnet_examples/blob/main/pretrained/examples/HuggingFaceHub/Bertology/unmask/calibration.pdf">here</a>
and <a href="https://github.com/kwchurch/deepnet_examples/blob/main/pretrained/examples/HuggingFaceHub/Bertology/unmask/calibration2.pdf">here</a>
show that the scores are overconfident (because the black circles are
below the dashed red line).  <p> The x-axis shows scores and the
y-axis shows estimates of Pr(correct).  That is, one might hope that
the scores could be interpreted as probabilities, but if that was
correct, the points would fall on the dashed red line (the y=x line).
We estimate Pr(correct) by binning a number of output rows from unmask.py.
Each row is counted as correct iff the word in the first column matches
the word in the second column.  Correct is a boolean value.  Pr(correct)
is the average of correct within a bin.  Rows are binned by score (101 bins from 0 to to 1 by 0.01).

<p>

We
believe that the points are below the dashed red line because scores
are computed using a softmax that compared the candidate to the sum of
all candidates that were considered.  The ratio is overconfident
because the denominator leaves out the mass for candidates that were
not considered.  In general, there will be many such words, especially
for content words.  Their probabilities are small, but if there are
many of them, their aggregate mass can be too large to ignore.

<p>

How did we compute these calibration plots?
These calibration plots are based on the wikitext test set (wikitext-103-raw-v1.test), also in this directory.
This file created by <a href="https://github.com/kwchurch/deepnet_examples/tree/main/datasets/HuggingFace">dataset_cat.py</a>.
<p>
We used unmask.py to produce wikitext-103-raw-v1.test.unmasked

```shell 
  # Warning, this will take a long time; we ran this in parallel on a cluster
  python unmask.py < wikitext-103-raw-v1.test > wikitext-103-raw-v1.test.unmasked
  ```

We also created hist.txt, another file in this directory,
by counting unigram frequencies from the training set (though we did not post the training set).

```shell
   cut -f2- -d'|' < wikitext-103-raw-v1.train | 
   awk '{for(i=1;i<=NF;i++) x[tolower($i)]++}; 
   END {for(i in x) print x[i] "\t" i}' > hist.txt

We combined hist.txt and wikitext-103-raw-v1.test.unmasked to create
wikitext-103-raw-v1.test.unmasked.forR.

```shell
   tr '|' '\t' < wikitext-103-raw-v1.test.unmasked |
   awk -F'\t' 'BEGIN {print "correct\tscore\tfreq"; while(getline < "hist.txt" > 0) freq[$2] += $1};
   $3 ~ /[0-9]/ {printf "%d\t%s\t%d\n", tolower($1) == tolower($2), $3, freq[tolower($1)]}' > wikitext-103-raw-v1.test.unmasked.forR
```

The file, R_notes.txt, shows how we generated the plots in this
directory from wikitext-103-raw-v1.test.unmasked.forR.  The blue stars
are computed using logistic regression, an alternative calibration
method to binning.  The regression makes use of two input variables:
(1) score as (2) unigram frequencies.  There is a well-known
freequency effect in psycholinguistics, where humans tend to perform a
number of tasks faster (and more accurately) for more frequent words.
The regresssions show that Pr(correct) not only depends on score but
also unigram frequency, suggesting there may be an opportunity to improve
transformer models by taking advantage of frequency.






  
  



