# Deep Nets for Poets

Kenneth Church
<p>

Please feel free to send me email: <a
href="mailto:kenneth.ward.church@gmail.com?subject="Deep Nets for
Poets">here</a>.  I would appreciate feedback on the examples posted
here, as well as contributions with more examples and more solutions
that would help users appreciate why they should be as excited about
deep nets as we are.

<p>
Many years ago, well before the web, I gave some lectures at a
Lingusitics Summer School on

<a
href="https://web.stanford.edu/class/cs124/kwc-unix-for-poets.pdf">Unix
for Poets</a>.  At the time, we were really excited by what we could
do with lexical resources and corpora (superhighway roadkill such as
email, bboards, faxes).

<ul>
<li> Question: What can we do with it all?</li>
<li> Answer: It is better to do something simple than nothing at all</li>
</ul>

These lectures argued that poets (users such as linguists,
lexicographers, translators) can do simple things themselves.
Moreover, DIY (do it yourself) is more satisfying than begging for
"help" from a system admin.

<p>

In ``Deep Nets for Poets,'' I want to make a similar argument,
but for deep nets (as opposed to Unix).  In particular, there are simple
examples in subdirectories below this that show

<ol>
<li>Image Classification: Pictures &#8594; Labels </li>
<li>OCR (Optical Character Recognition): Pictures &#8594; text (English and Chinese)</li>
<li>Sentiment Analysis: Text (English and Chinese) &#8594; positive or negative</li>
<li>NER: Named Entity Recognition: Text &#8594; named entities (substrings with labels such as person and location)</li>
<li>QA: Question Answering (SQuAD): Questions (text) &#8594; Answers (spans (substrings))</li>
<li>MT: Machine Translation: Text in source language &#8594; Text in target language</li>
<li>TTS: Text to Speech (also known as speech synthesis): Text &#8594; Audio </li>
<li>STT: Speech to Text (also known as speech recognition (ASR)): Audio &#8594; Text</li>
</ol>

Each subdirectory below should be self-explanatory.  Since there is so
much material here, I was concerned that users might feel overwhelmed.
To address this concern, there is a separate subdirectory for each
example.  Each subdirectory can be studied independently.  There are
no dependencies between subdirectories.

<p>

The emphasis is on simplicity and generality.  The code is short, easy
to read and easy to understand.  If one cares about speed or
performance on a leaderboard, there are probably better alternatives
elsewhere.

<p>

<ol>
<li> Click <a href="datasets">here</a> for examples of loading datasets from several sources, including PaddleNLP and HuggingFaceHub.</li>
<li> Click <a href="pretrained/examples">here</a> for examples of pretrained models for several applications from several sources.</li>
</ol>


Warning, there may be some intermittent timeouts while loading big
objects from far away.  If you run into a timeout, please try again.