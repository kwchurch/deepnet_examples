# fine-tuning example based on HuggingFaceHub

fine_tune_glue.sh is based on <a href="https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification">https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification</a>.


<p>
fine_tune_glue.sh takes two args
<br>
output goes to a directory named tmp.$1
<br>
$2 should be a task, one of: cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli
