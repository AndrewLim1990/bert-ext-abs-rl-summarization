# Abstractive Summarization with BERT

This project aims to summarize long form articles from the CNN/Daily Mail summarization dataset in two or three sentences. This project uses the method described in this [research paper by Chen and Bansal](https://www.aclweb.org/anthology/P18-1063/).

At a very high level, this method works in three broad steps:

1. Select sentences for extraction
2. Create abstract summaries from extracted sentences
3. Use reinforcement learning to improve extraction and abstraction

A short minimum viable example of text extraction can be seen here: https://github.com/AndrewLim1990/bert-ext-abs-rl-summarization/blob/master/demo.ipynb