## Who is this document for?

This document is for anyone who would like to get better at prompting post-trained LLMs. We assume that readers have had some basic interactions with some sort of LLM (e.g. Gemini), but we do not assume a rigorous technical understanding.

The first half of the document provides mental models on the nature of post-training and prompting. The second half of this document provides more concrete prescriptions and a high-level procedure for tuning prompts. Given the pace of innovation with LLMs, we suspect that the second half is likely to go stale a lot faster than the first half.

## Why a tuning playbook?

This playbook was inspired by the [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), a guide for tuning hyperparameters for deep learning workloads.

The “art” of prompting, much like the broader field of deep learning, is empirical at best and alchemical at worst. While LLMs are rapidly transforming numerous applications, effective prompting strategies remain an open question for the field. This document was born out of a few years of working with LLMs, and countless requests for prompt engineering assistance. It represents an attempt to consolidate and share both helpful intuitions and practical prompting techniques.

We are a pair of researchers and engineers that have worked with LLMs for a few years. Having said that, this document shouldn’t be viewed as a definitive truth nor should it be viewed as the collective position of the Gemini post-training team. Rather, it’s a collection of our personal observations and best practices. We hope that this playbook will act as a snapshot of our current thinking, which might get updated in the future on a best effort basis as our beliefs change and as new knowledge becomes available.

We hope that by writing down our concrete set of mental models and processes, the community can work together to find better and more systematic prompting strategies.

This playbook is exclusively focused on the various post-trained versions of Gemini. Anecdotally, some of the prescriptions in this document might generalize to other models. But we have less experience with them.

## Background: Pre-training vs. Post-training

### Pre-training

#### What is pre-training?
“Pre-training” is an old concept from deep learning. Essentially:

1. You have a small dataset that you actually care about (i.e. Dataset A), and a large Dataset B that isn’t actually A, but similar in at least some important aspects. For example, A could involve a small amount of mammography images and B could be a large academic dataset of natural images like ImageNet.
2. You train a model on the large Dataset B with the hope that it will learn some generally useful features. You then “fine-tune” it on Dataset A to get better performance on A’s validation set than if you trained the model directly from scratch on A. That is, you simply continue training on Dataset A using the same training procedure that you had used on Dataset B. This way, by the time your model encounters examples from Dataset A, it's able to make better use of them because it already knows a lot of generally-useful stuff from its extensive experience on Dataset B. 
3. To be more concrete, consider the mammography example again. By pretraining on the large set of readily-available images from the internet, your model can learn basic things like how to segment objects in an image, or how to recognize concepts regardless of their location within an image. These are important image processing skills that will be useful for your mammography application, but likely require lots of data in order to learn, and are not specific to mammograms. If you tried to teach your model these skills using only your (expensive to obtain, limited in supply) mammography data, it might never learn them, and thus never achieve its best performance. But if you pretrain on everyday images, your model can come to your mammography data armed with these general skills and ready to use your specialized data to learn only specialized skills that couldn’t be learned elsewhere.

One of the key ideas of training LLMs is to use “language modeling” -- that is, predicting the next word in a sentence -- as a pretraining task. It turns out that if you train a model to take an arbitrary piece of text from the internet, and do a good job of predicting the next word, the model implicitly learns a very rich structure of the world that’s been reflected within the web.

This seems easy enough to understand, until we try to answer the question: what world does the internet reflect? To try to wrap our heads around this question (and its answer) we suggest a useful if somewhat fanciful metaphor: the Cinematic Universe.

#### The “Cinematic Universe” Intuition of Pretraining

Large language models must learn about what the world is like by reading about the world in text. Text, though, has never been constrained to describe only things that are “true” in the conventional sense. Much attention is paid to misinformation or incorrect statements, but there are also lots of very innocent and desirable reasons why text does not and should not reflect a single factual reality corresponding to a single state of the world.

For example, consider the statement “Aragorn eventually becomes the king of Gondor”. Is that statement true? That depends. For example, it depends on some temporality. Moreover, whether that statement makes sense is also contingent on the broader premise or context within which it's being discussed. If the premise is Lord of the Rings (LoTR), then yeah, you could argue that this is a fact. But imagine that you’re instead talking within the premise of the Marvel Cinematic Universe. Then it’s not clearly factual. If you’re in the non-fictional cinematic universe compatible with what we conventionally consider “true”, then the statement we made about Aragorn is not true. It’s not true because Aragorn and Gondor are fictional characters that you can’t find on Earth. If you’re in the Marvel Cinematic Universe, then it’s also not true for a similar reason. But if you’re in the LoTR cinematic universe, then it becomes true.

This issue – i.e., the issue of struggling to define what it means for something to be “true” and with respect to what world – is not new to LLMs. It relates to a long history of philosophical and linguistic theory and argument. This history and theory is a worthwhile rabbit hole (see, e.g., [this overview](https://plato.stanford.edu/entries/truth/)). But, for practical purposes regarding prompting LLMs, it can be oversimplified as: Whether a statement is true or not depends on the “cinematic universe” that acts as the backdrop of the statement.

For the purposes of this document, you can think of the pretraining corpus as an approximation of the set union of all the cinematic universes produced by human culture. Or, more accurately, the cultures that heavily participate with the pretraining data sources like the web.

When you give the model a fixed context window (i.e. prefix), it will try to infer from that prefix what universe it is in, and it will then behave in accordance with the rules, conventions, and facts of that universe. If you provide a prompt with very strong signals about context, it will be easier for the LLM to recognize the script. For example, consider a prompt like “*The concrete jungle where dreams are made of isn't just a catchy lyric – it's the electric truth of New York City. From the soaring skyscrapers that pierce the clouds to the vibrant pulse of its diverse neighborhoods, NYC offers an experience unlike any other on Earth*”,   I.e., the first two lines of a blog post that I might write about NYC.) In this case, the model has very strong constraints on style and topic that will influence how it proceeds with the generation.

But, if your prompt is highly generic – say, “Hi, how are you?” — the LLM might not have enough context to understand which cinematic universe it’s supposed to be in. “Hi, how are you?” probably occurs in all kinds of contexts in the diverse corpora it was trained on. That is, there are many “modes” in the probability density function used to decode a generation. Or to put it in simpler terms, it sees many possibilities that it could role-play as. The text “Hi, how are you?”, or even something much longer, doesn’t give it enough context to disambiguate this.

That’s where post-training comes in.

### Post-training


