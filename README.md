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

Post-training provides the LLM with guidance about the “default” universe within which it exists. Rather than asking the LLM to infer this universe from a prompt alone, post training can constrain the LLM to make certain assumptions or resolve ambiguities in consistent ways. There are many reasons this is necessary for making models useful. For example, LLMs might need to be told that, by default, they follow instructions. Otherwise, given a prompt like “*Write a report about George Washington*”, an LLM without post-training might happily generate a continuation of the instruction, e.g., something like “*It's due by 4:59pm on Friday*”, rather than generate the report that was requested. But post-training can be used to impose other defaults as well, such as influencing the model’s default behavior to be more consistent with social norms, however defined, ideally to make it a safer or more productive tool for its particular assumed use cases.  

We really like Murray Shanahan’s articulation that one way to conceptualize what these models might be doing is that they’re engaging in a form of [role-playing](https://arxiv.org/abs/2305.16367) that’s a function of their overall training recipe. Our intuition is that post-training teaches these models a coherent and default role to play in diverse deployment settings. Here’s a non-exhaustive list of what they might learn during post-training, ranging from the mundane and practical to the subjective and personal.

* **That the model should follow a  specific format.** For example, [Gemma’s formatter](https://ai.google.dev/gemma/docs/formatting) teaches it that it’s in a cinematic universe where there’s always a conversation between it and some arbitrary human user. In that universe, the role it’s being asked to play is described in the system instructions. Depending on the formatter, in each conversation, the human’s turn is always first.
* **That the model should “follow instructions” from the user.** That is, if the user gives it a string prompting it to “write an essay about a dog”, it should actually do that, rather than respond to the user with an increasingly bossy continuation of the instruction.
* **That the model should match the “real world” (as opposed to some other cinematic universe).** Post-training is often used to improve the model’s factuality by aligning its implicit or default cinematic universe to one that most users are likely to care about. For example, if you ask it “Where was $CELEBRITY born in?”, it should assume by default that we’re talking about the “real world” rather than some fan fiction world that it might have encountered online that shares a celebrity with the same name.
* **That it should be “safe”.** The internet is a complex web of normative standards. Needless to say, a fair amount of the internet would not be considered sanitary within the context of most global commercial deployments. Post-training helps align the model to a chosen distribution that can embody a range of safety policies, thereby imposing a normative standard on what the model should or shouldn’t generate. Ultimately, it is not possible for a model to generate something sufficiently complex without making some assumptions about norms.

#### Post-training Data Collection
**Broad Takeaway -** these models are ultimately trained and evaluated by human raters. When instructing a post-trained LLM, you are implicitly asking a digital role-player (i.e. the LLM) to role-play as a human rater (i.e. the person generating the post-training data) who is getting paid to role-play as an AI Assistant.

This section is a massive oversimplification. Substantially longer documents could be written about the complexities and vagaries of tasking human annotators with post-training LLMs. Our goal in this section is to provide an overall intuition for human annotation in this context, since it directly impacts how one thinks about prompting.

Essentially, the process of human data collection for post-training is roughly:
1. Create a dataset of a diverse range of input examples–i.e., prompts describing tasks that an LLM might be asked to do. This could be anything from “reformat this data as json” to “help me plan my wedding”.
2. Create a pool of human “raters” whose job is to tell a model what to do for these tasks. The rater’s job might be to write the gold-standard answers for these input examples, e.g., actually provide wedding-planning tips themself. Or it might be to view different responses generated by the model and rank them from best to worst. At different points in post-training, models can use different types of human-generated data.
3. Write some guidelines on how these raters should do this job. Often, the developer will include examples or specific details about the task and context to help the rathers understand the task better.
4. Collect this data and “post-train” the pre-trained model on it.
5. Ship it.

A large part of why LLMs are able to “act human” is because these statistical models are fitted to a large dataset of carefully collected demonstrations of human behavior. The pre-training phase, model architecture, learning algorithm, etc provide the core infrastructure and underlying capability for the model. But post-training provides the overall orientation of the model (via human demonstrations) which dictates how it will actually behave when it actually is deployed.

Post-training teams spend a substantial amount of time on quality control on their data. A lot of effort goes into matching raters with the prompts for which they are best suited. For example, to provide a good demonstration of how to respond to a prompt containing a hard Python debugging problem, it's necessary to find a rater who is themself a good Python programmer.

Collecting “high quality” data from human raters is extremely challenging. Some reasons include:
* **Rating examples can be boring compared to other jobs that require the same skills:** If you are an excellent Python programmer, it is probably more fun to work on your own coding projects than to spend 8 hours a day debugging programs which will be used to train an AI system. If you are a talented poet, you likely want to write your own poetry, not rank AI poems from best to worst. Of course, if you spend ~8 hours a day rating, you’ll get paid for it. But rating examples can be incredibly repetitive, and raters often have incentives based on throughput. There can be challenges with feelings of agency or ownership as well – you don’t always know how that data changed the model’s overall quality, whether that data is going to get thrown away, what model it’s used for, etc. It’s possible that your only relationship to the data’s utility is whether your supervisor tells you that you did a good job. If you don’t have a clear narrative of why being there in that job makes a positive change in the broader world, you might not find it a very meaningful use of our time. That might impact your unconscious enthusiasm for doing a “good” job, even if in the abstract you want to do a good job.
* Defining what golden data/demonstration looks like for a given task is very challenging. Especially when the definition necessarily intersects with existing norms about factuality, good expository writing or some other capability. The “boundaries” between good/bad for most interesting human artifacts is actually quite nebulous and contingent on many normative factors.
   * There’s a parallel with how the common law system works. It’s extremely difficult to write down legislation that can anticipate a large array of edge-cases. So society uses the judicial system to arbitrate edge-cases and create precedents.
   * The full complexity of social reality is really hard to nail down into a clear set of propositions.
* The raters might not understand the task, or might not have the skills to do the task despite a lot of effort spent on recruiting. This could be especially true when dealing with the sort of edge-cases described above.
* Raters/crowdworkers are humans and sometimes despite the best of circumstances can make mistakes. Likewise for the overall organizational infrastructure surrounding them. Human beings sometimes very reasonably make mistakes when they’re hungry, tired, sleepy, etc.

## Considerations for Prompting
**Broad Takeaway -** We’re actually writing system instructions and prompts for something like the aggregated spirit of the post-training team’s rater pool, seeded by the aggregated spirit of the pre-training corpus. If we write instructions that the average rater (within that specific domain) is likely able to understand, comprehend and faithfully follow, the model is more likely to follow our instructions.

It’s helpful to imagine that there’s a friendly, well-meaning and competent rater prepared to role-play on the other side of the screen whenever we write any system instructions. The text we provide is all they’ve got. That is, when we make an API call to Gemini, imagine that there’s a human rater on the other side that will carefully read our prompt and provide a response. When constructing prompts, it’s extremely helpful to take on their perspective and to consider our instructions in that light. For example, suppose the instructions are about generating Python code. If we randomly picked a competent Python engineer off the street and asked them to respond to these instructions, would they understand what we want? This metaphor starts to break down when we consider that this proverbial rater might have access to all human knowledge. They simply lack the wisdom and context to peer beyond the prompt that we’ve provided.

Therefore, here are some considerations to help you improve the instructions in our prompts. The considerations below will likely go stale very quickly as models get better. We’d suggest attempting to align with the overall spirit of this bullet list rather than the letter.

* Are our instructions clear, legible, concise and explicit?
   * For example, suppose our instructions are about some Python coding task. If we picked up a random Python expert off the street and asked them to pretend to be Gemini, are our instructions good enough for them to immediately understand what we mean without asking any obvious clarifying questions?
   * Bad
      * Write a Python function that computes prime numbers.
   * Good
      * Write a Python function that computes prime numbers from 1 to 100. Include pytype annotations for the generated function and use 2-space indentation.
* Are our instructions self-contradictory or otherwise hard to follow?
   * Would a bored, hungry, tired, etc. rater actually read our overly verbose instructions and faithfully follow them? Note that there is often substantial quality control involved in making sure all the instructions in a given prompt are followed. But humans are humans. Are our instructions actually “easy” to follow? Or do they contain needless indirection, verbosity, etc?
   * When the authors of this playbook write down instructions, we often ask ourselves whether another employee in our company could faithfully follow them if the instructions were presented to them with no additional context.
   * Bad
      * Don't write a story about a mean dog, unless it's friendly, and also sad, but not really that sad, and make it long even though I want it to be short. Oh, and also the dog should be named Bob, or maybe Susan, doesn't matter. Write it about a cat. Just kidding, but make the dog fluffy. 
   * Good
      * Write a short story (200-300 words) about a loyal golden retriever named Buddy who gets lost in the woods during a family camping trip. The story should focus on Buddy's journey and his determination to find his way back to his family.
* Similarly, are there too many instructions in a given system instruction?
   * We’ve noticed an inverse relationship between the number of instructions in a prompt and the model’s ability to faithfully follow all of them. Although we’ve definitely seen many cases where a model was able to follow long chains of instructions reasonably well. This is just a rule of thumb.
   * It’s difficult to provide good/bad examples for this consideration, since it’s heavily dependent on the model under consideration.
* Use positive instructions rather than negative instructions.
   * For example, consider:
      * Negative instruction - “Don’t ever end your response with a full stop.”
      * Positive instruction - “Your response should always end with an exclamation mark or a question mark”.
   * The former says what the model shouldn’t do. But it doesn’t say what the model should do. The latter very explicitly lays out what the model should do.
   * There are actually parallels here between effective human-to-human communication that one learns in contexts like teacher training or couple’s therapy.
   * We should try to imagine that we’re attempting to communicate with someone that wants to give us what we want. But we need to give them very explicit guidelines on what “success” means, rather than telling them to “avoid failure”.
* Good system instructions can act as “reminders” for the model.
   * When iterating on a new prompt, it’s really important to consider a very diverse range of input examples.
   * It’s very common to see people write prompts that might work for ~60-70% of possible model inputs, but are either unspecified or vague for the remaining ~30-40%.
   * It’s often worth giving the model an explicit set of positive instructions of what it should do in those situations. We often create a separate section in our system instructions called “Additional Considerations” or “Additional Assumptions” that contains a bullet list of specifications for these edge cases.
* Prompts are the new hyperparameters. For example, tuning the learning rate correctly can make a huge difference in the final performance of a model on the validation set for a specific compute budget.
   * Similarly, the difference between a “good” prompt and a “bad” prompt can have a substantial impact on the system’s final performance.
   * In the same vein, there’s probably always a “slightly better” prompt with more prompt tuning. And we’ll never know if we’ve found the “best one”. As discussed in the section below, we can leverage similar intuitions as the ones used in the [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) to systematically find “better” prompts than whatever baseline we currently have. The playbook speaks of “trial budgets” within the context of hyperparameter search. For prompting, it can be quite useful to create a timebox for our experimentation.
* Experiment with giving the model explicit instructions for saying “I don’t know”.
   * For example, suppose we’re working on some sort of multi-class text classification task. We have some criteria for how an input example should get mapped to each of the classes. It can be really helpful to create an additional “unknown” or “edge case” class. And to provide an explicit positive instruction for the model to classify the input into this class if it thinks that the instructions are unclear for correctly classifying this example.
   * We can then look at the logs to see when/how this happened, and improve the prompt accordingly.
* Prompts can be deeply coupled with the checkpoint that they were developed on.
   * That is, if we take a prompt from Gemini 1.5 Flash and run it on Gemini 1.0 Pro, it might not work the “same way” and might have very different aggregate behavior on an eval. This sort of makes sense. Our mental model is that the prompts we write in natural language are akin to the parameters we’d train if we instead did SGD. To what extent this is true is a question of open research.
   * Models are sort of like the machine, the post-training procedure is sort of like a compiler and the system instructions are sort of like computer code. Except that the machine and compiler are totally fluid, and a given model can hold many different combinations of these. We suspect that the ecosystem will organically converge towards some consensus structure around how instructions look, that rapidly changes across time and remains reasonably backwards compatible in a combinatorially explosive way. This is analogous to the x86 instruction set remaining relatively stable across time, as compared to the explosion in diversity of programming languages built on top of it.
