# Planting a Latent Variable in Natural-Looking Text: a More Realistic Test of Belief States in LLMs and Their Link to Concept Geometry

**Blog post: [Planting a Latent Variable in Natural-Looking Text](https://alexjerpelea.com/geometry.html)** — this repo contains the code and experiment results behind it. What follows is a shortened version of the post.

by Alex Jerpelea

*This is work in progress.*

## Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Method](#method)
  - [The Generative Process](#the-generative-process)
  - [The corpus](#the-corpus)
  - [Selecting the SAE latents](#selecting-the-sae-latents)
  - [The optimal observer](#the-optimal-observer)
  - [The student](#the-student)
- [Results](#results)
  - [1. The student tracks the belief states](#1-the-student-tracks-the-belief-states)
  - [2. The eight states of the latent variable sit on a ring](#2-the-eight-states-of-the-latent-variable-sit-on-a-ring)
- [Conclusion](#conclusion)
  - [Future Work](#future-work)
  - [Limitations](#limitations)
- [Repo layout](#repo-layout)

## Abstract

LLMs are thought to track "belief states," i.e., running probability distributions over the latent variables that govern language (Shai et al., Sarfati et al.), but so far this has only been comprehensively demonstrated on toy synthetic data and in a few isolated case studies. It has also never been empirically connected to the geometry of LLM features (the concepts interpretability finds in model activations). In this work, we plant a controllable latent variable inside natural-looking text. An LLM teacher writes ordinary text while we "subliminally" steer it along one of K = 8 unrelated sparse autoencoder directions at each token, with the active directions following a ring-shaped Markov chain. A small transformer model trained on this corpus does indeed track the Bayesian posterior belief about our planted latent variable. Moreover, it also arranges the 8 states themselves on a ring, in the exact order of the Markov chain, which is supporting evidence that a concept's geometry can be formed by the statistical dynamics of the latent variable behind it.

## Introduction

It is thought that natural language is governed by a set of latent variables, so a good predictor of language must infer their state, gaining a sort of world model. Shai et al. (2024) trains small transformers on token sequences emitted by a Hidden Markov Model (HMM), and finds that the residual stream linearly encodes a belief state, i.e., a running probability distribution over the state of the HMM, updating at each token.

But the paper's setup is purely synthetic, as the HMM dictates the whole toy language, and just because transformers *can* model belief states, it does not mean they *will* in a real natural language setting. A realistic controlled experiment is hard to set up as we can't model English with a bunch of Markov Models. LLMs would not be needed if we could.

![Figure from Shai et al.: a model learns to model optimal beliefs sitting on the probability simplex](figures/geo_fig01.png)

In parallel, the interpretability literature is documenting the geometry of concepts. The Linear Representation Hypothesis (Park et al.), which proposes features as linear directions, has been weakened to allow concepts to sit on low-dimensional manifolds, like days of the week on a circle, or calendar years on a helix (Engels et al.). There is no full proof of why these concept manifolds form.

In a follow-up paper, Shai et al. (2026) show that when text is modeled by multiple HMMs at once, the transformer learns a belief state per factor, each living in its own near-orthogonal subspace. The authors suggest that multi-dimensional feature manifolds could be stemming from here. However, this requires further explanation. *First of all*, their models are once again trained on synthetic HMM tokens, which is not a sufficient argument. *Second of all*, probing the belief state, which gives a probability distribution over a latent variable's values, is not the same as asking how the values themselves that the latent variable takes are geometrically arranged.

The diagram below better illustrates the difference between belief states and concepts/features within the residual stream. It is unclear whether belief dynamics are actually correlated with concept geometry.

![Belief states vs concept geometry in the residual stream](figures/geo_fig02.png)

This work tries to tie both ends by introducing a method for generating natural-looking text influenced by a latent variable with a Markov structure of our choosing. We use an LLM teacher that writes ordinary text while we "subliminally" steer it (Morgulis et al.) along one of K = 8 unrelated and orthogonal sparse autoencoder directions. Switching from one autoencoder direction to another is dictated by a ring-shaped Markov chain that we fully control. We then train a small transformer from scratch on the generated corpus and ask two questions. Does it track the belief state of our variable? (Yes, confirming the Shai et al. hypothesis in a more realistic scenario). And do the 8 states themselves inherit the geometry of the Markov Model? (Yes, they sit on a ring, in the exact neighbor order).

## Method

The main difficulty in generating natural-looking text governed by controllable latent variables is that it's very hard to write down the latent variables of natural language as a set of HMMs. But that's exactly the method's intuition! LLMs already model language with high accuracy, and although they are not fully explainable or controllable, they are steerable to some degree. And for our goal, one controlled latent variable is enough. So instead of trying to write down language's latent variables, we use steering to add one of our own: at each token, we steer the teacher along one of K = 8 uncorrelated and orthogonal SAE directions, while generating otherwise ordinary text. We only steer along one SAE direction at a time, given by a Markov chain which updates at every token. We then train a student model from scratch on the generated corpus.

One could consider the SAE directions to be 8 separate latent variables of natural language. However, by imposing this artificial dynamic over them, we create a new latent variable, and the 8 directions become values that the new latent variable can take. We also specifically pick the 8 directions to be orthogonal and uncorrelated, such that our structure could not have come from anywhere else.

Note that steering is known to inject durable, learnable structure into generated text, even if the text is seemingly unrelated to the steering direction (which is what we aim for). In their study, Morgulis & Hewitt call this mechanism "subliminal steering," finding that fine-tuning a student on random number sequences written by a steered teacher makes the student inherit the teacher's steering direction.

Here's a visual summary of our proposed algorithm:

![Method pipeline](figures/geo_fig03.png)

### The Generative Process

The teacher model is a Gemma-2-2B (base), which we couple with a pretrained sparse autoencoder (Gemma Scope, 16K latents) for the residual stream of a middle layer (layer #12). From these we select K = 8 uncorrelated latents. Each selected latent $i$ has a unit decoder direction $d_i$ and an average firing magnitude $a_i$. Steering along latent $i$ means adding $s \cdot a_i \cdot \frac{\rho_\ell}{\rho_{12}} \cdot d_i$ to the residual stream at layers 12 through 23, where $\rho_\ell$ is the average residual-stream norm at layer $\ell$, which compensates for the stream's growing norm. Lastly, $s = 1.5$ was chosen by a sweep, such that the injection is strong enough, yet does not degenerate the quality of the text.

The corpus consists of independent documents of 256 tokens each. The first few tokens come from a pre-selected set of "openers," and the teacher writes the rest. At every token, we steer the teacher along exactly one of the 8 directions. We write $z_t \in \{1, \ldots, 8\}$ for the direction active while the token at position $t$ is sampled. The 8 directions, together with the dynamics we impose on them, constitute the synthetic latent variable. One can think of it as a hidden "concept" governing the text, like the emotion variable drifting through a story, except that here we control the variable and hold its ground truth, instead of finding it in the wild. The latent variable takes a walk along the ring:

![The ring Markov chain](figures/geo_fig04.png)

At $0.95$ "stay" probability, the variable dwells ~20 tokens per state before hopping to the neighbor. Each document's path $z_1, \ldots, z_{256}$ is sampled before generation; thus, nothing else in the LM could plant the ring geometry.

Very importantly, the KV cache is always rebuilt at each token. Steering never touches the cached context, so that each token only depends on the current visible text and the latent variable. This will be very important in order to compute the outputs of an ideal Bayesian inference model, to which we will compare our student model's belief state.

### The corpus

We generate 400,000 documents x 256 tokens (first 12 tokens are taken from a pre-selected set of "openers"). Approx. ~102M tokens. We also generate a control corpus with steering off, and we train a control student on it. Analysis runs on both students.

### Selecting the SAE latents

Out of the 16K SAE latents, we drop the ones that are dead, fire too often, or mostly fire on punctuation and position. We don't really care about what our chosen latents mean, and aim for them to be as hard to eyeball as possible (so that they are more "latent"). We also make sure that in our selection, the latents are near-orthogonal, don't co-activate on natural text, and are causal (i.e., steering along them actually modifies the token distribution).

### The optimal observer

At each token position in a document, we measure how well the state of the latent variable can be tracked by an observer who knows everything, i.e., knows the transition matrix of the 8 latents, and the softmax token distribution of the steered teacher model given any text. Note that we can token-label our transition matrix just like Shai et al.'s formalisms, so we truly have an HMM:

$$T^{(x)}_{ij} = T_{ij} \cdot \mathrm{Teacher}_j(x \mid x_{<t})\text{, where } \mathrm{Teacher}_j \text{ is just the LLM with } z_t = j.$$

We call our proposed reader the *optimal observer*, which performs Bayes inference over the operator defined above, maintaining the exact posterior $b_t(i) = P(z_t = i \mid x_{1:t})$ at every token. Since $b_t$ is a probability distribution over the 8 states, it is a point in the probability simplex $\Delta^7$. The posteriors get updated as such at each token position $t$:

$$b_t(j) \propto \textstyle\sum_i b_{t-1}(i) \cdot T^{(x_t)}_{ij}$$

Because the KV cache is reset at every token, the next token only depends on the visible text and the current state of the model, thus making the optimal observer computable at every point. We use $b_t$ at every token as the probe target for our student.

The optimal belief $b_t$ sits on a 7-dimensional probability simplex. But if we PCA on it we find that the two principal components reflect the ring structure of $T$:

![PCA of the optimal observer's beliefs](figures/geo_fig05.png)

### The student

We use a 110M-parameter Llama-style transformer (12 layers, width 768), trained from scratch on the corpus tokens and nothing else (~5 epochs, 32k vocab). Unlike the optimal observer, the student has no access to the teacher model, so $z$ reaches it intertwined with all the other latent variables of the Gemma model. We ask whether the factorized belief state over $z$ emerges anyway, and we also explore the geometry of the latent variable.

We also train two control students. **control1** was trained on a completely unsteered corpus from the same teacher. **control2** was trained on a corpus steered by the same 8 latents with the same $p_{stay}$, but on a state switch, the latent variable jumps uniformly to any of the other 7 states instead of to neighbors on the ring. It has the same exposure to the 8 features and incentive to track them, but no transition map to learn:

![The student and the two controls](figures/geo_fig06.png)

## Results

Recall that this work has two objectives: 1. to confirm that transformers learn belief states over latent variables in a more realistic setting than the toy setup of Shai et al., and 2. to find the relationship between the geometry of a latent variable itself (*which is different than the geometry of the belief state!*) and its underlying statistical dynamics.

### 1. The student tracks the belief states

Let $h^{(\ell)}_t \in \mathbb{R}^{d_{model}}$ be the student's residual stream at layer $\ell$ and token position $t$. We want to test whether the belief state can be read out linearly out of this vector, so we fit a ridge regression from the hidden states to the posteriors of the optimal observer: $\hat{b}_t = W h^{(\ell)}_t + c$. We fit the map on 4,000 held-out documents from our circularly steered corpus, and evaluate it on 1,000 more. Per layer, we report the regression's $R^2$ against $b_t$, and the accuracy $\arg\max_i \hat{b}_t(i)$ against the true state $z_t$, averaged over all token positions.

![Belief probe R² and accuracy per layer](figures/geo_fig07.png)

Indeed, the belief posterior does live in the residual stream. The best single layer reaches $R^2 = 0.49$, with argmax accuracy of the actual state of $0.577$, which is approximately three quarters of the optimal observer's ceiling of $0.762$. This is significant given that the student had no direct access to the steered teacher model, as the optimal observer did.

As expected, **control1** does not map the belief state (low $R^2$), but does guess the current state above chance. The SAE directions we steer along are natural in language, so ordinary text has coincidentally exposed it to them without even steering. But note that **control2** reaches a significant $R^2 = 0.41$ on the same documents, also with high argmax accuracy ($0.52$). This is also expected, as control2 has seen the eight states just as much, so after enough tokens in one state, it can decently estimate which state is active. In other words, after enough tokens in a state, $b_t$ converges towards a one-hot vector, so there is nothing intrinsic about the ring geometry in it.

We believe the $R^2$ ($\approx 0.08$) difference between the two is the advantage that the student model has from knowing how the previous state constrains the next one, i.e., internalizing the HMM's structure. To confirm this, we re-evaluate on the *first dwell* only, where we define the first dwell of a document to be the prefix where the latent variable has not yet switched from its initial state. In this scenario, control2 and our student become indistinguishable. As there is no prior information, the transition map is useless even to the optimal observer, so this ablation is a positive result towards showing that the student calculates belief states:

![First-dwell evaluation](figures/geo_fig08.png)

For our student, we can also look at the probe's read-out $\hat{b}_t$ in the same PCA view we used for the optimal observer. The student's belief state is a noisier version of the optimal observer's:

![PCA of the student's belief read-out](figures/geo_fig09.png)

### 2. The eight states of the latent variable sit on a ring

So far we have shown that the student does hold beliefs resembling $b_t$. But control2 also does a decent job at inferring the belief state, because in our setup, each token carries some evidence about the current state, and, after enough tokens, that evidence alone suffices for identifying the state. Knowing the previous state only helps our student model early on, right after a state switch.

We now show where the models truly differ, which is how they represent the latent variable. Note that the geometry of the latent variable is a different object than the belief geometry above. We are no longer curious about the shape of a probability distribution, but about the shape of the states themselves. Specifically, we take the mean of the residual-stream activations for tokens in each of the 8 states (call them *centroids*) and we test what shape the 8 points form. If the geometry of the dynamics is inherited, they should sit on a ring.

We project the centroids onto their top-2 principal components. In layers 9-11, the 8 centroids sit in the exact neighbor order of our Markov chain, and pairwise distance between centroids correlates with distances along the ring at $0.45$. The control1 student shows nothing (with correlation $0.11$):

![Centroid PCA: student vs control1](figures/geo_fig10.png)

But note that this is a trap! Right after a switch from state 2 to state 3, the activations might still remember state 2 for a few tokens. So when we average all of state 3's tokens to get state 3's centroid, we are also getting some of state 2 (and state 4), because the previous state in the text is always a ring neighbor. This means that each of the 8 centroids gets dragged towards its neighbors, and that alone draws a ring. From now on, we only compute centroids on each document's *first dwell* (i.e., before the latent variable ever switches states), so that no previous states leak in. In this setting, the PCA circle does not survive:

![First-dwell centroid PCA](figures/geo_fig11.png)

However, this does not mean the ring is gone, but rather that the ring geometry is not the *principal* geometry (we only looked at the first 2 PCA axes). We use Fourier Analysis to directly query if there is any plane on which the centroids trace a circle, in the exact ring order of the HMM. Specifically, we look for two directions $u$ and $v$ such that:

$$\mu_i \approx u \cos(2\pi i/8) + v \sin(2\pi i/8),$$

where $\mu_i$, with $i \in \{1, 2, \ldots, 8\}$, is the centroid of the i-th state. The fitted probes $(u, v)$ explain 38% of the centroid variance at layer 11. For a significance test, we refit the same pattern under every possible way of arranging the 8 states on a ring (2520 orderings), and the true ring ranks 1st out of all 2520 in terms of explained variance, for our student. However, for the control1 and control2 models, the true order lands mid-distribution. So, the circle geometry is real, respecting the exact ring-order, and can be attributed to the HMM structure. Moreover, we have also finally differentiated the student from control2.

![Fourier ring test across all 2520 orderings](figures/geo_fig12.png)

We think the ring geometry is not dominant simply because our latent variable is not that relevant. Most of the residual stream is spent on ordinary language features, and our variable only injects ~0.05-0.1 nats of evidence per token. Moreover, the Gemma teacher surely models other latent variables too, and our 8 states could also be attributes that other latent variables land on. In other words, each centroid possibly sits in multiple geometries at once, each corresponding to other latent variables.

The ring structure is also visible in raw similarities. In the figure below, pairwise cosine similarity between centroids is warmest on the neighbor diagonals, resembling the matrix $T$. Moreover, the columns of a whitened ridge probe are most anti-correlated on the same diagonals. This supports our case, as the probe has to spend capacity on differentiating exactly the pairs that are most similar (i.e., the ones on the ring):

![Centroid cosine similarities and probe anti-correlations](figures/geo_fig13.png)

## Conclusion

We introduced a method for planting a latent variable inside natural-looking text, where an LLM teacher writes ordinary text while we "subliminally" steer it along SAE directions whose activity follows a Markov chain of our choosing.

By doing so, we confirmed that transformers learn belief state geometries in a more realistic setting than current work. Moreover, we have tied belief states to concept geometries: the geometry of a latent variable (not only the geometry of the belief state about that latent variable) is influenced by the dynamics of the data generating process.

### Future Work

- A first natural follow up is to plant multiple HMMs at once with our algorithm, including more complicated interactions between them.
- Many existing theories attribute feature manifolds to semantic similarity. Our setup can put the two in direct conflict: take semantically similar states (like different colors, for example) and tie them together in a Markov chain that's uncorrelated to their semantic similarity.
- Our ring was not the dominant geometry. Perhaps each state participates in multiple other latent variables. We are curious whether a possible manifold entanglement is happening in such scenarios.
- We are also interested in hierarchical concepts and how they fit in this picture.
- Lastly, to close the loop, we want to use this theory to make predictions about real LLMs: find a latent variable (maybe something like syntax), estimate its transition dynamics from the training corpus, and predict the geometry of that concept in a real LLM, like OLMo.

### Limitations

- Our setup, although more realistic than the toy setups, is still not full proof of Shai et al.'s hypothesis about belief state geometries. We impose a latent variable on language, which shows that transformers pick up such structure when it is there, and, although our setup is more natural, it suffers from the same limitation as the initial proponents of the theory.
- Our evidence for the concept geometry is just correlational as we have not done any causality experiments.
- We also have to expand our configurations for a more comprehensive study, i.e., more HMM structures, other choices of SAE latents, other teacher models, etc.
- Lastly, in order to construct a new latent variable, we had to use SAE directions, which could be latent variables themselves, or attributes that other latent variables take. This is not ideal as we were not able to fully isolate our planted latent variable (the ring geometry was not dominant), so there remains the open question of how to better inject latent variables in natural-looking text.

## Repo layout

One folder per experiment under `experiments/`, numbered and self-contained: each has its own `README.md` (spec + findings), `src/`, and `results/`. The headline experiment behind the blog post is `experiments/06_short_dwell`; earlier experiments (`01_latent_features`, `03_steered_corpus`, `04_clean_cache`, `05_multilayer_add`) built up the corpus generation, clean-cache machinery, and steering pipeline, and `07_uniform_control` trains the uniform-jump control student.
