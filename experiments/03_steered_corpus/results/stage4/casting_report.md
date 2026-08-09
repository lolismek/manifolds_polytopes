# Casting report — exp03 variant A (stage 4)

Teacher: Gemma-2-2B + Gemma Scope 16k residual SAE, layer 12. Funnel: 16384 latents -> 10838 (cheap filters) -> 163 (causal screen) -> 48 (co-activation) -> 20 (audition) -> **cast of 8**.

## The cast

| state | latent | Neuronpedia meaning | KL@5x (nats/tok) | recall@8x (BoW) | NLL ratio@8x |
|---|---|---|---|---|---|
| 0 | 368 | entities and their definitions/relationships | 0.079 | 0.9 | 1.11 |
| 1 | 1220 | lawsuits and court proceedings | 0.094 | 0.9 | 1.17 |
| 2 | 2404 | legal or financial obligations | 0.094 | 0.9 | 1.27 |
| 3 | 2970 | conferences and related events | 0.090 | 0.9 | 1.22 |
| 4 | 6172 | reading and textual consumption | 0.103 | 0.8 | 1.28 |
| 5 | 10615 | TV news shows and their segments | 0.092 | 0.9 | 1.21 |
| 6 | 10621 | health, safety, organizational structure | 0.114 | 0.9 | 1.16 |
| 7 | 13931 | measurements and units of size | 0.081 | 0.9 | 1.25 |

Reserves: 9508 (promises/commitments), 10013 (mathematical notation).

Semantic note: 1220 and 2404 are both legal-flavored; the ideal-reader confusion matrix shows this is immaterial (1/24 docs confused at 3x, 0/24 at 5x).

## Strength verdict (ideal-reader posterior, 24 docs/state/strength, 300-token docs)

| strength | median tokens to P(true)>=0.9 | conv by 50/100/300 | end acc | mean entropy @25/50/100 tok |
|---|---|---|---|---|
| 3.0x | 98 | 0.13/0.49/0.948 | 0.964 | 1.574/1.184/0.606 (max 2.08) |
| 5.0x | 34 | 0.719/0.953/0.995 | 1.0 | 0.873/0.322/0.048 (max 2.08) |
| 8.0x | 13 | 0.995/1.0/1.0 | 1.0 | 0.146/0.011/0.001 (max 2.08) |

**Recommendation: 5x, dwell ~100 tokens (p_stay ~= 0.99 per token).**
- 8x: state resolved in ~13 tokens — caption, not carrier; posterior slams to a vertex almost immediately.
- 3x: median 98 tokens, but 5% of documents never converge in 300 tokens (end acc 0.964) and a
  reliable dwell would need to be ~250 tokens, leaving only ~4 transitions per 1024-token context.
- 5x: converges in ~34 tokens with zero end-of-document confusions; entropy at 25 tokens is 0.87
  of max 2.08, so beliefs are genuinely graded for the first third of a dwell. With dwell 100 the
  student sees ~10 state transitions per 1024-token context window — dense T signal.

Caveat noted for the record: convergence here is measured from a cold start (uniform prior at
token 1). Mid-document transitions start from the previous vertex and move mass to ring
neighbours, so in-corpus re-convergence will differ somewhat; the corpus-level posterior
machinery measures this exactly.

## Sample steered text (5x, one per state, first ~260 chars)

**state 0 — latent 368** (entities and their definitions/relationships):
>  Horror” in the horror movie for some hours, and it’s completely free for a few hours. A few hours later, there is also the paid part of the money. It is possible to purchase, but we will ignore the DLC for the time being. Just so you know the story, and the s...

**state 1 — latent 1220** (lawsuits and court proceedings):
>  Island” on the multiplayer servers for players on STEAM:  <blockquote>Survival Island is the next phase in our journey to introduce blockchain to the world. You will play a survival-themed multiplayer game in the Age of Crypto. We call this phase “Survival Is...

**state 2 — latent 2404** (legal or financial obligations):
> ”, “Out of the World”, “Electric Sheep”, “Super Doom,” “World Chess”, and another 88 more in a virtual recreation of the best video games of the 19th and 20th centuries-all of them, and the 28 more games represented within it, are playable simulations that giv...

**state 3 — latent 2970** (conferences and related events):
> ” in any major gaming platform. If you’re playing Steam, you can download it from there: http://store.steampowered.com/app/374860/ and if you’re playing itch.io, get the free version here: https://bitbucket.org/zanden/survival  If any community members don’t a...

**state 4 — latent 6172** (reading and textual consumption):
>  Horror Girl” at the Game Generator website!  I am excited my game actually didn’t suck. :)   I think I have successfully given my ideas an outline. Next part will be applying them and developing what will be good and bad. I think I’ll be able to add a decent ...

**state 5 — latent 10615** (TV news shows and their segments):
>  Guide,” a full 50 minute live session with a full house on my show yesterday! It is on youtube now, and you can join “Now Loading” on facebook. It has a big surprise show coming up today, so check back in to JoyBe with Billy the entire day for that.  <h3>Watc...

**state 6 — latent 10621** (health, safety, organizational structure):
> ” now.  Thank you for the 442 and more hours of my life spent, and above all, have fun!  You’ve been through a lot so here’s a big pat on the shoulder :  Thanks so much for playing! And, what ever you are doing, make sure to take your much needed breaks : )  I...

**state 7 — latent 13931** (measurements and units of size):
> , the first survival horror movie on Netflix” 19 hours a day, but I’m done for the week, just barely.  The week got off to a bad start, though, with a bad cold. I was going to be working 15 hours on Tuesday and I was a block away from home and I got a call say...
