# Hint Penalized RL: Boosting Reasoning with Verifiable Implicit Process Reward

## 1. Motivation

Outcome-reward RL on long-horizon reasoning problems is bottlenecked by the sparsity of the reward signal. On a problem requiring $K$ sequential reasoning steps where each step is hard, the policy receives reward $1$ only if every step succeeds and $0$ otherwise. Intermediate progress — getting $K-1$ of $K$ steps right — is invisible to the optimizer. As $K$ grows, the probability of an all-correct rollout collapses, and learning stalls.

Process reward models (PRMs) address this by issuing per-step rewards, but require per-step correctness annotations that are expensive to obtain at scale.

**Hint Penalized RL** is a method that recovers a per-step learning signal *without* per-step correctness labels, by:

1. Pairing each training problem with a corpus of step-level hints curated offline by a stronger model.
2. Letting the policy call a hint tool during rollout; on each call, a separate frozen **selector model** picks which hint to surface from the corpus given the current trajectory, and the call incurs a multiplicative penalty on the trajectory’s reward.
3. Maintaining a per-problem **call-count budget** that ratchets downward as the policy demonstrates capability, and ratchets upward when the policy plateaus.

The mechanism converts an all-or-nothing reward landscape into one with $K$ levels of partial credit. Per-step capability gains manifest as one fewer hint called, which directly raises the trajectory’s reward. The hint corpus serves as a substitute for a trained PRM: the budget mechanism is an *implicit* process reward model whose granularity is determined by the hint decomposition, not by an annotation pipeline.

Delegating *which* hint to surface to a separate reasoning model — rather than training the on-policy model to select — keeps the method lightweight: the policy only ever learns to reason, and all the selection-side training machinery (a discrete selection action, a second training sequence, an auxiliary selection loss) is removed.

## 2. Sample Efficiency: Exponential → Roughly Linear

This section is the heart of the proposal. It explains *why* the method is expected to help, in terms a newcomer can sanity-check.

### 2.1 The vanilla outcome-reward regime

Consider a problem decomposed into $K = 5$ canonical steps. Suppose the base policy succeeds on each individual step with probability $p = 0.1$, independently. Under vanilla outcome-reward RL, the policy is rewarded only on full trajectory success:

$$
\Pr[\text{trajectory success}] = p^K = 0.1^5 = 10^{-5}.
$$

To collect even *one* positive sample in expectation, the trainer must roll out $\sim 10^5$ trajectories on this problem. The group-relative advantage in GRPO is essentially always zero — no within-group variance, no gradient signal. Hard problems sit in a sparse-reward dead zone, and the policy never learns from them.

More generally, with success probability $p$ per step and $K$ steps, the expected number of rollouts to obtain one success is $1/p^K$. Sample complexity is **exponential in $K$**.

### 2.2 The hint RL regime

Now equip the policy with the hint tool and a budget of $K$ hint calls. The policy can ask for a hint at any subset of the $K$ steps; with all $K$ hints used, suppose the trajectory succeeds with probability close to $1$.

The key observation: trajectories with $K-1$ hints (i.e., the model attempts exactly one step unaided) succeed with probability $p \approx 0.1$ on the unaided step. This is *not* a rare event. In a group of $G = 8$ rollouts, the trainer will routinely observe a mix of $\{K, K-1, K-2, \ldots\}$ hint counts, with corresponding mix of successes.