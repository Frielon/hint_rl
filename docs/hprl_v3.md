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

The reward differentiation across the group is what creates the learning signal:

- Trajectory using all $K$ hints and succeeding: reward $R_{\text{acc}} \cdot (1 - K \cdot \bar{w})$ — small but positive.
- Trajectory using $K-1$ hints and succeeding: reward $R_{\text{acc}} \cdot (1 - (K-1) \cdot \bar{w})$ — larger.
- Trajectory using $K-2$ hints and succeeding: even larger.

Each step the policy learns to bridge unaided raises its expected reward by one penalty unit. The optimizer sees a smooth gradient, *step by step*, instead of a single all-or-nothing event.

### 2.3 The complexity transformation

Informally, the per-step learning problem under hint RL has sample complexity $\sim 1/p$: bridging one specific step unaided requires observing $\sim 1/p$ rollouts in which the model attempts that step and succeeds. With $K$ steps to learn, total sample complexity is $\sim K/p$ — **linear in $K$** (up to log factors from group dynamics and budget ratcheting).

This is the central claim: hint RL transforms sample complexity from $1/p^K$ (exponential in problem depth) to roughly $K/p$ (linear in problem depth). On the $K = 5$, $p = 0.1$ example, that is $10^5$ vs $50$ rollouts in expectation — a four-orders-of-magnitude swing.

### 2.4 What can go wrong with this argument

The linear-complexity claim assumes:

- **Step independence of difficulty.** If steps are highly correlated (failing step 2 makes step 5 impossible to attempt), the per-step decomposition collapses. The hint corpus must restore independence by providing each step’s prerequisite context.
- **Hint sufficiency.** With all $K$ hints, the model must actually succeed at reasonable rate. If even full assistance fails, the trajectory yields zero reward at every budget level and the gradient channel never opens. See Section 10.
- **Hint non-leakage.** Hints must reduce step difficulty without giving away the step’s specific output. A leaky hint makes the unaided-step success probability under hint conditioning identical to the assisted-step probability — the partial-credit reward levels collapse into one.

When these conditions hold, the budget ratchet (Section 7) operationalizes the linear-complexity dynamic: easy steps are conquered first (budget drops), hard steps remain hinted (budget retained), and the per-problem curriculum emerges automatically.

## 3. Method Overview

A single rollout proceeds as follows:

1. A problem $q$ is sampled. Its current call-count budget $B_q$ is read from trainer state and inserted into the system prompt: *“You may call the hint tool at most $B_q$ times for this problem.”*
2. The policy generates reasoning tokens. At any point it may emit a hint-tool call.
3. On a hint call, the request is routed to a frozen **selector model** $\pi_{\text{sel}}$ — a stronger, off-policy reasoning model. The selector is given the policy’s current trajectory (the reasoning so far) together with the full pool $\mathcal{H}^q = \{h_1^q, \ldots, h_{K_q}^q\}$ of step-level hints, and returns a single index $\sigma_t \in \{1, \ldots, K_q\}$. Only the selected hint $h_{\sigma_t}^q$ is incorporated into the policy’s context for further reasoning; the policy never sees the pool or the selection reasoning (see Section 5).
4. If the policy attempts to call the tool more than $B_q$ times, the tool returns a no-op response.
5. The rollout completes with a final answer. Outcome correctness $R_{\text{acc}} \in \{0, 1\}$ is verified.
6. Each selected hint $h_{j_k}^q$ carries an importance weight $w_{j_k}^q$. The trajectory’s discounted reward is
    
    $$
    R = R_{\text{acc}} \cdot \max\left(0,\ 1 - \sum_k w_{j_k}^q\right).
    $$
    
7. After each training step, the budget $B_q$ ratchets downward on consistent success and upward on plateau (Section 7).
8. Only the **continuation reasoning** (the post-hint solving) is trained, under a standard single-sequence GRPO objective (Section 8). Hint selection is *not* learned by the policy — it is delegated to the frozen selector model, so no selection action, selection trajectory, or auxiliary selection loss is required.

## 4. What has been implemented

1. training data with hint pool: /share5/users/xutao.ma/project/hint_rl/dataset/dapo-3740-hint-verl-simplified.parquet

2. hint selection: calling llm, prompt: /share5/users/xutao.ma/project/hint_rl/selector/seletor_prompt.py
    see parser in: /share5/users/xutao.ma/project/hint_rl/selector/run_hint_selection_model.py

Now write a training script of hprl in /share5/users/xutao.ma/project/hint_rl/script/hint_rl, put every scripted added in this folder. for the reward computation, leave the hint penality function blank. in each rollout, maintain a state that record all the applied hints.