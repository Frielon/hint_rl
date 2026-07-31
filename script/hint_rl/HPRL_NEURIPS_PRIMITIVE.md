# Hint-Penalized Reinforcement Learning: Process-Level Credit Assignment from Frozen Hint Guidance


## Abstract

Reinforcement learning with final-answer rewards provides little signal on long
mathematical derivations: a trajectory that completes all but one necessary step is
indistinguishable from one that makes no useful progress. Process reward models can
densify this supervision, but typically require a separate step-level annotation and
reward-model training pipeline. We introduce **Hint-Penalized Reinforcement Learning
(HPRL)**, a training-time scaffolding method that converts an ordered corpus of solution
hints into process-level policy-gradient credit without training a process reward model.
The policy receives an ordinary math prompt and repeatedly produces complete boxed
answers. After an incorrect attempt, a frozen selector examines the full trace, marks
newly completed substeps, identifies the first unfinished substep, and injects one hint
as a user message. Hints carry difficulty-weighted costs and define an ordered state
space over completed substeps.

For each problem, HPRL estimates state values directly from a group of sampled
rollouts. The resulting temporal-difference decomposition assigns a shared advantage to
each assistant turn, rewarding independently completed substeps and charging costs for
substeps that required hints. A value-integrated overlength term adds a group-relative preference
for completing a turn within its token limit, and per-group scale normalization produces
a stable policy-gradient magnitude. A persistent, raise-only per-problem budget grants
additional hints when an entire rollout group fails, increasing the chance that hard
problems eventually produce an anchored learning signal. HPRL therefore separates
reasoning from hint selection: the policy learns only to solve, while a frozen external
model supplies citation-grounded progress judgments and targeted training-time
interventions.

## 1. Introduction

Outcome-supervised reinforcement learning has become a simple and scalable way to train
language models for mathematical reasoning. A completion is sampled, its final answer
is checked, and a policy-gradient update increases the probability of successful
tokens. The simplicity of this pipeline is also its central weakness. If a problem
requires several dependent reasoning steps, the reward is observed only after the
entire chain succeeds. Correct intermediate work in an ultimately wrong solution is
invisible.

Consider an idealized problem with $K$ required steps and per-step success
probabilities $u_0,\ldots,u_{K-1}$. If all steps must succeed, the probability of an
unaided correct trajectory is

$$
P_0 = \prod_{k=0}^{K-1} u_k.
$$

When $u_k=u<1$, this probability is $u^K$. Thus the expected number of independent
rollouts required to observe one correct solution is $u^{-K}$. Group-relative methods
such as GRPO improve optimization once a group contains useful reward variation, but
they do not by themselves make a rare correct trajectory more likely.

Process supervision addresses the credit-assignment problem by judging intermediate
reasoning steps. Existing process-supervised approaches, however, commonly depend on
human step labels or a separately trained process reward model. This raises a practical
question: can we obtain a useful process-level signal from artifacts that are cheaper
and easier to generate than correctness labels over every policy trajectory?

HPRL begins with an ordered set of hints for each problem. The hints describe canonical
solution substeps and may be generated offline by a stronger model. During training, a
separate frozen selector compares the student's current reasoning with the hint set.
It identifies which pending substeps were completed independently and selects the first
substep that remains unfinished. The rollout loop then injects that hint. The policy
does not decide when or which hint to request and is not told that a hint budget exists.
It only learns to produce mathematical solutions.

The ordered hints serve three roles:

1. **Scaffolding.** They increase the probability that a difficult rollout eventually
   reaches a correct answer.
2. **State construction.** Selector judgments map free-form reasoning into discrete
   progress states.
3. **Credit assignment.** Needing a hint incurs a cost, while completing its underlying
   substep without help advances to a higher-value state.

The concrete HPRL variant in this paper makes the following contributions:

- We formulate a **push-hint rollout protocol** in which an external frozen selector
  injects one targeted hint after each incorrect boxed answer, leaving the policy's
  original prompt hint-agnostic.
- We view each completed rollout as a **walk through every ordered hint state**. A
  group of $N$ complete walks supplies $N$ return samples per state, from which we
  derive a group-local value estimator and its censored-walk generalization without
  training a critic or process reward model.
- We derive a **whole-turn macro-action advantage** that telescopes the underlying
  per-step temporal-difference terms while removing fuzzy citation boundaries from the
  token-level gradient.
- We combine the learning signal with **value-integrated overlength regularization**,
  per-group advantage-scale normalization, a persistent raise-only hint budget, and
  budget-grouped sampling.

This paper specifies the algorithm and its implementation. Empirical evaluation is
intentionally outside the scope of this primitive.

## 2. Related Work

### 2.1 Outcome-supervised reinforcement learning

PPO optimizes a clipped policy-gradient surrogate using on-policy trajectories
[1]. GRPO removes the learned critic by using rewards from multiple responses to the
same prompt as a group-relative baseline and has been applied to mathematical
reasoning [2]. HPRL retains grouped on-policy sampling and a clipped PPO update, but
replaces the final per-token GRPO advantage with a process-structured value
decomposition derived from the rollouts in each problem group.

Final-answer verifiers have long been used to improve mathematical reasoning, either
by ranking sampled solutions or by supplying outcome rewards [3]. Outcome verification
is attractive because exact answers are often cheap to check. HPRL preserves this exact
final-answer signal while using hints to expose distinctions among trajectories that
would otherwise share the same outcome.

### 2.2 Process supervision and reward models

Process supervision evaluates intermediate steps rather than only the final answer.
Prior work has compared process- and outcome-based feedback [4] and trained process
reward models from large step-level human-feedback corpora [5]. HPRL targets the same
credit-assignment bottleneck but uses a different supervision object. The offline
artifact is an ordered hint decomposition, not a label for every step of every sampled
trajectory. A frozen selector performs online alignment between the free-form trace and
that decomposition.

HPRL is not a formally verified process reward. Selector judgments can be wrong. Its
outputs are instead **citation-grounded**: newly completed hints must be supported by a
quote from the student trace. This makes the progress judgment auditable and provides a
localization signal, but it does not prove mathematical correctness.

### 2.3 Language-model interaction with tools and external feedback

ReAct interleaves language reasoning with external actions [6], while Toolformer trains
models to decide when and how to invoke APIs [7]. HPRL deliberately removes the
tool-selection action from the learned policy. The rollout controller, rather than the
policy, decides when to query the selector and which hint to inject. This separation
avoids an auxiliary tool-use policy and prevents hint-request behavior from becoming a
second learned objective.

### 2.4 Curriculum and self-improvement

Curriculum learning changes the examples or conditions presented to the learner over
training [8]. STaR and related self-improvement methods bootstrap reasoning from
successful generated rationales [9]. HPRL uses a per-problem feasibility controller:
when all sampled rollouts fail, the problem's future hint allowance increases. This is
a one-sided curriculum over available assistance, not over the policy prompt or the
evaluation distribution.

## 3. Problem Formulation

### 3.1 Tasks and policy

Let $q\sim\mathcal D$ denote a training problem with ordinary chat prompt $x_q$ and
ground-truth answer $a_q^\star$. The student policy is an autoregressive language model
$\pi_\theta$. A response is considered correct only if it contains a boxed answer and
the extracted answer passes an exact or symbolic grader:

$$
c(\tau)=
\mathbf 1\!\left[
\operatorname{grade}\!\left(
\operatorname{box}(\tau),a_q^\star
\right)=1
\right].
$$

The training rollout may contain multiple assistant turns and controller-injected user
turns. At evaluation time, the policy receives only the ordinary problem prompt and
generates unaided.

### 3.2 Ordered hint decomposition

Each problem has an effective ordered hint sequence

$$
\mathcal H_q=(h_0,h_1,\ldots,h_{K_q-1}).
$$

The implementation begins from hints grouped into major solution steps and removes
general step-guidance entries of the form `X.0`. The remaining elements are ordered
substep hints. We suppress the problem subscript when unambiguous and write
$K=K_q$.

The hint order induces $K+1$ progress states:

$$
S_k
=
\text{the state in which the first }k\text{ effective hints are completed},
\qquad k\in\{0,\ldots,K\}.
$$

$S_0$ represents no completed substep and $S_K$ is the fully solved state. The state
definition assumes that progress is a contiguous prefix of the canonical hint order.

### 3.3 Rollouts as walks through the state chain

For each sampled problem, the trainer generates a group of $N$ rollouts:

$$
\mathcal G_q=\{\tau_1,\ldots,\tau_N\}.
$$

The central view underlying HPRL is that a rollout is a walk through the ordered state
chain

$$
S_0
\longrightarrow
S_1
\longrightarrow
\cdots
\longrightarrow
S_K.
$$

The conceptual transition $S_k\to S_{k+1}$ is realized in one of two ways. The policy
may complete the substep associated with $h_k$ independently, in which case the
transition has zero hint cost. Alternatively, the policy may fail that substep and the
controller supplies $h_k$, in which case the transition has cost $p_k$ (defined in
Section 4). A single assistant turn can independently complete several consecutive
substeps; conceptually, that turn traverses all of the corresponding intermediate
states even though the implementation stores it as one macro-action.

Every correct rollout reaches $S_K$. It can therefore be expanded into a complete walk
that visits every state and supplies one observed transition outcome at each $S_k$, for
$k=0,\ldots,K-1$. If all $N$ rollouts complete the walk, then HPRL has exactly $N$
samples for estimating the value of every state. This repeated visitation is what
makes state-value estimation possible without a learned critic.

Not every sampled rollout is guaranteed to finish. Let $m_i$ be the furthest state
reached by $\tau_i$, with $m_i=K$ for a correct rollout. An incorrect rollout is a
right-censored prefix

$$
\tau_i:
S_0
\longrightarrow
\cdots
\longrightarrow
S_{m_i}.
$$

Define the state-visit indicator and visit count

$$
I_{i,k}
=
\mathbf 1[m_i\ge k],
\qquad
D_k
=
\sum_{i=1}^{N}I_{i,k}.
$$

Thus $D_k$ is the number of rollout samples available at state $S_k$. For $N$ complete
walks, $D_k=N$ at every state. With premature termination, Section 7 uses the
state-specific risk set of size $D_k$. The implemented configuration uses $N=8$, and a
group with no correct walk is excluded from the actor update because no rollout
supplies a verified correct terminal anchor. No learned value network is introduced.

## 4. Hint Costs

Each effective hint has a non-negative cost $p_k$. Costs are computed from major-step
and substep difficulty labels. Let

$$
\ell(d)=
\begin{cases}
0, & d=\text{easy},\\
1, & d=\text{moderate},\\
2, & d=\text{hard},
\end{cases}
\qquad
w(d)=\beta^{\ell(d)},
$$

where the implementation uses $\beta=1.5$.

Let $\mathcal M$ be the set of major steps and let $d_m$ be the difficulty of step
$m$. A total penalty mass $\tau_p$ is first divided across major steps:

$$
P_m
=
\tau_p
\frac{w(d_m)}
{\sum_{m'\in\mathcal M} w(d_{m'})}.
$$

Let $\mathcal H_m$ be the effective substep hints in major step $m$, with substep
difficulties $d_h$. The cost of $h\in\mathcal H_m$ is

$$
p_h
=
P_m
\frac{w(d_h)}
{\sum_{h'\in\mathcal H_m} w(d_{h'})}.
$$

The implemented method uses $\tau_p=1$. Hence, assuming every major step contains at
least one effective substep,

$$
\sum_{k=0}^{K-1}p_k=1.
$$

The original data also contains `X.0` guidance hints. Their weight is set to zero and
redistributed among the effective substeps, matching the fact that those guidance
hints are removed from the selector's state space.

## 5. Auto-Hint Rollout

### 5.1 Hint-agnostic student prompt

The policy receives a standard system instruction to solve the problem step by step
and place the final answer in `\boxed{}`. The prompt does not mention hints, a tool, or
a budget. The controller stores the hint pool and budget outside the prompt.

This design prevents the policy from optimizing a separate hint-request action. A
trajectory consists only of mathematical answer attempts, while the controller decides
when to intervene.

### 5.2 Turn protocol

Let $B_q$ be the maximum number of hints that the controller may inject for problem
$q$. Starting from $S_0$, the controller repeats:

1. Sample one assistant turn from $\pi_\theta$.
2. Require that the current turn contains a boxed answer.
3. Grade the accumulated solution.
4. If correct, terminate.
5. If incorrect and fewer than $B_q$ hints have been injected, invoke the selector,
   inject one hint, and continue.
6. If incorrect and the budget is exhausted, invoke the selector once more to label
   the terminal failed substep, but do not inject another hint.

Every assistant turn is therefore a complete answer attempt. A turn that terminates
without a box is treated as a turn-level failure, as is a turn that reaches its
per-turn token cap.

### 5.3 Frozen selector

Let $f_\phi$ be a frozen selector model. At round $j$, it receives:

$$
\left(
q,\,
\operatorname{trace}_{1:j},\,
\mathcal H_q,\,
\mathcal C_j
\right),
$$

where $\mathcal C_j$ is the set of hints already completed or delivered. Each hint is
rendered with status `completed` or `pending`.

The selector scans pending hints in order. It returns:

1. a set $\mathcal U_j$ of pending hints newly achieved by the current trace;
2. an exact supporting quote and a progress paraphrase for each element of
   $\mathcal U_j$; and
3. the first remaining pending hint $h_{e_j}$ that is not yet achieved.

The completed set is updated as

$$
\mathcal C_{j+1}
=
\mathcal C_j
\cup \mathcal U_j
\cup \{h_{e_j}\},
$$

when $h_{e_j}$ is actually delivered. For the terminal budget-exhausted label,
$h_{e_j}$ is used for credit assignment but is not added as an applied hint.

The progress state before a turn is the length of the longest completed prefix:

$$
s_j
=
\max\left\{
k:
\{h_0,\ldots,h_{k-1}\}\subseteq\mathcal C_j
\right\}.
$$

If the selector identifies $h_{e_j}$ as the first unfinished hint, then the turn has
independently progressed through state $S_{e_j}$ and failed the transition associated
with $h_{e_j}$. After that hint is delivered, the next turn begins at $S_{e_j+1}$.

### 5.4 Progress-aware injection

The injected user message contains:

- a statement that the previous boxed answer is invalid;
- a numbered list of all selector-verified progress and all previously delivered
  hints;
- the newly selected hint; and
- an instruction to continue from the accumulated progress and provide a new boxed
  answer.

The current hint is presented as new assistance, not as already completed. It is added
to the cumulative progress list only on later rounds. Injected user tokens are
conditioning context but have zero policy-loss mask.

### 5.5 Robust stopping

The controller stops rather than fabricating process labels when:

- the effective hint pool is exhausted;
- the selector API call fails;
- selector output cannot be parsed; or
- the selector explicitly declines because no usable pending hint remains.

These cases receive a zero-progress, no-failure turn annotation. The failure is logged
for observability but does not create an invented hint cost or state transition.

## 6. Scalar Outcome Reward

Let $\mathcal A_i$ be the set of hints actually injected into rollout $\tau_i$. The
scalar hint-penalized outcome reward is

$$
R_i
=
\begin{cases}
\max\!\left(
1-\displaystyle\sum_{h_k\in\mathcal A_i}p_k,\,
R_{\min}
\right),
& c(\tau_i)=1,\\[10pt]
0,
& c(\tau_i)=0,
\end{cases}
$$

with $R_{\min}=0.05$ in the implementation.

A missing-box turn or a hard length cut is assigned the incorrect reward $0$ and
`acc=0`. The selected variant has no hint-call bonus, format bonus, effort-shaping
term, pool-exhaustion penalty, or score-as-correct treatment for an incorrect answer.

The scalar reward records outcome quality and the cost of assistance. It also supplies
the correctness and hint-count statistics used by the budget controller. The actor
does **not** train on a scalar $R_i$ broadcast over every token: HPRL overwrites that
advantage with the process-structured signal below.

## 7. Group-Derived State Values

### 7.1 Rollout statistics

Using the walk representation from Section 3, let:

- $m_i=K$ if rollout $i$ is correct; otherwise, $m_i$ is the furthest
  selector-verified state it reached;
- $\mathcal F_i=(f_{i,1},\ldots,f_{i,J_i})$ be the list of hint indices recorded as
  failed transitions in rollout $i$;
- $z_i\in\{0,1\}$ indicate whether rollout $i$ ended in a turn-level failure; and
- $\kappa_i$ be the state index of that final turn-level failure when $z_i=1$.

For state $k$, define

$$
F_k
=
\sum_{i=1}^{N}
\sum_{j=1}^{J_i}
\mathbf 1[f_{i,j}=k],
$$

$$
D_k
=
\sum_{i=1}^{N}I_{i,k}
=
\sum_{i=1}^{N}
\mathbf 1[m_i\ge k],
$$

and

$$
T_k
=
\sum_{i=1}^{N}
\mathbf 1[z_i=1\ \land\ \kappa_i=k].
$$

$F_k$ is the number of recorded failed transitions at $h_k$, $D_k$ is the number of
rollouts that reached the corresponding state, and $T_k$ counts the subset that failed
the per-turn completion protocol at that state. Selector reconciliation is intended to
make state progress monotone, so a rollout contributes at most once to a given $F_k$.
Under that invariant,

$$
0\le T_k\le F_k\le D_k.
$$

### 7.2 Each state visit as a transition sample

Write $r_k=-p_k$ for the immediate reward when a rollout needs hint $h_k$, and
let $\lambda\ge0$ be the additional surcharge for a turn-level failure. For every
rollout that visits $S_k$, define its observed local transition reward as

$$
g_{i,k}
=
r_k\mathbf 1[k\in\mathcal F_i]
-\lambda\mathbf 1[z_i=1\ \land\ \kappa_i=k].
$$

The observation is $0$ if the rollout clears transition $S_k\to S_{k+1}$
independently, $-p_k$ if it needs hint $h_k$, and $-p_k-\lambda$ if that failure is
also a turn-level failure. A neutral controller stop at $S_k$, caused for example by
unusable selector output, contributes zero to the value-estimator numerator instead of
an invented failure label.

HPRL estimates the one-step reward at $S_k$ by averaging these observations over the
$D_k$ rollouts that visited the state:

$$
\widehat g_k
=
\frac{1}{D_k}
\sum_{i=1}^{N}
I_{i,k}g_{i,k}.
$$

Substituting the definitions of $F_k$ and $T_k$ gives

$$
\widehat g_k
=
\frac{F_kr_k-T_k\lambda}{D_k}
=
-\frac{F_kp_k+T_k\lambda}{D_k}.
$$

In the full-walk case emphasized in Section 3, all $N$ rollouts visit every state, so
$D_k=N$ and

$$
\widehat g_k
=
\frac{1}{N}
\sum_{i=1}^{N}g_{i,k}.
$$

Thus every state value is backed by $N$ rollout samples in the complete case. The
state-specific denominator $D_k$ is the implementation's censored-data generalization
when some rollouts stop early.

### 7.3 Backward value calculation

Set the terminal-state value to the unit correctness reward:

$$
V_K=1.
$$

Because the canonical transition from $S_k$ leads to $S_{k+1}$, the empirical
one-step Bellman backup is

$$
V_k
=
V_{k+1}
+\widehat g_k
=
V_{k+1}
+\frac{F_kr_k-T_k\lambda}{D_k}
=
V_{k+1}
-\frac{F_kp_k+T_k\lambda}{D_k},
\qquad
k=K-1,\ldots,0.
$$

Unrolling the recursion from the terminal anchor yields

$$
V_k
=
1
-\sum_{\ell=k}^{K-1}
\frac{F_\ell p_\ell+T_\ell\lambda}{D_\ell}.
$$

The connection to the $N$ complete walks is explicit. Define the realized downstream
return of rollout $i$ from state $S_k$ as

$$
G_{i,k}
=
1
+\sum_{\ell=k}^{K-1}g_{i,\ell}.
$$

When all $N$ rollouts visit every state, $D_\ell=N$ for all $\ell$, and the recursion
is exactly the sample-mean state-value estimator

$$
\begin{aligned}
V_k
&=
1+\sum_{\ell=k}^{K-1}\widehat g_\ell\\
&=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
1+\sum_{\ell=k}^{K-1}g_{i,\ell}
\right)\\
&=
\frac{1}{N}
\sum_{i=1}^{N}G_{i,k}.
\end{aligned}
$$

Thus each of the $N$ complete rollouts supplies one return sample for every state it
visits. When walks are censored, the implemented recursion replaces the common $N$
with the state-specific visit count $D_\ell$ at each transition.

The implementation uses $\lambda=0.1$. The value at $S_k$ is therefore the terminal
unit reward minus the estimated downstream assistance and turn-failure costs from
$S_k$ onward.

In a scored group, at least one rollout is correct, so $D_k\ge 1$ for every
$k<K$. If a general implementation encounters $D_k=0$, it sets $V_k=V_{k+1}$.

The recursion is empirical and problem-local: it is built from repeated visits to the
same ordered states rather than from a parametric critic. Because
$\widehat g_k\le0$, values are non-decreasing toward the solved state:

$$
V_0\le V_1\le\cdots\le V_K.
$$

### 7.4 Zeroing groups without a complete walk

If no rollout in the group is correct, the implementation assigns zero advantage to
every token:

$$
\left(\sum_{i=1}^{N}c(\tau_i)=0\right)
\Longrightarrow
A_{i,t}=0
\quad
\forall i,t.
$$

Without a correct complete walk, the unit correctness anchor is not observed. Zeroing
prevents selector-measured partial progress from being optimized toward a goal that no
member of the current group completed. The budget controller in Section 10 responds
by increasing future assistance.

## 8. Whole-Turn Advantage

### 8.1 Macro-action view

HPRL treats one complete assistant turn as a single macro-action. For turn $j$, let
$s_j$ be its starting state, $e_j$ its selector-verified state before any new hint is
applied, $q_j\in\{0,1\}$ indicate that the controller determines hint $h_{e_j}$ is
needed, and $z_j\in\{0,1\}$ indicate a terminal turn-level failure. The indicator
$q_j$ is a controller annotation, not a separately optimized policy action.

Define the post-turn state

$$
\bar e_j=e_j+q_j
$$

and adopt the harmless convention $p_K=0$. The advantage assigned to the whole turn is

$$
A_j^{\mathrm{turn}}
=
V_{\bar e_j}
-V_{s_j}
-q_jp_{e_j}
-z_j\lambda.
$$

This one expression covers solving turns, turns that independently traverse one or
more states, turns followed by an injected hint, and terminally labeled turns. For a
budget-exhausted terminal label, $\bar e_j=e_j+1$ is counterfactual: the equation
prices the hint the student still needed even though it was not injected. Every
trainable assistant token in turn $j$ receives the same
$A_j^{\mathrm{turn}}$.

### 8.2 Telescoping property

For the whole-turn transition, set $\sigma_j=s_j$,
$\sigma_{j+1}=\bar e_j$, and
$c_j^{\mathrm{turn}}=q_jp_{e_j}+z_j\lambda$. Then

$$
A_j^{\mathrm{turn}}
=
-c_j^{\mathrm{turn}}
+V_{\sigma_{j+1}}
-V_{\sigma_j}.
$$

Summing over a contiguous sequence of $J$ turns gives

$$
\sum_{j=0}^{J-1} A_j^{\mathrm{turn}}
=
-\sum_{j=0}^{J-1}c_j^{\mathrm{turn}}
+V_{\sigma_J}
-V_{\sigma_0}.
$$

For a solved rollout, $\sigma_J=K$ and $V_{\sigma_J}=1$. Thus the action-level
advantages telescope to the hint- and length-penalized path return minus the initial
state value. This identity is at the macro-action level. Because the implementation
broadcasts $A_j^{\mathrm{turn}}$ to every token of turn $j$, a token-weighted sum
additionally depends on turn length.

## 9. Advantage Scaling and Policy Optimization

### 9.1 Token assignment

Let $\mathcal T_j$ be the trainable assistant-token indices in turn $j$. HPRL assigns

$$
A_{i,t}^{\mathrm{raw}}=A_{i,j}
\qquad
\forall t\in\mathcal T_{i,j}.
$$

Controller-injected user tokens have loss mask zero and advantage zero. Tokens not
covered by a trusted turn annotation are also left at zero.

### 9.2 Per-group scale normalization

Let $\Omega_q$ be all trainable assistant-token positions in the problem group and
$M_q=|\Omega_q|$. Define

$$
\mu_q
=
\frac{1}{M_q}
\sum_{(i,t)\in\Omega_q}
A_{i,t}^{\mathrm{raw}},
$$

$$
\sigma_q
=
\sqrt{
\frac{1}{M_q}
\sum_{(i,t)\in\Omega_q}
\left(
A_{i,t}^{\mathrm{raw}}-\mu_q
\right)^2
}.
$$

For target scale $\alpha>0$, the normalized advantage is

$$
\widetilde A_{i,t}
=
\begin{cases}
\displaystyle
\alpha\,
\frac{A_{i,t}^{\mathrm{raw}}}{\sigma_q},
& \sigma_q>\varepsilon,\\[10pt]
\alpha A_{i,t}^{\mathrm{raw}},
& \sigma_q\le\varepsilon.
\end{cases}
$$

The implementation uses $\alpha=1$ and $\varepsilon=10^{-6}$. Importantly, it divides
by the standard deviation but does **not** subtract $\mu_q$ from each advantage. The
value recursion supplies the relative baseline; subtracting a second mean could change
the intended sign of progress and failure terms.

### 9.3 Asymmetric dual-clipped objective

Let

$$
\rho_{i,t}(\theta)
=
\exp\!\left[
\operatorname{clip}\!\left(
\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
-\log\pi_{\mathrm{old}}(y_{i,t}\mid x_i,y_{i,<t}),
-20,\,
20
\right)
\right]
$$

be the numerically stabilized token-level importance ratio used by the implementation.
Without the inner clamp, this is the usual probability ratio. First define the
asymmetric PPO-clipped token objective

$$
g_{\mathrm{clip}}(\rho,A)
=
\min\left(
\rho A,\,
\operatorname{clip}\!\left(
\rho,\,
1-\epsilon_{\mathrm{low}},\,
1+\epsilon_{\mathrm{high}}
\right)A
\right).
$$

The inherited verl actor loss also applies the dual-clip lower bound for negative
advantages [10]:

$$
g_{\mathrm{dual}}(\rho,A)
=
\begin{cases}
\max\!\left(g_{\mathrm{clip}}(\rho,A),\,c_{\mathrm{dual}}A\right),
& A<0,\\[4pt]
g_{\mathrm{clip}}(\rho,A),
& A\ge 0.
\end{cases}
$$

Because $A<0$ in the first branch, the term $c_{\mathrm{dual}}A$ prevents a large
importance ratio from making the negative-advantage objective arbitrarily negative.
The actor maximizes

$$
\mathcal L_{\mathrm{HPRL}}(\theta)
=
\frac{1}{|\Omega|}
\sum_{(i,t)\in\Omega}
g_{\mathrm{dual}}\!\left(
\rho_{i,t}(\theta),
\widetilde A_{i,t}
\right).
$$

The concrete configuration uses

$$
\epsilon_{\mathrm{low}}=0.20,
\qquad
\epsilon_{\mathrm{high}}=0.28,
\qquad
c_{\mathrm{dual}}=3.0.
$$

Loss aggregation is a mean over trainable tokens. No KL reward, KL loss, or entropy
bonus is used in this variant.

## 10. Dynamic Hint Budget

### 10.1 Persistent per-problem state

Each problem has a persistent budget $B_q$ equal to the maximum number of hints that
may be injected in a rollout. The budget is stored outside the model prompt and keyed
by `problem_id`. Data-loader workers read the current value and update only the
rollout-controller metadata.

Let

$$
C_q=\sum_{i=1}^{N}c(\tau_i)
$$

be the number of correct rollouts for problem $q$ in the current group. The configured
raise-only rule is

$$
B_q'
=
\begin{cases}
\min(B_q+1,B_{\max}),
& C_q=0,\\
B_q,
& C_q>0.
\end{cases}
$$

The implementation uses $B_{\max}=6$ and a fallback budget of 6. The displayed rule
assumes the stored value satisfies $B_q\le B_{\max}$. A no-decrease guard vetoes every
proposed reduction, so successful groups hold their current budget; if a pre-existing
stored value exceeds $B_{\max}$, it is held rather than reduced to the cap.

### 10.2 Interpretation

The budget is a feasibility controller for the anchored value decomposition. An
all-incorrect group has no actor update; increasing $B_q$ makes the next encounter
more likely to reach a correct terminal state through additional scaffolding. Once any
rollout succeeds, the group can define values and process credit. The budget then
holds, while the hint costs continue to distinguish solutions that require different
amounts of help.

This differs from a conventional competence curriculum that steadily removes
assistance. The selected HPRL variant is intentionally one-sided.

### 10.3 Budget-grouped sampling

At the beginning of each epoch, the sampler reads the live budget table, randomly
permutes problems, stable-sorts them by $B_q$, forms fixed-size prompt batches, and
randomly shuffles the order of those batches. The approximate homogeneity of budgets
within a batch reduces synchronous rollout stragglers because problems in the same
optimizer step tend to require a similar number of selector and generation rounds.

This sampling changes problem-to-step assignment but not the rollout group for a
problem or the per-epoch problem multiset, apart from the usual changing remainder
dropped for exact batch divisibility.

## 11. Why HPRL Densifies Learning

### 11.1 Probability of obtaining a value anchor

Under binary outcome training, let $P_0$ be the probability of an unaided correct
rollout. The probability that a group of $N$ independent rollouts contains at least
one success is

$$
Q_0
=
1-(1-P_0)^N.
$$

For small $P_0$,

$$
Q_0\approx NP_0.
$$

If $P_0=u^K$, the expected number of groups required to observe a success is
approximately

$$
\frac{1}{Q_0}
\approx
\frac{1}{Nu^K}.
$$

Let $P_B$ denote the success probability under an auto-hint budget $B$. The probability
of obtaining the correct-rollout anchor required by HPRL is

$$
Q_B
=
1-(1-P_B)^N,
$$

and the expected number of groups to obtain an anchor is $1/Q_B$.

If a sufficiently large hint budget keeps $P_B$ bounded below as $K$ grows, then anchor
acquisition no longer inherits the $u^{-K}$ dependence of unaided outcome sampling.
The number of controller interventions per rollout is bounded by $B$ and is at most
linear in $K$ when $B\le K$. This statement concerns interaction rounds, not selector
token cost: full-trace selector prompts grow across rounds and can have superlinear
aggregate length.
This is a conditional motivation, not an unconditional sample-complexity theorem:
poor hints, an inaccurate selector, or an insufficient budget may leave $P_B$ small.

### 11.2 State-local relative credit

Define the one-step value gap

$$
\Delta_k
=
V_{k+1}-V_k
=
\frac{F_kp_k+T_k\lambda}{D_k}.
$$

At state $k$, the underlying transition-level advantages are:

$$
A_k^{\mathrm{clear}}
=
\Delta_k,
$$

$$
A_k^{\mathrm{fail}}
=
-p_k+\Delta_k,
$$

and

$$
A_k^{\mathrm{trunc}}
=
-p_k-\lambda+\Delta_k.
$$

Let $N_k^0$ count rollouts included in $D_k$ that stop at $S_k$ without a trusted
clear or failed transition, for example after an unusable selector response. There are
$D_k-F_k-N_k^0$ clear transitions, $F_k-T_k$ ordinary failed transitions, $T_k$
turn-level failures, and $N_k^0$ neutral stops. The count-weighted sum of the assigned
transition advantages is

$$
\begin{aligned}
&(D_k-F_k-N_k^0)A_k^{\mathrm{clear}}
+(F_k-T_k)A_k^{\mathrm{fail}}
+T_kA_k^{\mathrm{trunc}}
+N_k^0\cdot 0\\
&=
(D_k-N_k^0)\Delta_k-F_kp_k-T_k\lambda\\
&=-N_k^0\Delta_k.
\end{aligned}
$$

Thus the balance is exactly zero on the regular labeled path, where $N_k^0=0$. With a
neutral controller stop, the implementation prefers a conservative residual over
inventing process credit from an untrusted label. Clearing a substep receives a
positive increment. Failing it receives the same value increment minus its hint cost;
a turn-level failure receives an additional $\lambda$ penalty. The whole-turn
advantage telescopes these local terms when one turn completes several substeps.


## 12. Implementation

### 12.1 Student and selector

The reference implementation trains Qwen3-8B-Base and uses an OpenAI reasoning model
as the frozen selector. The selector is not updated by HPRL. Reasoning-model requests
use a completion-token budget and reasoning-effort parameter; sampling temperature and
top-$p$ are omitted.

The Qwen3 base checkpoint is wrapped with a generation configuration that recognizes
both `<|endoftext|>` and `<|im_end|>` as EOS. This ensures that a complete chat turn
stops instead of running to its token cap.

### 12.2 Context and generation

The 32,768-token native context is allocated as:

| Quantity | Value |
|---|---:|
| Original prompt limit | 2,048 |
| Accumulated response limit | 30,720 |
| Assistant-turn limit | 8,192 |
| Maximum assistant turns | 10 |
| Rollouts per problem | 8 |
| Problems per optimizer step | 64 |

The accumulated response includes assistant tokens and controller-injected user tokens.
Only assistant tokens contribute to the policy loss.

### 12.3 Training configuration

The actor uses:

| Component | Value |
|---|---:|
| Learning rate | $10^{-6}$ |
| Schedule | constant after one warm-up step |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Loss aggregation | token mean |
| PPO clipping | asymmetric, $(0.20,0.28)$ |
| Dual-clip constant | 3.0 |
| KL reward | disabled |
| KL loss | disabled |
| Entropy coefficient | 0 |
| Advantage target standard deviation | 1.0 |
| Overlength surcharge $\lambda$ | 0.1 |

All training pods participate in one synchronous Ray/verl job. Agent-loop workers call
the external selector API directly; there are no selector-serving pods.

### 12.4 Observability and resume

The implementation records:

- per-step rollout traces;
- exact selector prompts, parsed selections, and raw outputs;
- selector failures, latency, citations, and token usage;
- state-value and advantage statistics;
- length-failure statistics;
- live per-problem budgets;
- validation rollouts; and
- a source snapshot for each run.

Checkpoints preserve model, optimizer, and data-loader state. The budget table is stored
outside the checkpoint, so an in-place resume must reuse both the experiment checkpoint
directory and its log directory.

## 13. Discussion

### 13.1 Hints as an implicit process interface

HPRL does not infer an unrestricted latent process state. It projects reasoning onto an
author-provided ordered hint ladder. This restriction is useful: it yields a small,
interpretable state space and gives every failed transition a concrete remediation.
It is also a modeling assumption. If the hint order is incomplete or incompatible with
a valid alternative solution, selector-derived states may misrepresent genuine
progress.

### 13.2 Separation of reasoning and selection

The policy is optimized only for reasoning tokens. The selector is frozen, its messages
are masked from the loss, and the policy never emits a hint action. This isolates the
student objective from selection behavior and lets the selector be replaced without
adding a second policy head or loss.

The separation also creates a dependency: the quality and availability of the external
selector directly affect the rollout distribution and process labels.

### 13.3 Whole-turn versus boundary-level credit

Whole-turn credit removes the citation boundary from gradient assignment. If a turn
makes useful progress and then fails, all of its tokens receive the combined
progress-plus-failure advantage. This is robust to fuzzy quote localization but coarse:
irrelevant restatement and useful reasoning within the same turn receive the same
coefficient.

### 13.4 Training-time scaffolding

Hints are present only during training. The intended transfer mechanism is that
substeps completed with help in one rollout become independently reproducible in future
rollouts, reducing the number of interventions needed before a correct answer. Held-out
evaluation remains unaided and measures the student rather than the student-selector
system.


## 15. Broader Impact

HPRL may reduce the amount of direct process labeling needed to improve reasoning
models, and its selector traces provide an auditable record of why assistance was
given. The same mechanism could support tutoring, code repair, or other tasks with
ordered remediation artifacts.

The method can also amplify errors in a stronger model's hints, encode a narrow
canonical solution path, and make training dependent on a proprietary external API.
Repeated full-trace calls increase compute, monetary cost, and exposure of training
content to the selector provider. Deployments should consider data governance,
selector access controls, prompt retention policies, and audits for systematic bias in
which reasoning styles are recognized as progress.

HPRL improves a model's ability to internalize externally supplied reasoning. As with
other reasoning-enhancement methods, this capability is dual use and should be
evaluated alongside domain-specific safety requirements.

## 16. Conclusion

HPRL converts an ordered hint corpus into a training-time process interface. A frozen
selector maps free-form student reasoning onto ordered progress states and supplies the
first missing substep after an incorrect answer. Difficulty-weighted hint costs and
group-derived state values then decompose the learning signal across assistant turns.
The whole-turn temporal-difference advantage, per-group scale normalization,
value-integrated overlength term, and raise-only assistance budget form a single
implementable algorithm.

The method preserves the operational simplicity of exact final-answer grading while
recovering structured credit from intermediate progress. Its central hypothesis is
not that hints replace outcome verification, but that targeted scaffolding can make
successful trajectories reachable and turn their internal state transitions into a
useful group-relative learning signal.

## References

[1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
“Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017.
<https://arxiv.org/abs/1707.06347>

[2] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. “DeepSeekMath: Pushing the
Limits of Mathematical Reasoning in Open Language Models.” arXiv:2402.03300, 2024.
<https://arxiv.org/abs/2402.03300>

[3] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz
Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher
Hesse, and John Schulman. “Training Verifiers to Solve Math Word Problems.”
arXiv:2110.14168, 2021. <https://arxiv.org/abs/2110.14168>

[4] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang,
Antonia Creswell, Geoffrey Irving, and Irina Higgins. “Solving Math Word Problems with
Process- and Outcome-Based Feedback.” arXiv:2211.14275, 2022.
<https://arxiv.org/abs/2211.14275>

[5] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy
Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. “Let's Verify Step by
Step.” arXiv:2305.20050, 2023. <https://arxiv.org/abs/2305.20050>

[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and
Yuan Cao. “ReAct: Synergizing Reasoning and Acting in Language Models.”
arXiv:2210.03629, 2022. <https://arxiv.org/abs/2210.03629>

[7] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke
Zettlemoyer, Nicola Cancedda, and Thomas Scialom. “Toolformer: Language Models Can
Teach Themselves to Use Tools.” arXiv:2302.04761, 2023.
<https://arxiv.org/abs/2302.04761>

[8] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. “Curriculum
Learning.” Proceedings of the 26th International Conference on Machine Learning,
pages 41–48, 2009. <https://doi.org/10.1145/1553374.1553380>

[9] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. “STaR: Bootstrapping
Reasoning With Reasoning.” arXiv:2203.14465, 2022.
<https://arxiv.org/abs/2203.14465>

[10] Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu,
Shaojie Yang, Xipeng Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei
Shi, Liang Wang, Qiang Fu, Wei Yang, and Lanxiao Huang. “Mastering Complex Control in
MOBA Games with Deep Reinforcement Learning.” arXiv:1912.09729, 2019.
<https://arxiv.org/abs/1912.09729>

---

# Appendix A. Algorithm

## Algorithm 1: Auto-Hint HPRL rollout and update

```text
Inputs:
  policy pi_theta
  frozen selector f_phi
  problem q with answer a*
  ordered effective hints H = (h_0, ..., h_{K-1})
  hint costs p = (p_0, ..., p_{K-1})
  current budget B_q
  group size N

for rollout i in {1, ..., N}:
    completed C_i <- empty set
    applied A_i <- empty list
    progress messages P_i <- empty list
    turn records Z_i <- empty list

    repeat:
        s <- length of contiguous completed prefix in C_i
        y <- sample one boxed-answer turn from pi_theta

        if y has no box or hits the per-turn cap:
            record turn (start=s, end=s, fail=true, turn_failure=true)
            mark rollout incorrect
            stop

        if accumulated answer is correct:
            record turn (start=s, end=K, fail=false, turn_failure=false)
            mark rollout correct
            stop

        terminal_label <- (number of applied hints >= B_q)

        if no pending hint exists:
            record turn (start=s, end=s, fail=false, turn_failure=false)
            mark rollout incorrect
            stop

        selection <- f_phi(q, full trace, H, C_i)

        if selection is unusable:
            record turn (start=s, end=s, fail=false, turn_failure=false)
            mark rollout incorrect
            stop

        U <- pending hints newly completed according to selection
        e <- index of selected first unfinished hint
        record turn (start=s, end=e, fail=true, turn_failure=false)
        C_i <- C_i union U

        if terminal_label:
            mark rollout incorrect
            stop

        inject cumulative progress plus h_e as a masked user turn
        append h_e to A_i
        C_i <- C_i union {h_e}

for each state k:
    F_k <- number of recorded failed transitions at h_k
    D_k <- number of rollouts that reached S_k
    T_k <- number of turn-level failures at S_k

if no rollout is correct:
    set every actor advantage in the group to zero
    if B_q < B_max:
        B_q <- B_q + 1
    else:
        hold B_q unchanged
else:
    V_K <- 1
    for k from K-1 down to 0:
        V_k <- V_{k+1} - (F_k p_k + T_k lambda) / D_k

    assign one macro-action advantage to every assistant turn:
        no failure:  A_j <- V[e_j] - V[s_j]
        failure:     A_j <- -p[e_j] - z_j lambda
                              + V[e_j+1] - V[s_j]

    divide group advantages by their trained-token standard deviation
    update pi_theta with the asymmetric dual-clipped token-mean PPO objective
    hold B_q unchanged

atomically persist the updated per-problem budget table
```

# Appendix B. Derivations

## B.1 Plain step-level decomposition

Set $\lambda=0$. Then

$$
V_k
=
V_{k+1}
-\frac{F_kp_k}{D_k},
$$

so

$$
\Delta_k
=
V_{k+1}-V_k
=
\frac{F_kp_k}{D_k}.
$$

A rollout that clears the step receives $\Delta_k$. A rollout that fails receives

$$
-p_k+\Delta_k
=
-p_k\left(1-\frac{F_k}{D_k}\right).
$$

Therefore

$$
\Delta_k\ge 0
$$

and

$$
-p_k+\Delta_k\le 0
$$

because $0\le F_k\le D_k$.

## B.2 Value-integrated turn-failure term

With $T_k$ turn-level failures and surcharge $\lambda$,

$$
\Delta_k
=
\frac{F_kp_k+T_k\lambda}{D_k}.
$$

Relative to a non-truncated failure at the same state, a turn-level failure receives
exactly $\lambda$ less:

$$
A_k^{\mathrm{fail}}-A_k^{\mathrm{trunc}}
=
\lambda.
$$

Relative to a clear transition, the gaps are

$$
A_k^{\mathrm{clear}}-A_k^{\mathrm{fail}}
=
p_k
$$

and

$$
A_k^{\mathrm{clear}}-A_k^{\mathrm{trunc}}
=
p_k+\lambda.
$$

The value integration redistributes the group-relative baseline but preserves these
pairwise gaps.

## B.3 Group transition balance

Let $N_k^0$ be the number of denominator rollouts that stop neutrally at $S_k$. After
substituting the three labeled transition advantages and the zero assigned to neutral
stops,

$$
\begin{aligned}
& (D_k-F_k-N_k^0)\Delta_k
+(F_k-T_k)(-p_k+\Delta_k)
+T_k(-p_k-\lambda+\Delta_k)
+N_k^0\cdot 0\\
&=
(D_k-N_k^0)\Delta_k-F_kp_k-T_k\lambda\\
&=
(D_k-N_k^0)
\frac{F_kp_k+T_k\lambda}{D_k}
-F_kp_k-T_k\lambda\\
&=
-N_k^0
\frac{F_kp_k+T_k\lambda}{D_k}\\
&=-N_k^0\Delta_k.
\end{aligned}
$$

The count-weighted balance is exactly zero when $N_k^0=0$. Neutral failure paths leave
a non-positive residual because the controller withholds an untrusted transition
label. Even when the count-weighted balance is zero, the token-weighted mean need not
be zero when turns have different lengths.

## B.4 Whole-turn telescoping

For a failed turn that progresses from $S_s$ to $S_e$ before needing $h_e$,

$$
\begin{aligned}
A_{\mathrm{turn}}
&=
\sum_{k=s}^{e-1}
\left(V_{k+1}-V_k\right)
+\left(r_e+V_{e+1}-V_e\right)\\
&=
V_e-V_s+r_e+V_{e+1}-V_e\\
&=
r_e+V_{e+1}-V_s.
\end{aligned}
$$

The overlength surcharge, when present, subtracts $\lambda$ from the last expression.

# Appendix C. Exact Selected Variant

| Design axis | Selected setting |
|---|---|
| Student interaction | controller-pushed hint after wrong boxed answer |
| Student prompt | ordinary solver prompt; no hint/tool/budget text |
| Selector | frozen OpenAI reasoning model |
| Hint granularity | one ordered substep per round |
| Guidance hints `X.0` | pruned |
| Progress message | cumulative |
| Reward strategy | per-hint |
| Hint-call bonus | off |
| Effort shaping | off |
| Finalize-incorrect reward | off |
| Hint total penalty | 1.0 |
| Step advantage | on |
| Advantage assignment | whole turn |
| Advantage scale normalization | on, target standard deviation 1.0 |
| Overlength routing | value-integrated |
| Overlength surcharge | 0.1 |
| Policy loss | asymmetric dual-clip PPO, $(0.20,0.28,c_{\mathrm{dual}}=3.0)$ |
| Verified-prefix mask | bypassed |
| Budget update | adaptive raise-only |
| Budget decrease | disabled |
| Budget-grouped sampling | on |
| K-pack probing | off |
| Training mode | synchronous |

# Appendix D. Failure Semantics

| Event | Outcome label | Process annotation |
|---|---:|---|
| Correct boxed answer | correct | solving turn reaches $S_K$ |
| Wrong box with budget remaining | incorrect so far | failed selected substep; inject hint |
| Wrong box at budget limit | incorrect | terminal failed substep; no injection |
| Missing box | incorrect | turn-level failure at current state |
| Per-turn token cap | incorrect | turn-level failure at current state |
| First-turn global cap | incorrect | failed first substep without overlength surcharge |
| Later global cap | incorrect | later unrecorded tail remains neutral |
| Pool exhausted | incorrect | zero-progress, no-failure segment |
| Selector failure | incorrect | zero-progress, no-failure segment |
| Explicit selector decline | incorrect | zero-progress, no-failure segment |

# Appendix E. Reproducibility Map

| Component | Repository file |
|---|---|
| Cluster and OpenAI launch | `launch_hprl_cluster_openai.sh` |
| Selected Qwen3 method defaults | `run_auto_hint_qwen3_8b_base.sh` |
| PPO and runtime configuration | `run_hprl_qwen2.5_7b.sh` |
| Trainer entry point | `main_hprl.py` |
| Auto-hint rollout | `auto_hint_agent_loop.py` |
| Selector prompt and progress state | `selector_multi.py` |
| Selector API client | `hint_selector.py` |
| State values and advantages | `step_advantage.py` |
| Advantage injection and budget hook | `hprl_ray_trainer.py` |
| Scalar reward | `hint_reward.py` |
| Hint costs | `hint_penalty.py` |
| Budget controller | `budget_manager.py` |
| Live budget data injection | `hint_dataset.py` |
| Budget-grouped sampler | `budget_sampler.py` |
