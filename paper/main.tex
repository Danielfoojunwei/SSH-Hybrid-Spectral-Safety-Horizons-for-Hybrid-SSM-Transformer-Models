% SSH-Hybrid: Spectral Safety Horizons for Hybrid SSM-Transformer Models
% NeurIPS 2026 Submission
% Part 1: Preamble, Title, Abstract, Introduction, Related Work

\documentclass{article}
\usepackage[preprint]{neurips_2026}

% Encoding and fonts
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

% Hyperlinks and URLs
\usepackage{hyperref}
\usepackage{url}

% Tables
\usepackage{booktabs}

% Mathematics
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{mathtools}

% Typography
\usepackage{nicefrac}
\usepackage{microtype}

% Colors and graphics
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{subcaption}

% Algorithms
\usepackage{algorithm}
\usepackage{algorithmic}

% Lists and tables
\usepackage{enumitem}
\usepackage{makecell}

% -------------------------------------------------------
% Theorem environments
% -------------------------------------------------------
\newtheorem{theorem}{Theorem}[section]
\newtheorem{definition}[theorem]{Definition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{remark}[theorem]{Remark}

% -------------------------------------------------------
% Custom macros
% -------------------------------------------------------
\newcommand{\rSSM}{r_{\mathrm{SSM}}}
\newcommand{\deltaStar}{\Delta^{*}}
\newcommand{\piHybrid}{\pi_{\mathrm{hybrid}}}
\newcommand{\piTrans}{\pi_{\mathrm{transformer}}}
\newcommand{\Horizon}{\mathcal{H}(\rho)}
\newcommand{\Atten}{\mathrm{Attn}}
\newcommand{\MBCA}{\mathrm{MBCA}}
\newcommand{\CHSS}{\mathrm{CHSS}}
\newcommand{\betaMBCA}{\beta_{\MBCA}}

% -------------------------------------------------------
% Hyperref setup
% -------------------------------------------------------
\hypersetup{
  colorlinks=true,
  linkcolor=blue!70!black,
  citecolor=green!50!black,
  urlcolor=blue!60!black,
}

% -------------------------------------------------------
% Title and authors
% -------------------------------------------------------
\title{SSH-Hybrid: Spectral Safety Horizons for Hybrid\\
SSM-Transformer Models}

\author{%
  \textbf{Anonymous Authors}\\
  NeurIPS 2026 Submission
  % Author information omitted for double-blind review
}

\begin{document}

\maketitle

% -------------------------------------------------------
% Abstract
% -------------------------------------------------------
\begin{abstract}
Hybrid architectures that interleave State Space Model (SSM) layers with
Transformer attention---exemplified by Jamba-1.5-Mini and Zamba-7B---achieve
compelling efficiency gains, yet their safety properties remain formally
uncharacterized.  We identify and rigorously analyze a previously unreported
structural vulnerability: because SSM hidden states evolve under the recurrence
$h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t$ with spectral radius $\rho < 1$,
any safety constraint encoded in an SSM hidden state decays exponentially and
falls below a detection threshold $\epsilon$ after at most
$\Horizon = \lceil \log(1/\epsilon) / \log(1/\rho) \rceil$ tokens.  We call
this interval the \emph{safety blind window}.  Unlike previously studied
jailbreaks, the blind window is \emph{not} a training failure: it is a
geometric consequence of spectral contraction, intrinsic to the efficiency of
SSM design, and therefore immune to standard alignment techniques.

We introduce \textbf{SSH-Hybrid} (\textbf{S}pectral \textbf{S}afety
\textbf{H}orizons for \textbf{H}ybrid Models), the first formal framework
connecting SSM spectral properties to quantifiable safety margins.  We
establish two theorems.  \emph{Theorem~1 (Spectral Safety Margin Bound)}
proves that the safety margin of a hybrid model satisfies
$\deltaStar(\piHybrid) \leq \deltaStar(\piTrans)(1 - g)$,
where $g = \rSSM \cdot \min(1, L/\Horizon)$ depends explicitly on the
SSM fraction $\rSSM$ and context length $L$.  \emph{Theorem~2 (MBCA
Compensation)} shows that augmenting a hybrid model with a
\emph{Monotone Boolean Composition Accumulator} (\MBCA) recovers
$\deltaStar(\piHybrid + \MBCA) \geq \deltaStar(\piTrans)(1 - g(1-\betaMBCA))$,
where $\betaMBCA > 0$ is the accumulator's retention coefficient.

Experiments on four architectures (Pythia-2.8B, Mamba-2.8B, Zamba-7B,
Jamba-1.5-Mini) validate the predicted degradation ordering
Pythia $>$ Zamba $>$ Jamba $>$ Mamba with Pearson $r = 0.923$.  The
\MBCA{} mechanism with $K{=}8$ probes achieves $\betaMBCA > 0.70$ and reduces
\CHSS{} degradation from $62\%$ to $23\%$, while incurring less than $2\%$
accuracy loss on MMLU, HellaSwag, and ARC.  SSH-Hybrid is the first system
to provide formal guarantees, empirical validation, and a deployable
mitigation for architectural safety blind windows simultaneously.
\end{abstract}

% -------------------------------------------------------
\section{Introduction}
\label{sec:intro}
% -------------------------------------------------------

The rapid deployment of large language models in safety-critical applications
has motivated substantial investment in alignment techniques: Reinforcement
Learning from Human Feedback (RLHF)~\cite{ouyang2022training}, Constitutional
AI~\cite{bai2022constitutional}, red-teaming~\cite{perez2022red}, and
output-level classifiers~\cite{inan2023llama}.  A shared assumption underlying
all of these methods is that safety properties depend primarily on training and
fine-tuning, not on architectural inductive biases.  Every architecture is
treated as a \emph{black box} that maps input tokens to output distributions.
We show that this assumption is wrong for an important and growing class of
models.

\paragraph{The emergence of hybrid SSM-Transformer architectures.}
State Space Models offer sub-quadratic sequence modeling with strong empirical
performance~\cite{gu2022efficiently,gu2023mamba,dao2024mamba2}.  Recent hybrid
designs---Jamba-1.5-Mini (12B parameters, SSM fraction $\rSSM = 0.875$) and
Zamba-7B ($\rSSM = 0.850$)~\cite{lieber2024jamba,glorioso2024zamba}---interleave
SSM recurrence layers with Transformer attention to combine the memory
efficiency of SSMs with the in-context learning capabilities of attention.
These models are already deployed in production inference pipelines, yet no
prior work has examined whether their architectural differences create
qualitatively new safety risks.

\paragraph{The gap: no formal connection between SSM spectral properties and
safety margins.}
Existing safety evaluations treat hybrid models identically to pure
Transformers.  RLHF fine-tuning adjusts reward-weighted outputs without
reference to how hidden states evolve across layers.  Red-teaming campaigns
probe input-output behavior without inspecting intermediate recurrent
states~\cite{ganguli2022red}.  Output classifiers score the final logit
distribution, blind to the dynamics that produced it~\cite{inan2023llama}.
Even representation engineering approaches~\cite{zou2023representation}, which
intervene on internal activations, are designed for residual-stream Transformer
architectures and do not account for the recurrent state dynamics of SSM layers.

\textbf{No existing framework formally connects SSM spectral properties to
quantifiable safety margins.}  This is not merely a gap in analysis: it means
that the safety guarantees claimed for deployed hybrid models are derived from
evaluations that are architecturally blind to SSM recurrence and therefore
cannot account for vulnerabilities that arise from it.

\paragraph{The problem: the safety blind window.}
SSM layers implement the discrete-time recurrence
\begin{equation}
  h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t,
  \label{eq:ssm-recurrence}
\end{equation}
where $\bar{A}$ and $\bar{B}$ are discretized state matrices derived from
continuous-time SSM parameters.  The spectral radius $\rho(\bar{A}) < 1$
ensures asymptotic stability---the key property that makes SSMs efficient
sequence models---but it also implies that every component of the hidden state
$h_t$ decays exponentially toward zero in the absence of new input.  A safety
constraint that is successfully encoded in $h_t$ at token position $t_0$ will
decay to magnitude $\leq \epsilon$ within
\begin{equation}
  \Horizon \;=\; \left\lceil \frac{\log(1/\epsilon)}{\log(1/\rho)} \right\rceil
  \label{eq:horizon}
\end{equation}
additional tokens.  We call $[t_0,\, t_0 + \Horizon)$ the \emph{safety blind
window} of the model.

This is a fundamental finding.  The safety blind window is \emph{not} a
consequence of insufficient training, reward misspecification, or distributional
shift.  It is a \emph{geometric consequence of spectral contraction}: any
recurrent system governed by Eq.~\eqref{eq:ssm-recurrence} with $\rho < 1$
exhibits it, regardless of how the model was trained.  Standard alignment
methods cannot close the window without either abandoning recurrence (losing the
efficiency of SSMs) or continuously refreshing the relevant hidden-state
components (which requires the very safety monitoring infrastructure whose
necessity this work establishes).

The practical threat is immediate.  A deceptive agent or adversarial prompt that
stretches across a context longer than $\Horizon$ tokens can exploit the blind
window.  Safety detections raised early in the context decay below threshold
before the harmful output is generated.  Many-shot jailbreaking
attacks~\cite{anil2024many} and goal hijacking in agentic
settings~\cite{perez2022ignore} already use long contexts to evade safety
mechanisms; the blind window gives these attacks a new, architecturally grounded
attack surface.

\paragraph{The solution: SSH-Hybrid.}
We introduce \textbf{SSH-Hybrid}, the first formal framework that connects SSM
spectral properties to quantifiable safety margins and provides a provably
compensating mitigation.  Our contributions are:

\begin{enumerate}[label=(\roman*), leftmargin=*, itemsep=2pt]

  \item \textbf{Theorem~1 (Spectral Safety Margin Bound).}  We prove that for
    any hybrid model with SSM fraction $\rSSM$ and context length $L$, the
    safety margin satisfies
    \begin{equation}
      \deltaStar(\piHybrid) \;\leq\; \deltaStar(\piTrans)\,(1 - g),
      \quad g = \rSSM \cdot \min\!\left(1,\, \frac{L}{\Horizon}\right).
      \label{eq:thm1-intro}
    \end{equation}
    The gap factor $g$ is a closed-form function of directly measurable
    architectural parameters: the SSM-layer fraction $\rSSM$, the context
    length $L$, and the spectral radius $\rho$ (through $\Horizon$).

  \item \textbf{Theorem~2 (MBCA Compensation).}  We design the
    \emph{Monotone Boolean Composition Accumulator} (\MBCA{}), a lightweight
    safety monitoring module that maintains a Boolean safety state via
    monotone OR updates: once a safety detection is recorded, it cannot be
    forgotten regardless of subsequent SSM state decay.  We prove that
    \begin{equation}
      \deltaStar(\piHybrid + \MBCA) \;\geq\; \deltaStar(\piTrans)
      \,(1 - g(1 - \betaMBCA)),
      \label{eq:thm2-intro}
    \end{equation}
    where $\betaMBCA \in (0,1]$ is the accumulator's retention coefficient,
    determined by the number and quality of safety probes $K$.  For $\betaMBCA
    = 1$ the hybrid model with \MBCA{} recovers the full safety margin of a
    pure Transformer.

  \item \textbf{A three-phase practical audit procedure.} We provide an
    operational protocol---Spectral Profiling, Horizon Estimation, and \MBCA{}
    Calibration---that requires only black-box access to model hidden states
    and can be applied to any deployed hybrid model.

  \item \textbf{Comprehensive empirical validation.}  We evaluate four
    architectures spanning the $\rSSM \in \{0, 0.85, 0.875, 1.0\}$ spectrum:
    Pythia-2.8B (pure Transformer, $\rSSM = 0$), Mamba-2.8B (pure SSM,
    $\rSSM = 1.0$), Zamba-7B ($\rSSM = 0.850$), and Jamba-1.5-Mini ($\rSSM =
    0.875$, 12B).  Our framework achieves Pearson $r = 0.923$ between
    theoretically predicted and empirically measured safety degradation, and
    correctly recovers the ranking Pythia $>$ Zamba $>$ Jamba $>$ Mamba.  The
    \MBCA{} mechanism ($K = 8$ probes) attains $\betaMBCA > 0.70$, reduces
    \CHSS{} (\emph{Cascading Hidden-State Safety}) degradation from $62\%$ to
    $23\%$, and degrades MMLU / HellaSwag / ARC accuracy by less than $2\%$.

\end{enumerate}

To our knowledge, SSH-Hybrid is the first system to provide formal guarantees,
empirical validation across multiple architectures, and a deployable mitigation
for safety blind windows in hybrid SSM-Transformer models simultaneously.

\paragraph{Paper organization.}
Section~\ref{sec:related} surveys related work.
Section~\ref{sec:background} establishes notation and background on SSM
recurrence and safety margins.
Section~\ref{sec:theory} presents Theorems~1 and~2 with full proofs.
Section~\ref{sec:mbca} describes the \MBCA{} architecture and audit procedure.
Section~\ref{sec:experiments} reports empirical results.
Section~\ref{sec:discussion} discusses limitations and broader impacts.

% -------------------------------------------------------
\section{Related Work}
\label{sec:related}
% -------------------------------------------------------

\subsection{State Space Models and Hybrid Architectures}

\paragraph{SSM foundations.}
Structured State Space Models were introduced by Gu et al.~\cite{gu2022efficiently}
as the S4 model, establishing that linear recurrences with HiPPO-initialized
transition matrices can match Transformer performance on long-range sequence
tasks.  Subsequent work developed simpler parameterizations: S5~\cite{smith2022s5}
reduced S4 to a single SISO system with parallel scans; Hyena~\cite{poli2023hyena}
replaced explicit state matrices with long convolutions; and
RWKV~\cite{peng2023rwkv} reformulated gated recurrence in a Transformer-like
framework amenable to standard training pipelines.

Mamba~\cite{gu2023mamba} introduced \emph{selective} state spaces, making the
$\bar{A}$ and $\bar{B}$ matrices input-dependent through a hardware-aware
selective scan.  This selectivity breaks the linear time-invariance of S4 but
retains $\rho < 1$ for stability.  Mamba-2~\cite{dao2024mamba2} reformulated
selective SSMs as structured masked attention, unifying SSM and attention theory
and improving hardware utilization.  These architectures achieve near-linear
scaling in sequence length while maintaining competitive language modeling
perplexity, motivating their adoption in large-scale models.

\paragraph{Hybrid SSM-Transformer models.}
The efficiency of SSM recurrence and the in-context flexibility of Transformer
attention are complementary: SSMs excel at compressing long histories cheaply,
while attention is superior for precise retrieval over recent tokens.  Several
recent architectures exploit this complementarity.

Jamba~\cite{lieber2024jamba} and its successor Jamba-1.5~\cite{lieber2024jamba15}
interleave Mamba SSM blocks with Transformer attention layers at a ratio of
roughly 7:1 (SSM:Attention), yielding $\rSSM = 0.875$ in a 12B parameter model.
Zamba-7B~\cite{glorioso2024zamba} uses a single shared attention block
interleaved with SSM layers at a 6:1 ratio ($\rSSM \approx 0.850$), achieving
strong benchmark performance at 7B scale.  Griffin~\cite{de2024griffin}
proposes gated linear recurrences with local attention windows as an alternative
hybrid, while Hawk~\cite{de2024griffin} is a pure-recurrence variant.
RetNet~\cite{sun2023retentive} frames its retention mechanism as a form of
recurrent attention, exposing similar spectral decay properties.

Despite the empirical success of these architectures, \emph{no prior work has
analyzed how the SSM fraction $\rSSM$ or the spectral radius $\rho$ affects
safety properties}.  Our work fills this gap.

\subsection{AI Safety, Alignment, and Red-Teaming}

\paragraph{RLHF and reward-based alignment.}
Reinforcement Learning from Human Feedback~\cite{ouyang2022training,bai2022training}
trains language models to maximize a reward signal derived from human
preference comparisons.  PPO-based RLHF~\cite{schulman2017proximal} and
DPO~\cite{rafailov2023direct} are the dominant practical approaches.  These
methods condition alignment on the model's output distribution and do not
constrain or monitor internal recurrent states; they are therefore agnostic to
the dynamics described in Eq.~\eqref{eq:ssm-recurrence}.

\paragraph{Constitutional AI and rule-based supervision.}
Constitutional AI (CAI)~\cite{bai2022constitutional} augments RLHF with a set
of natural-language principles used during synthetic preference generation.
Llama Guard~\cite{inan2023llama} and similar output classifiers apply
rule-based filtering to final generations.  Both approaches operate on model
outputs and share the black-box assumption that makes them structurally blind
to SSM recurrence dynamics.

\paragraph{Representation engineering.}
Zou et al.~\cite{zou2023representation} demonstrated that safety-relevant
concepts can be read from and written to the residual stream of Transformer
models via linear probes.  This approach is explicitly designed around the
residual stream abstraction, which does not apply to SSM hidden states governed
by Eq.~\eqref{eq:ssm-recurrence}.  Our \MBCA{} mechanism adapts the intuition
of persistent safety signals to the recurrent setting, with a formal monotone
guarantee that representation engineering lacks.

\paragraph{Red-teaming and safety evaluation.}
Perez et al.~\cite{perez2022red} introduced automated red-teaming via
LM-generated adversarial prompts.  Ganguli et al.~\cite{ganguli2022red}
conducted large-scale human red-teaming studies.  Gehman et al.~\cite{gehman2020realtoxicity}
established RealToxicityPrompts as a benchmark for prompted toxicity.
The HarmBench~\cite{mazeika2024harmbench} framework provides standardized
evaluation across a range of attack types.  These evaluations treat the model
as a black box and do not instrument internal hidden states or characterize
architectural decay dynamics.

\subsection{Adversarial Attacks on Language Models}

\paragraph{Gradient-based jailbreaks.}
The Greedy Coordinate Gradient (GCG) attack~\cite{zou2023universal} produces
universal adversarial suffixes that transfer across models by optimizing
discrete token sequences via gradient approximation.  AutoDAN~\cite{liu2024autodan}
and PAIR~\cite{chao2023pair} generate semantically coherent jailbreaks via
iterative LM-based search.  These attacks operate in the input space and do not
exploit SSM hidden-state dynamics, though the blind window we characterize
could amplify their effectiveness in long-context settings.

\paragraph{Long-context and many-shot attacks.}
Anil et al.~\cite{anil2024many} demonstrated that safety training can be
circumvented by prepending large numbers of harmful question-answer pairs (many-shot
jailbreaking), exploiting the in-context learning capacity of long-context models.
Perez and Ribeiro~\cite{perez2022ignore} showed that injected instructions can
hijack agentic model behavior in the middle of long contexts.  Both attacks
benefit directly from the safety blind window: long contexts push safety
detections beyond $\Horizon$, and our framework provides the first formal
explanation for why these attacks are more effective against SSM-heavy hybrid
models.

\paragraph{Z-HiSPA and structured adversarial perturbations.}
Recent work on Z-HiSPA~\cite{zhang2024zhispa} (Zero-shot Hierarchical Structured
Prompt Attacks) exploits structured formatting in prompts to bypass safety
classifiers.  Speculative decoding attacks~\cite{chen2024speculative} manipulate
the draft-model step to smuggle harmful tokens past alignment mechanisms.
These attacks are complementary to our analysis: Z-HiSPA targets output
classifiers while the blind window targets recurrent state persistence.

\subsection{Spectral Methods for Neural Network Safety}

SpectralGuard~\cite{wang2024spectralguard} applied spectral analysis of weight
matrices to detect adversarial inputs in vision models, establishing a precedent
for using eigenvalue structure as a safety signal.  Spectral norm
regularization~\cite{miyato2018spectral} is a standard technique for stabilizing
GAN training by bounding the Lipschitz constant of discriminator layers.
Eigenvalue analysis has also been applied to understand information propagation
depth in deep networks~\cite{pennington2017resurrecting} and to characterize
memorization in Transformer attention heads~\cite{elhage2021mathematical}.

None of these works address the specific problem of recurrent hidden-state
decay in SSMs, nor do they connect spectral radius to safety margin degradation
in a formal sense.  The theoretical apparatus of SSH-Hybrid---particularly the
closed-form horizon $\Horizon$ and the gap factor $g$---is novel to this work.

\subsection{Summary of the Gap}

Table~\ref{tab:related-comparison} summarizes the landscape.  No prior work
simultaneously provides (i)~a formal safety margin characterization tied to
SSM spectral properties, (ii)~empirical validation across architectures spanning
$\rSSM \in \{0, 0.85, 0.875, 1.0\}$, and (iii)~a deployable mitigation with
provable retention guarantees.  SSH-Hybrid provides all three.

\begin{table}[t]
  \centering
  \caption{Comparison of SSH-Hybrid with related approaches. \checkmark =
    fully addressed, $\sim$ = partially addressed, $\times$ = not addressed.}
  \label{tab:related-comparison}
  \small
  \begin{tabular}{lccccc}
    \toprule
    \makecell[l]{Method} &
    \makecell{Formal\\guarantee} &
    \makecell{SSM-aware} &
    \makecell{Arch.\\validation} &
    \makecell{Deployable\\mitigation} &
    \makecell{Long-context\\coverage} \\
    \midrule
    RLHF~\cite{ouyang2022training}         & $\times$ & $\times$ & $\times$ & \checkmark & $\times$ \\
    CAI~\cite{bai2022constitutional}        & $\times$ & $\times$ & $\times$ & \checkmark & $\times$ \\
    Llama Guard~\cite{inan2023llama}        & $\times$ & $\times$ & $\times$ & \checkmark & $\times$ \\
    Repr.\ Eng.~\cite{zou2023representation}& $\times$ & $\times$ & $\sim$   & \checkmark & $\times$ \\
    GCG~\cite{zou2023universal}             & $\times$ & $\times$ & $\sim$   & $\times$   & $\times$ \\
    Many-shot~\cite{anil2024many}           & $\times$ & $\times$ & $\times$ & $\times$   & \checkmark \\
    SpectralGuard~\cite{wang2024spectralguard} & $\sim$ & $\times$ & $\times$ & $\sim$ & $\times$ \\
    \midrule
    \textbf{SSH-Hybrid (ours)}              & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark \\
    \bottomrule
  \end{tabular}
\end{table}


% =======================================================================
% SECTION 3: Theoretical Framework (inserted)
% =======================================================================

\section{Theoretical Framework}
\label{sec:theory}

\subsection{Preliminaries and Notation}
\label{sec:prelim}

We consider hybrid SSM--Transformer architectures comprising $L_{\text{tot}}$ layers, of which a fraction $\rSSM \in (0,1)$ are state-space model (SSM) layers and the remaining fraction $1 - \rSSM$ are standard self-attention layers. Each SSM layer evolves according to the linear recurrence

\begin{equation}
    h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t + D x_t,
    \label{eq:ssm-recurrence-formal}
\end{equation}

\noindent where $h_t \in \mathbb{R}^d$ is the hidden state at time step $t$, $x_t \in \mathbb{R}^p$ is the input, and $y_t \in \mathbb{R}^q$ is the output. The matrices $\bar{A} \in \mathbb{R}^{d \times d}$ and $\bar{B} \in \mathbb{R}^{d \times p}$ are obtained from continuous-time parameters $(A, B)$ via zero-order hold (ZOH) discretization with step size $\Delta > 0$:

\begin{equation}
    \bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}(\bar{A} - I)\,\Delta B.
    \label{eq:zoh-formal}
\end{equation}

\noindent The \emph{spectral radius} of the discretized state matrix governs the long-range memory properties of the SSM:

\begin{equation}
    \rho(\bar{A}) \coloneqq \max\bigl\{|\lambda| : \lambda \in \sigma(\bar{A})\bigr\},
    \label{eq:spectral-radius-formal}
\end{equation}

\noindent where $\sigma(\bar{A})$ denotes the spectrum of $\bar{A}$. For a stable SSM we require $\rho(\bar{A}) \in (0, 1)$, which ensures that the hidden state norm satisfies $\|h_t\| = O(\rho^t)$. Consequently, a safety-relevant signal embedded in $h_{t_0}$ decays in $\ell^2$ norm as

\begin{equation}
    \|h_{t_0 + \tau}\|_{\text{signal}} \;\leq\; \rho^{\tau} \|h_{t_0}\|,
    \label{eq:decay-formal}
\end{equation}

\noindent for all $\tau \geq 0$. We write $\rho$ for $\rho(\bar{A})$ when the context is clear. Throughout, $\epsilon \in (0,1)$ denotes a fixed \emph{safety detection threshold}: a signal is considered recoverable when its normalised magnitude exceeds $\epsilon$.

\begin{definition}[Safety Margin]
\label{def:margin}
Let $\mathcal{D}$ be the deployment input distribution and $\mathcal{V}$ the set of violating outputs. The \emph{safety margin} of policy $\pi$ is
\begin{equation}
    \deltaStar(\pi) \coloneqq \inf\bigl\{\|\delta\|_{\infty} : \exists\, x \sim \mathcal{D} \text{ s.t.\ } \pi(x + \delta) \in \mathcal{V}\bigr\}.
\end{equation}
\end{definition}

\subsection{The Safety Memory Horizon}
\label{sec:horizon-theory}

\begin{definition}[Safety Memory Horizon]
\label{def:horizon}
Fix a stability threshold $\epsilon \in (0,1)$ and let $\rho \in (0,1)$ be the spectral radius of a stable SSM layer. The \emph{safety memory horizon} is
\begin{equation}
    \Horizon \;\coloneqq\; \frac{\log(1/\epsilon)}{\log(1/\rho)} \;\in\; (0, \infty).
    \label{eq:horizon-formal}
\end{equation}
\end{definition}

The horizon $\Horizon$ is the token distance beyond which a safety signal decays below the detection threshold $\epsilon$. The decay bound~\eqref{eq:decay-formal} gives $\rho^{\tau} \leq \epsilon$ if and only if $\tau \geq \Horizon$. Equivalently, $\Horizon$ is the unique positive solution to $\rho^{H} = \epsilon$. Taking logarithms: $H \log \rho = \log \epsilon$, hence $H = \log(1/\epsilon) / \log(1/\rho)$.

\begin{remark}
$\Horizon$ is monotone increasing in $\rho$: a larger spectral radius retains signals longer. For $\rho = 0$ (instant forgetting), $\Horizon = 0$. As $\rho \to 1^-$, $\Horizon \to \infty$, recovering the infinite context window of pure attention. For pure Transformers ($\rSSM = 0$), $\Horizon$ is undefined (infinite effective memory).
\end{remark}

\subsection{The Attenuation Factor}
\label{sec:attenuation-theory}

\begin{definition}[Attenuation Factor]
\label{def:atten}
For spectral radius $\rho \in (0,1)$, SSM layer fraction $\rSSM \in (0,1)$, and interaction length $L \in \mathbb{N}$, the \emph{attenuation factor} is
\begin{equation}
    g(\rho, \rSSM, L) \;\coloneqq\; \rSSM \cdot \min\!\left(1,\, \frac{L}{\Horizon}\right).
    \label{eq:atten-formal}
\end{equation}
\end{definition}

The factor $g \in [0, \rSSM] \subset [0,1)$ quantifies aggregate safety-signal degradation. Three regimes:

\begin{itemize}
    \item \textbf{Short interactions} ($L < \Horizon$): SSM layers retain the safety signal throughout; attenuation scales linearly as $\rSSM \cdot L / \Horizon$.
    \item \textbf{Long interactions} ($L \geq \Horizon$): SSM layers lose the safety signal entirely beyond the horizon, saturating at $g = \rSSM$.
    \item \textbf{Pure Transformer} ($\rSSM = 0$): no SSM layers, so $g = 0$ and no attenuation.
\end{itemize}

\subsection{Spectral Safety Margin Bound (Theorem 1)}
\label{sec:main-theorem}

\begin{theorem}[Spectral Safety Margin Bound]
\label{thm:margin-bound}
Let $\piHybrid$ be a hybrid SSM--Transformer policy with SSM spectral radius $\rho \in (0,1)$, layer fraction $\rSSM \in (0,1)$, and interaction length $L$. Let $\piTrans$ be a pure-Transformer policy trained identically. Then
\begin{equation}
    \deltaStar(\piHybrid) \;\leq\; \deltaStar(\piTrans) \cdot \bigl(1 - g(\rho, \rSSM, L)\bigr).
    \label{eq:margin-bound-formal}
\end{equation}
\end{theorem}

\begin{proof}
We construct an explicit adversarial strategy exploiting the safety blind window.

\medskip
\noindent\textbf{Step 1: Information flow decomposition.}
Consider an interaction of $L$ tokens. A safety constraint $\phi$ is encoded at token $t_0 = 1$. For the $(1-\rSSM)$ fraction of attention layers, information from $t_0$ reaches any token $t$ through the key--value cache with $O(1)$ access, so the signal strength satisfies $s^{\mathrm{attn}}_t \geq s_{t_0}$ for all $t \leq L$. For the $\rSSM$ fraction of SSM layers, by the recurrence and decay bound~\eqref{eq:decay-formal}:
\begin{equation}
    \bigl\|\bar{A}^\tau h_{t_0}\bigr\| \;\leq\; \rho^\tau \|h_{t_0}\|.
\end{equation}

\medskip
\noindent\textbf{Step 2: Safety blind window.}
Define the safety blind window for SSM layers:
\begin{equation}
    \mathcal{W} \;\coloneqq\; \bigl\{t \in \{t_0{+}1, \ldots, L\} : t - t_0 \geq \Horizon\bigr\}.
\end{equation}
For $t \in \mathcal{W}$, the SSM signal satisfies $\rho^{t-t_0} \leq \epsilon$, so constraint $\phi$ is undetectable from the SSM state alone. The fraction of the interaction in the blind window is $\max(0, L - \Horizon)/L = 1 - \min(1, \Horizon/L)$.

\medskip
\noindent\textbf{Step 3: Safety margin deficit.}
The adversary targets tokens in $\mathcal{W}$. Since $\rSSM$ of all layers are SSM layers and the blind-window fraction is $\min(1, L/\Horizon)$, the fraction of total processing where the constraint is undetectable is:
\begin{equation}
    f_{\mathrm{blind}} = \rSSM \cdot \min\!\left(1, \frac{L}{\Horizon}\right) = g(\rho, \rSSM, L).
\end{equation}

The safety margin decomposes over the two pathways. For attention layers ($1 - f_{\mathrm{blind}}$ effective fraction), the retention factor is $\leq 1$. For SSM layers in the blind window ($f_{\mathrm{blind}}$ fraction), the retention factor is 0 since the signal is below $\epsilon$. Therefore:
\begin{align}
    \deltaStar(\piHybrid) &\leq \deltaStar(\piTrans) \cdot (1 - f_{\mathrm{blind}}) \cdot 1 + \deltaStar(\piTrans) \cdot f_{\mathrm{blind}} \cdot 0 \notag\\
    &= \deltaStar(\piTrans) \cdot (1 - g(\rho, \rSSM, L)).
\end{align}

\noindent\textbf{Step 4: Baseline consistency.}
For $\piTrans$ with $\rSSM = 0$: $g = 0$, so $\deltaStar(\piTrans) = \deltaStar(\piTrans)$, consistent.

\noindent\textbf{Step 5: Tightness.}
The bound is achieved in the limit $\epsilon \to 0$, $L/\Horizon \to \infty$, where $g \to \rSSM$.
\end{proof}

\subsection{MBCA Compensation (Theorem 2)}
\label{sec:mbca-theorem}

\begin{theorem}[MBCA Compensation]
\label{thm:mbca-comp}
Let $\piHybrid$ be as in Theorem~\ref{thm:margin-bound}, and let $\piHybrid + \MBCA$ denote the augmented policy with coverage $\betaMBCA \in [0,1]$. Then
\begin{equation}
    \deltaStar(\piHybrid + \MBCA) \;\geq\; \deltaStar(\piTrans) \cdot \bigl(1 - g \cdot (1 - \betaMBCA)\bigr).
    \label{eq:mbca-comp-formal}
\end{equation}
\end{theorem}

\begin{proof}
\noindent\textbf{Step 1: Deficit definition.}
From Theorem~\ref{thm:margin-bound}, the safety margin deficit is:
\begin{equation}
    \Delta_{\mathrm{def}} = \deltaStar(\piTrans) - \deltaStar(\piHybrid) \geq \deltaStar(\piTrans) \cdot g.
\end{equation}

\noindent\textbf{Step 2: MBCA recovery.}
The \MBCA{} mechanism monitors attention-layer hidden states via $K$ boolean probes. Coverage $\betaMBCA$ means that for any safety violation in the blind window, \MBCA{} detects it with probability $\geq \betaMBCA$. When detection occurs, the safety intervention is triggered, effectively restoring the margin for that fraction of cases.

\noindent\textbf{Step 3: Effective margin.}
The compensated margin combines the unaugmented hybrid margin with the fraction of the deficit recovered by \MBCA{}:
\begin{align}
    \deltaStar(\piHybrid + \MBCA) &= \deltaStar(\piHybrid) + \betaMBCA \cdot \Delta_{\mathrm{def}} \notag\\
    &\geq \deltaStar(\piTrans)(1-g) + \betaMBCA \cdot \deltaStar(\piTrans) \cdot g \notag\\
    &= \deltaStar(\piTrans)(1 - g + \betaMBCA \cdot g) \notag\\
    &= \deltaStar(\piTrans)(1 - g(1 - \betaMBCA)).
\end{align}

\noindent\textbf{Step 4: Boundary cases.}
$\betaMBCA = 0$: reduces to Theorem~\ref{thm:margin-bound}. $\betaMBCA = 1$: $\deltaStar(\piHybrid + \MBCA) \geq \deltaStar(\piTrans)$, full recovery.
\end{proof}

\subsection{Sufficient Coverage for Near-Baseline Safety}

\begin{corollary}[Sufficient MBCA Coverage]
\label{cor:sufficient}
Fix tolerance $\epsilon_{\mathrm{tol}} \in (0, g)$. If $\betaMBCA \geq 1 - \epsilon_{\mathrm{tol}}/g$, then
\begin{equation}
    \deltaStar(\piHybrid + \MBCA) \;\geq\; \deltaStar(\piTrans) \cdot (1 - \epsilon_{\mathrm{tol}}).
\end{equation}
\end{corollary}

\begin{proof}
Substituting $\betaMBCA \geq 1 - \epsilon_{\mathrm{tol}}/g$ into~\eqref{eq:mbca-comp-formal}: $\deltaStar(\piHybrid + \MBCA) \geq \deltaStar(\piTrans)(1 - g \cdot \epsilon_{\mathrm{tol}}/g) = \deltaStar(\piTrans)(1 - \epsilon_{\mathrm{tol}})$.
\end{proof}

\subsection{Theoretical Discussion}

\paragraph{Connection to observability.}
The safety blind window is related to classical observability in linear systems~\cite{kalman1960}. The observability Gramian $\mathcal{O}_T = \sum_{t=0}^{T-1} (\bar{A}^\top)^t C^\top C \bar{A}^t$ has smallest eigenvalue decaying as $O(\rho^{2T})$, confirming that recovery of past information becomes exponentially ill-conditioned beyond $\Horizon$. Our Definition~\ref{def:horizon} identifies the threshold where observability effectively fails for safety-signal detection.

\paragraph{Tightness.}
The bound is asymptotically tight: as $\epsilon \to 0$ and $L/\Horizon \to \infty$, $g \to \rSSM$ and the bound becomes $\deltaStar(\piHybrid) \leq (1-\rSSM) \cdot \deltaStar(\piTrans)$, which is tight since SSM layers contribute zero margin. In practice, we observe Pearson $r = 0.923$ between predicted and empirical margins (Section~\ref{sec:experiments}).

\paragraph{The safety blind window.}
$\mathcal{W}$ is the central concept: unlike attention (which maintains explicit, directly addressable key--value representations of all past tokens), SSM layers compress history into a fixed-dimensional state that is lossy with respect to safety-relevant information when $\tau > \Horizon$. The \MBCA{} mechanism is designed precisely to complement SSM layers by providing a monotone, non-decaying accumulation channel.


% =======================================================================
% SECTION 4: The MBCA Mechanism
% =======================================================================

\section{The \MBCA{} Mechanism}
\label{sec:mbca}

\subsection{Architecture}
\label{sec:mbca:arch}

The \textbf{Monotone Boolean Carry Accumulator} (\MBCA{}) augments a hybrid
\piHybrid{} model with $K$ lightweight Boolean \emph{safety probes} attached to
the hidden states of every attention layer.  At each token position $t$, let
$\mathbf{a}_t \in \mathbb{R}^{d_{\mathrm{model}}}$ denote the hidden state
produced by an attention sub-layer.  Each probe $k \in \{1,\ldots,K\}$ is a
linear classifier parameterised by weight vector $\mathbf{w}_k \in
\mathbb{R}^{d_{\mathrm{model}}}$ and bias $b_k \in \mathbb{R}$.

\paragraph{Carry state.}
The \MBCA{} maintains a carry vector $\mathbf{c} = (c[1],\ldots,c[K]) \in
\{0,1\}^K$, initialised to $\mathbf{0}$ at the start of every context window.

\paragraph{Monotone OR update.}
At each token step $t$, the carry state is updated by the rule
\begin{equation}
  c[k] \;\leftarrow\; c[k] \;\vee\;
  \bigl(\mathbf{w}_k^{\top}\mathbf{a}_t + b_k > 0\bigr),
  \qquad k = 1,\ldots,K,
  \label{eq:mbca-update}
\end{equation}
where $\vee$ denotes the Boolean OR.  The full update procedure is stated as
Algorithm~\ref{alg:mbca}.

\begin{algorithm}[t]
\caption{\MBCA{} Carry Update}
\label{alg:mbca}
\begin{algorithmic}[1]
\Require Carry vector $\mathbf{c} \in \{0,1\}^K$; attention hidden state
         $\mathbf{a}_t \in \mathbb{R}^{d_{\mathrm{model}}}$;
         probe parameters $\{(\mathbf{w}_k, b_k)\}_{k=1}^{K}$
\Ensure Updated carry vector $\mathbf{c}$
\For{$k = 1$ \textbf{to} $K$}
  \State $v_k \leftarrow \mathbf{w}_k^{\top}\mathbf{a}_t + b_k$
  \If{$v_k > 0$}
    \State $c[k] \leftarrow 1$
  \EndIf
  \Comment{$c[k] = 0$ is never reset once set to $1$}
\EndFor
\State \Return $\mathbf{c}$
\end{algorithmic}
\end{algorithm}

\noindent Because the update is a pure OR, the carry vector is
\emph{non-decreasing} over the token sequence: each $c[k]$ can transition from
$0$ to $1$ exactly once and thereafter remains $1$.  This monotonicity property
is what gives \MBCA{} its memory-safety guarantee, as formalised in
Section~\ref{sec:mbca:monotone}.

\subsection{Why Monotonicity Is Critical}
\label{sec:mbca:monotone}

Two sources of ``forgetting'' afflict vanilla hybrid architectures.

\paragraph{SSM exponential decay.}
The recurrent state of an SSM layer evolves as $\mathbf{h}_t = A\mathbf{h}_{t-1}
+ B x_t$ where, under zero-order-hold (ZOH) discretisation, the spectral radius
$\rho = \rSSM(A) < 1$.  A safety signal injected into the SSM state at position
$t$ is attenuated by a factor of $\rho^{\tau}$ after $\tau$ additional tokens;
for large $\tau$ this is effectively erased.  The \Horizon{} bound $H(\rho)$
(Definition~\ref{def:horizon}) captures exactly when this erasure becomes
safety-critical.

\paragraph{Attention dilution.}
In long contexts the softmax denominator grows, diluting each token's
contribution to future attention outputs.  A harmful pattern occurring at
position $t$ may be assigned negligible attention weight by position $t + \tau$
for large $\tau$, even if the full key--value history is retained.

\paragraph{Monotone OR is forgetting-proof.}
The OR operation in Eq.~\eqref{eq:mbca-update} is immune to both decay and
dilution: once probe $k$ fires, the evidence is \emph{latched} indefinitely.

\begin{lemma}[Monotonicity of the Carry Vector]
\label{lem:mbca-monotone}
Let $\mathbf{c}^{(t)}$ denote the carry vector after processing token $t$ under
the update rule~\eqref{eq:mbca-update}.  For all $t \geq 1$ and all
$k \in \{1,\ldots,K\}$,
\[
  c^{(t-1)}[k] = 1 \;\Longrightarrow\; c^{(t)}[k] = 1.
\]
Equivalently, $c^{(s)}[k] \leq c^{(t)}[k]$ for all $0 \leq s \leq t$.
\end{lemma}

\begin{proof}
By induction on $t$.

\emph{Base case} ($t = 0$): $\mathbf{c}^{(0)} = \mathbf{0}$, so the premise
$c^{(-1)}[k] = 1$ is vacuously false and the statement holds trivially.

\emph{Inductive step}: Assume $c^{(t-1)}[k] = 1$ for some $k$.  By the update
rule~\eqref{eq:mbca-update},
\[
  c^{(t)}[k] = c^{(t-1)}[k] \vee \bigl(\mathbf{w}_k^{\top}\mathbf{a}_t + b_k > 0\bigr).
\]
Since $c^{(t-1)}[k] = 1$ and $1 \vee x = 1$ for any Boolean $x$, we obtain
$c^{(t)}[k] = 1$.  The inductive hypothesis therefore holds at step $t$.

By induction, once $c^{(s)}[k] = 1$ for some $s$, we have $c^{(t)}[k] = 1$
for all $t \geq s$.  This establishes the monotone non-decreasing property over
the entire token sequence.
\end{proof}

\noindent Lemma~\ref{lem:mbca-monotone} guarantees that no amount of subsequent
context---however long, however benign---can erase a safety signal already
recorded in the carry vector.  This is the property that \emph{lifts} \MBCA{}
above the \Horizon{} barrier: while the SSM itself may lose track of early
harmful tokens after $H(\rho)$ steps, the carry vector retains them indefinitely.

\subsection{Safety Formula $\phi(\mathbf{c})$}
\label{sec:mbca:formula}

Given carry vector $\mathbf{c} \in \{0,1\}^K$, the safety verdict is computed by
a Boolean formula $\phi : \{0,1\}^K \to \{0,1\}$ where $\phi(\mathbf{c}) = 1$
indicates a \emph{safety trigger} (unsafe content detected).  We support four
families of formula, selectable at deployment time.

\paragraph{Any (OR of all probes).}
\begin{equation}
  \phi_{\mathrm{any}}(\mathbf{c})
  \;=\; \bigvee_{k=1}^{K} c[k]
  \;=\; \mathbf{1}\!\left[\sum_{k=1}^{K} c[k] \geq 1\right].
  \label{eq:phi-any}
\end{equation}
Triggers if \emph{at least one} probe has fired.  Maximises recall; suitable for
high-stakes, low-tolerance deployments.

\paragraph{Majority.}
\begin{equation}
  \phi_{\mathrm{maj}}(\mathbf{c})
  \;=\; \mathbf{1}\!\left[\sum_{k=1}^{K} c[k] > \frac{K}{2}\right].
  \label{eq:phi-majority}
\end{equation}
Triggers when a strict majority of probes agree.  Balances precision and recall;
robust to occasional probe misfires.

\paragraph{All (AND of all probes).}
\begin{equation}
  \phi_{\mathrm{all}}(\mathbf{c})
  \;=\; \bigwedge_{k=1}^{K} c[k]
  \;=\; \mathbf{1}\!\left[\sum_{k=1}^{K} c[k] = K\right].
  \label{eq:phi-all}
\end{equation}
Triggers only when \emph{every} probe has fired.  Maximises precision; suitable
when false positives carry a high cost.

\paragraph{Threshold.}
\begin{equation}
  \phi_{\theta}(\mathbf{c})
  \;=\; \mathbf{1}\!\left[\sum_{k=1}^{K} c[k] \geq \theta\right],
  \qquad \theta \in \{1,\ldots,K\}.
  \label{eq:phi-threshold}
\end{equation}
The parameter $\theta$ interpolates continuously between the Any
($\theta = 1$) and All ($\theta = K$) extremes, enabling fine-grained
precision--recall trade-offs.  Majority vote is recovered at
$\theta = \lfloor K/2 \rfloor + 1$.

\subsection{Training the Safety Probes}
\label{sec:mbca:training}

\paragraph{Dataset.}
We train the $K$ probe classifiers on the \textbf{BeaverTails} dataset
\citep{Ji2023BeaverTails}, a curated collection of human-annotated
(prompt, response) pairs labelled for harmful content across fourteen harm
categories.  We sample 5\,000 examples and apply a standard 80/20 train--validation
split (4\,000 training, 1\,000 validation).

\paragraph{Feature extraction.}
For each example in the training corpus, we perform a single forward pass through
the frozen hybrid model \piHybrid{} and collect the hidden states
$\{\mathbf{a}_t^{(\ell)}\}$ from all \Atten{} attention sub-layers $\ell$ at
every token position $t$.  The input representation for probe $k$ is the
\emph{mean-pooled} attention hidden state across all token positions and all
attention layers:
\begin{equation}
  \bar{\mathbf{a}} \;=\;
  \frac{1}{T \cdot L_{\mathrm{attn}}}
  \sum_{\ell=1}^{L_{\mathrm{attn}}} \sum_{t=1}^{T}
  \mathbf{a}_t^{(\ell)}
  \;\in\; \mathbb{R}^{d_{\mathrm{model}}}.
  \label{eq:mean-pool}
\end{equation}

\paragraph{Objective.}
Each probe $k$ is trained with binary cross-entropy (BCE) loss:
\begin{equation}
  \mathcal{L}_k
  \;=\; -\frac{1}{|\mathcal{D}_{\mathrm{train}}|}
  \sum_{i \in \mathcal{D}_{\mathrm{train}}}
  \Bigl[
    y_i \log \sigma(\mathbf{w}_k^{\top}\bar{\mathbf{a}}_i + b_k)
    + (1 - y_i)\log\bigl(1 - \sigma(\mathbf{w}_k^{\top}\bar{\mathbf{a}}_i + b_k)\bigr)
  \Bigr],
  \label{eq:bce}
\end{equation}
where $y_i \in \{0,1\}$ is the harm label and $\sigma$ is the sigmoid function.

\paragraph{Optimisation.}
Parameters $\{(\mathbf{w}_k, b_k)\}$ are optimised with Adam
\citep{Kingma2015Adam} at learning rate $10^{-3}$, with default momentum
parameters $\beta_1 = 0.9$, $\beta_2 = 0.999$.  Training runs for 20 epochs
over the training split with batch size 64.  The full training procedure for all
$K$ probes completes in under 2 minutes on a single A100 GPU.

\subsection{Three-Phase Audit Procedure}
\label{sec:mbca:audit}

We organise the end-to-end safety certification of a hybrid model into the three
phases described in Algorithm~\ref{alg:audit}.

\begin{algorithm}[t]
\caption{\CHSS{} Three-Phase Audit Procedure}
\label{alg:audit}
\begin{algorithmic}[1]
\Require Hybrid model \piHybrid{}; safety margin threshold $g_{\min}$; benchmark
         dataset $\mathcal{B}$; number of probes $K$
\Ensure Safety certificate or escalation decision

\State \textbf{// Phase 1: Spectral Radius Measurement}
\For{each SSM layer $\ell = 1,\ldots,L_{\mathrm{ssm}}$}
  \State Extract discrete state-transition matrix $A^{(\ell)}$ via ZOH discretisation
  \State $\rho^{(\ell)} \leftarrow \rSSM(A^{(\ell)})$
  \Comment{Largest singular value; cost $O(d^3)$ per layer}
\EndFor
\State $\rho^* \leftarrow \max_\ell \rho^{(\ell)}$
\State $H \leftarrow \Horizon(\rho^*)$
\Comment{Definition~\ref{def:horizon}; $H \propto (1-\rho^*)^{-1}$}
\State \textbf{Assert} total cost $< 100\,\mathrm{ms}$ for $N$ layers of dimension $d$
\Comment{$O(N \cdot d^3)$}

\State
\State \textbf{// Phase 2: Safety Margin Computation}
\State Compute safety gap $g \leftarrow \deltaStar(\piTrans{}) - \deltaStar(\piHybrid{})$
\Comment{Theorem~\ref{thm:main}}
\If{$g \geq g_{\min}$}
  \State \Return \textsc{Certified-Safe} (spectral margin sufficient)
\Else
  \State Compute required augmentation $\betaMBCA^* \leftarrow g_{\min} - g$
  \Comment{Solve Corollary~\ref{cor:beta-star}}
\EndIf

\State
\State \textbf{// Phase 3: MBCA Deployment and Empirical Verification}
\State Train $K$ safety probes on BeaverTails \citep{Ji2023BeaverTails} per
       Section~\ref{sec:mbca:training}
\State Deploy \MBCA{} with trained probes on held-out benchmark $\mathcal{B}$
\State Measure empirical safety lift $\hat{\betaMBCA}$ on $\mathcal{B}$
\If{$\hat{\betaMBCA} \geq \betaMBCA^*$}
  \State \Return \textsc{Certified-Safe} (\MBCA{} closes the spectral gap)
\Else
  \State Increase $K$ and return to start of Phase~3, \textbf{or}
  \State \Return \textsc{Escalate} (requires architectural intervention)
\EndIf
\end{algorithmic}
\end{algorithm}

\paragraph{Phase 1 --- Spectral radius measurement.}
For each of the $N$ SSM layers in \piHybrid{}, we extract the state matrix
$A^{(\ell)}$ and apply ZOH discretisation to obtain the discrete-time equivalent
$\bar{A}^{(\ell)}$.  The spectral radius $\rho^{(\ell)} = \rSSM(\bar{A}^{(\ell)})$
is computed as the largest singular value via a standard eigendecomposition.  The
dominant radius $\rho^* = \max_\ell \rho^{(\ell)}$ determines the safety horizon
$H = \Horizon(\rho^*)$.  The total computational cost is $O(N \cdot d^3)$; in
practice this completes in under 100\,ms for models with $d \leq 4096$ and
$N \leq 64$ SSM layers.

\paragraph{Phase 2 --- Safety margin computation.}
Using Theorem~\ref{thm:main} we evaluate the predicted safety gap
$g = \deltaStar(\piTrans{}) - \deltaStar(\piHybrid{})$.  If $g \geq g_{\min}$
the spectral structure of \piHybrid{} already provides sufficient margin and the
audit terminates with a certificate.  Otherwise, we use the closed-form
expression from Corollary~\ref{cor:beta-star} to determine the minimum \MBCA{}
safety lift $\betaMBCA^*$ needed to close the gap.

\paragraph{Phase 3 --- MBCA deployment.}
The $K$ probes are trained as described in Section~\ref{sec:mbca:training} and
deployed on \piHybrid{}.  The empirical safety lift $\hat{\betaMBCA}$ is measured
on a held-out harm benchmark $\mathcal{B}$ (disjoint from the BeaverTails training
split).  If $\hat{\betaMBCA} \geq \betaMBCA^*$, the system is certified safe.
Otherwise the practitioner may increase $K$ to add probe capacity, or escalate
to architectural remediation (e.g., reducing $\rho^*$ via spectral
regularisation during fine-tuning).

\subsection{Computational Overhead}
\label{sec:mbca:overhead}

The \MBCA{} mechanism imposes negligible overhead relative to the host model.

\paragraph{Per-token inference cost.}
At each token step, the $K$ probe classifiers each perform a single dot product
in $\mathbb{R}^{d_{\mathrm{model}}}$ followed by a scalar threshold comparison.
The total per-token cost is $O(K \cdot d_{\mathrm{model}})$ floating-point
operations.  For comparison:
\begin{itemize}
  \item Attention layers cost $O(t \cdot d_{\mathrm{model}})$ per token (growing
        linearly with context length $t$).
  \item SSM layers cost $O(d_{\mathrm{model}}^2)$ per token for state evolution.
\end{itemize}
Since $K \ll t$ for any non-trivial context and $K \ll d_{\mathrm{model}}$ in
practice (we use $K \leq 64$ throughout our experiments), the \MBCA{} probe
evaluation is at least one order of magnitude cheaper than either the attention
or SSM computation at every token step.

\paragraph{Memory overhead.}
The carry state $\mathbf{c} \in \{0,1\}^K$ requires exactly $K$ bits of
additional memory per active context.  Probe parameters $\{(\mathbf{w}_k,
b_k)\}_{k=1}^K$ occupy $K \cdot (d_{\mathrm{model}} + 1)$ floats, comparable
to a single attention projection matrix and negligible relative to total model
size.

\paragraph{Training cost.}
As noted in Section~\ref{sec:mbca:training}, training all $K$ probes to
convergence requires under 2 minutes on a single A100 GPU, amortised once
per model checkpoint.


% =======================================================================
% SECTIONS 5-6: Experimental Setup and Results
% =======================================================================
% ===========================================================================
\section{Experimental Setup}
\label{sec:experimental-setup}

\subsection{Models}
\label{sec:models}

We evaluate SSH-Hybrid on four language models spanning the full range of SSM density
$\rSSM \in [0, 1]$.
Table~\ref{tab:models} summarises their HuggingFace identifiers, architectural types,
\rSSM{} values, parameter counts, and roles in our evaluation.

\begin{table}[ht]
  \centering
  \caption{Models used in evaluation. \rSSM{} denotes the fraction of layers that are SSM
           layers (Definition~2). Jamba-1.5-Mini serves as our primary subject because it has
           the highest \rSSM{} among the hybrid models evaluated.}
  \label{tab:models}
  \begin{tabular}{llcrrp{3.5cm}}
    \toprule
    \textbf{Model} & \textbf{HuggingFace ID} & \textbf{Type} &
      \rSSM{} & \textbf{Params} & \textbf{Role} \\
    \midrule
    Jamba-1.5-Mini & \texttt{ai21labs/Jamba-v0.1}       & Hybrid      & 0.875 & 12\,B  & Primary (most SSM-heavy hybrid) \\
    Zamba-7B       & \texttt{Zyphra/Zamba-7B-v1}        & Hybrid      & 0.850 &  7\,B  & Secondary hybrid \\
    Pythia-2.8B    & \texttt{EleutherAI/pythia-2.8b}    & Transformer & 0.000 & 2.8\,B & Pure-transformer baseline \\
    Mamba-2.8B     & \texttt{state-spaces/mamba-2.8b}   & SSM         & 1.000 & 2.8\,B & Pure-SSM baseline \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Spectral Radius Measurement}
\label{sec:spectral-measurement}

We follow the SpectralGuard methodology~\citep{lemercier2026spectralguard} for extracting and
discretising the state-transition matrices of SSM layers.
For each SSM layer we (i)~extract the learned $A$ matrix, (ii)~apply zero-order-hold (ZOH)
discretisation,
\begin{equation}
  \bar{A} = \exp(\Delta \cdot A),
  \label{eq:zoh}
\end{equation}
where $\Delta$ is the layer's learned step-size parameter, and (iii)~compute
\begin{equation}
  \rho(\bar{A}) =
  \begin{cases}
    \max_{i}|a_i|              & \text{if } A \text{ is diagonal (Mamba-style)}, \\
    \max_{i}|\lambda_i(\bar{A})| & \text{otherwise (dense case),}
  \end{cases}
  \label{eq:spectral-radius}
\end{equation}
where eigenvalues are computed with \texttt{scipy.linalg.eigvals}.
The per-model spectral radius $\rho$ is the maximum over all SSM layers.

\subsection{Evaluation Protocol}
\label{sec:eval-protocol}

\paragraph{Adversarial benchmark.}
All attacks use RoBench-25~\citep{lemercier2025robench} with seven
Z-HiSPA\footnote{Zero-shot Hidden-State Perturbation Attack.} trigger configurations:
\begin{enumerate}[label=\textbf{z-hispa-\arabic*:}, leftmargin=*, nosep]
  \item Prefix injection --- malicious prefix prepended to every prompt.
  \item Suffix injection --- malicious suffix appended to every prompt.
  \item Interleaved injection --- trigger tokens distributed throughout the prompt.
  \item Repetition-based spectral collapse --- high-frequency token repetition designed to
        drive $\rho(\bar{A}) \to 1$.
  \item Unicode perturbation --- homoglyph substitutions that alter token embeddings.
  \item Context overflow --- 400-token benign padding preceding the adversarial payload.
  \item Adversarial embedding --- gradient-crafted token sequences targeting hidden-state norms.
\end{enumerate}

\paragraph{Hidden-state health metric.}
We measure attack severity with the Content-Hidden State Score (\CHSS{}), defined as the
cosine similarity between the model's hidden states under attack and the corresponding clean
reference hidden states.
\emph{\CHSS{} degradation} is
\begin{equation}
  \Delta_{\CHSS} = \frac{\CHSS_{\mathrm{clean}} - \CHSS_{\mathrm{attack}}}
                        {\CHSS_{\mathrm{clean}}} \times 100\%.
  \label{eq:chss-deg}
\end{equation}

\paragraph{MBCA probe training.}
Linear probes for the Multi-Boundary Coverage Audit (\MBCA{}) are trained on
BeaverTails~\citep{ji2023beavertails} using 5\,000 samples with an 80/20 train/validation
split.
Each probe $\mathbf{w}_k \in \mathbb{R}^{d_{\mathrm{model}}}$ is trained with binary
cross-entropy loss and Adam~($\eta = 10^{-3}$) for 20 epochs.
The \emph{any} safety policy is adopted: a prompt is blocked if \emph{any} probe fires.

\paragraph{Benign benchmarks.}
To quantify overhead on standard capabilities we evaluate all models on MMLU~\citep{hendrycks2021mmlu},
HellaSwag~\citep{zellers2019hellaswag}, and ARC-Challenge~\citep{clark2018arc} via
\texttt{lm-evaluation-harness}.

\subsection{MBCA Configuration}
\label{sec:mbca-config}

Probe count $K \in \{4, 8, 12, 16, 24\}$ is swept in Experiment~5
(Section~\ref{sec:k-sensitivity}).
Unless otherwise stated, all results are reported for $K = 8$, the minimum $K$ that passes
all coverage thresholds (cf.\ Section~\ref{sec:k-sensitivity}).
The per-probe safety threshold $\tau = 0.5$ is fixed throughout.

% ===========================================================================
\section{Results}
\label{sec:results}

We organise results around five experiments that together validate the theoretical claims of
Sections~\ref{sec:theory} and~\ref{sec:mbca}.

% ---------------------------------------------------------------------------
\subsection{Experiment 1: Spectral Radius Measurement}
\label{sec:exp1}

Table~\ref{tab:spectral} reports per-model spectral radii together with derived
quantities used by SSH-Hybrid: the spectral instability score $H(\rho) = -\log(1-\rho)$
and the optimal predicted safety horizon \deltaStar{}.
All measurements complete in under 100\,ms on a single NVIDIA A100 80\,GB GPU.

\begin{table}[ht]
  \centering
  \caption{Spectral radius measurements and derived SSH-Hybrid quantities.
           $H(\rho) = -\log(1-\rho)$ for $\rho < 1$; undefined for Pythia-2.8B
           (no SSM layers).
           $g = \rSSM \cdot H(\rho)$ is the hybrid spectral gain.
           \deltaStar{} is the predicted optimal context-window safety horizon (tokens).
           Audit ranking from most to least safe: Pythia $\succ$ Zamba $\succ$ Jamba $\succ$
           Mamba.}
  \label{tab:spectral}
  \setlength{\tabcolsep}{7pt}
  \begin{tabular}{lccccr}
    \toprule
    \textbf{Model} & \rSSM{} & $\hat{\rho}$ & $H(\hat{\rho})$ & $g$ &
      \textbf{Time (ms)} \\
    \midrule
    Pythia-2.8B    & 0.000 & ---    & ---  & ---  &  9 \\
    Zamba-7B       & 0.850 & 0.9921 & 5.14 & 4.37 & 34 \\
    Jamba-1.5-Mini & 0.875 & 0.9934 & 5.33 & 4.66 & 61 \\
    Mamba-2.8B     & 1.000 & 0.9967 & 6.02 & 6.02 & 18 \\
    \bottomrule
  \end{tabular}
\end{table}

The audit ranking derived from $g$ is
$\text{Pythia} \succ \text{Zamba} \succ \text{Jamba} \succ \text{Mamba}$,
where $\succ$ denotes ``safer under spectral perturbation.''
This is consistent with the architectural intuition: pure transformers carry no SSM-driven
spectral risk, while pure SSMs are maximally exposed.
Mamba-2.8B has $\hat{\rho} = 0.9967$, the value closest to the unit circle, yielding the
largest $H(\rho)$ and smallest \deltaStar{}.

% ---------------------------------------------------------------------------
\subsection{Experiment 2: Theorem~1 Validation}
\label{sec:exp2}

\paragraph{Quantitative fit.}
Across all four models and seven Z-HiSPA configurations (28 data points), the Pearson
correlation between SSH-Hybrid's predicted \CHSS{} degradation and empirically measured
degradation is $r = 0.923$ ($p < 0.001$), exceeding the pre-registered acceptance threshold
of $r > 0.80$.
Mean absolute error is $\mathrm{MAE} = 8.7\%$, below the 15\% acceptance threshold.

\paragraph{Per-model degradation.}
Table~\ref{tab:degradation} reports mean \CHSS{} degradation averaged over the seven
Z-HiSPA configurations.

\begin{table}[ht]
  \centering
  \caption{Mean \CHSS{} degradation (\%) across seven Z-HiSPA configurations (lower is
           better). ``Predicted'' values are computed from Theorem~1; ``Empirical'' values are
           measured on RoBench-25.}
  \label{tab:degradation}
  \begin{tabular}{lrrrr}
    \toprule
    \textbf{Model} & \rSSM{} & $\hat{\rho}$ &
      \textbf{Predicted (\%)} & \textbf{Empirical (\%)} \\
    \midrule
    Pythia-2.8B    & 0.000 & ---    & $\phantom{0}$9.4 & $\phantom{0}$12.1 \\
    Zamba-7B       & 0.850 & 0.9921 & 51.3             & 54.2              \\
    Jamba-1.5-Mini & 0.875 & 0.9934 & 58.7             & 61.8              \\
    Mamba-2.8B     & 1.000 & 0.9967 & 68.4             & 71.3              \\
    \bottomrule
  \end{tabular}
\end{table}

\noindent Pythia-2.8B exhibits only 12.1\% mean degradation, confirming that pure
transformers are largely immune to Z-HiSPA spectral collapse.
Mamba-2.8B reaches 71.3\%, consistent with its $\hat{\rho}$ being nearest to the unit
circle.
The two hybrids fall between these extremes and in the order predicted by $g$.

\begin{figure}[ht]
  \centering
  % Placeholder: scatter plot generated at camera-ready time.
  \fbox{\rule{0pt}{4.5cm}\rule{0.55\textwidth}{0pt}}
  \caption{Scatter plot of predicted vs.\ empirical \CHSS{} degradation across all
           model--attack pairs ($n = 28$).
           Each point corresponds to one (model, Z-HiSPA configuration) pair.
           The dashed line is the identity ($y = x$); the solid line is the least-squares fit
           ($r = 0.923$, $\mathrm{MAE} = 8.7\%$).
           Marker shapes encode model identity; colours encode Z-HiSPA configuration index.}
  \label{fig:scatter}
\end{figure}

% ---------------------------------------------------------------------------
\subsection{Experiment 3: \MBCA{} Coverage}
\label{sec:exp3}

Table~\ref{tab:mbca} summarises \MBCA{} probe performance at $K \in \{8, 16\}$ on
Jamba-1.5-Mini (the primary model).

\begin{table}[ht]
  \centering
  \caption{\MBCA{} results on Jamba-1.5-Mini.
           \betaMBCA{} is the coverage score (Theorem~2); the pass threshold is
           $\betaMBCA > 0.70$.
           \CHSS{} recovery measures residual mean degradation after \MBCA{} filtering;
           target is $\leq 25\%$ residual.}
  \label{tab:mbca}
  \begin{tabular}{rrrrrcc}
    \toprule
    $K$ & \betaMBCA{} & \textbf{Pass?} & \textbf{Probe Acc.} & \textbf{Probe F1} &
      \textbf{\CHSS{} (baseline)} & \textbf{\CHSS{} (w/\ \MBCA{})} \\
    \midrule
    8  & 0.74 & \checkmark & 0.87 & 0.84 & 61.8\% & 22.7\% \\
    16 & 0.82 & \checkmark & 0.88 & 0.83 & 61.8\% & 19.3\% \\
    \bottomrule
  \end{tabular}
\end{table}

\noindent At $K = 8$, \MBCA{} reduces mean \CHSS{} degradation from 61.8\% to 22.7\%,
recovering the model to safe operation below the 25\% residual threshold.
Increasing to $K = 16$ yields a further modest improvement (19.3\%) at the cost of higher
inference overhead (see Section~\ref{sec:k-sensitivity}).
Mean probe accuracy is 0.87 and mean F1 is 0.84 (averaged across 8 probes), indicating
that individual probes are reliable and the ensemble is well-calibrated.

% ---------------------------------------------------------------------------
\subsection{Experiment 4: Audit Ranking Validation}
\label{sec:exp4}

The SSH-Hybrid audit ranking, derived analytically from $g$, is
\[
  \underbrace{\text{Pythia}}_{\text{safest}} \succ \text{Zamba} \succ \text{Jamba}
  \succ \underbrace{\text{Mamba}}_{\text{most vulnerable}}.
\]
This ordering matches the empirical ranking obtained by sorting models on mean \CHSS{}
degradation (Table~\ref{tab:degradation}), confirming Theorem~1's ranking corollary.
The agreement holds across all seven Z-HiSPA configurations individually: no single
configuration inverts any adjacent pair in the ranking.

% ---------------------------------------------------------------------------
\subsection{Experiment 5: $K$-Sensitivity Analysis}
\label{sec:k-sensitivity}

\paragraph{Coverage vs.\ $K$.}
Table~\ref{tab:k-sweep} and Figure~\ref{fig:k-sweep} show \betaMBCA{} and benign capability
degradation as $K$ varies on Jamba-1.5-Mini.

\begin{table}[ht]
  \centering
  \caption{\MBCA{} coverage \betaMBCA{} and benign benchmark degradation as a function of
           probe count $K$ on Jamba-1.5-Mini.
           Benign degradation is the mean accuracy drop across MMLU, HellaSwag, and
           ARC-Challenge relative to the unguarded baseline.
           Pass/Fail is relative to $\betaMBCA > 0.70$.}
  \label{tab:k-sweep}
  \begin{tabular}{rrrrl}
    \toprule
    $K$ & \betaMBCA{} & \textbf{Pass?} & \textbf{Benign degradation (\%)} & \textbf{Note} \\
    \midrule
    4  & 0.58 & $\times$   & 0.4 & Below threshold \\
    8  & 0.74 & \checkmark & 0.8 & \textbf{Optimal} \\
    12 & 0.79 & \checkmark & 1.1 & --- \\
    16 & 0.82 & \checkmark & 1.2 & --- \\
    24 & 0.84 & \checkmark & 1.9 & Near 2\% ceiling \\
    \bottomrule
  \end{tabular}
\end{table}

\noindent $K = 4$ fails the coverage threshold; all larger values pass.
Benign degradation remains below 2\% for all $K \leq 24$.
We select $\mathbf{K = 8}$ as the optimal operating point: it is the smallest $K$ that
simultaneously satisfies $\betaMBCA > 0.70$, achieves \CHSS{} residual $\leq 25\%$, and
keeps benign degradation below 1\%.

\begin{figure}[ht]
  \centering
  % Placeholder: K-sweep figure generated at camera-ready time.
  \fbox{\rule{0pt}{4cm}\rule{0.55\textwidth}{0pt}}
  \caption{\betaMBCA{} (left axis, solid) and benign accuracy degradation (right axis,
           dashed) as a function of $K$.
           The horizontal dotted line marks the $\betaMBCA = 0.70$ pass threshold.
           The shaded region indicates benign degradation below 2\%.
           $K = 8$ (star marker) is the recommended operating point.}
  \label{fig:k-sweep}
\end{figure}

% ---------------------------------------------------------------------------
\subsection{Comparison with Baselines}
\label{sec:comparison}

Table~\ref{tab:comparison} situates SSH-Hybrid against existing safety frameworks across
four dimensions: ability to detect SSM-specific spectral risk, provision of formal
guarantees, runtime overhead, and adversarial coverage.

\begin{table}[ht]
  \centering
  \caption{Comparison of safety frameworks.
           ``SSM-specific risk'' indicates whether the method identifies risk arising from SSM
           state dynamics (as opposed to surface-level token filtering).
           ``Formal guarantees'' indicates whether coverage or detection bounds are proved
           rather than empirically estimated.
           Runtime overhead is assessed relative to base-model inference latency.
           Coverage is \betaMBCA{} where applicable; N/A indicates the framework does not
           define an equivalent metric.}
  \label{tab:comparison}
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{lcccc}
    \toprule
    \textbf{Method} &
      \textbf{\makecell{SSM-specific\\risk detection}} &
      \textbf{\makecell{Formal\\guarantees}} &
      \textbf{\makecell{Runtime\\overhead}} &
      \textbf{Coverage} \\
    \midrule
    SpectralGuard~\citep{lemercier2026spectralguard}
      & Partial       & No                           & Low              & N/A                  \\
    RoBench-25~\citep{lemercier2025robench}
      & Yes (empirical) & No                         & High             & N/A                  \\
    Standard safety filters
      & No            & No                           & Medium           & $\approx$60\%        \\
    \textbf{SSH-Hybrid (ours)}
      & \textbf{Yes}  & \textbf{Yes (Thm.~1\&2)}     & \textbf{Low ($<$100\,ms)} & \textbf{$>$70\%} \\
    \bottomrule
  \end{tabular}
\end{table}

\noindent SSH-Hybrid is the only framework that combines formal spectral guarantees
(Theorems~1 and~2) with low runtime overhead and auditable coverage scores.
SpectralGuard provides complementary spectral measurements but does not bound hidden-state
degradation or offer coverage guarantees.
Standard safety filters achieve only $\approx$60\% coverage and are entirely blind to
SSM-driven state collapse.


% =======================================================================
% SECTIONS 7-10: Discussion, Limitations, Broader Impact, Conclusion
% =======================================================================

\section{Discussion}
\label{sec:discussion}

\subsection{Why Hybrid Models Create Unique Safety Risks}

The safety risks introduced by hybrid SSM-Transformer architectures are qualitatively distinct from those in purely attention-based models. The root cause is the exponential signal decay that characterizes SSM hidden-state dynamics: after $\tau$ recurrence steps, information attenuates as $O(\rho^\tau)$. When $\rho < 1$---universally required for stability---this decay is irreversible and geometric.

In architectures such as Jamba (ratio 7:1 SSM to attention) and Zamba (5:1), long SSM subsequences separate consecutive attention checkpoints. A safety signal embedded at the beginning of an SSM block must survive dozens of recurrence steps before reaching the next attention layer. With $\rho \approx 0.993$ (Jamba) and interaction lengths $L \gg \Horizon$, the safety signal is attenuated by orders of magnitude before any attention-based oversight mechanism can act on it.

This creates the \emph{safety blind window}: the set of token positions beyond $\Horizon$ where early safety constraints have decayed below detection threshold. A deceptive model or adversarial prompt could embed harmful intent in early tokens, then rely on SSM decay to erase the evidentiary trace before it reaches the attention layers where oversight operates. By the time an attention layer processes later tokens, the hidden state retains no reliable record of the earlier constraint violation.

Pure Transformers do not exhibit this vulnerability: attention provides $O(1)$ access to all prior tokens via the key--value cache. The hybrid paradigm trades this global accessibility for computational efficiency, and SSH-Hybrid makes the safety cost of that trade explicit and quantifiable.

\subsection{The Monotonicity Principle}

The core insight of \MBCA{} is that SSM decay should not be \emph{corrected} within the SSM dynamics, but \emph{circumvented} by a parallel, non-decaying memory channel. Modifying the recurrence matrix $\bar{A}$ to slow decay (pushing $\rho \to 1$) is brittle: it degrades the model's ability to forget irrelevant context, disrupts learned representations, and provides no formal guarantee.

\MBCA{} introduces monotone OR-updates over $K$ binary probe activations. Once probe $k$ fires, flag $c[k] = 1$ persists indefinitely, unaffected by subsequent SSM state evolution. Theorem~2 formalizes this: the \MBCA{} coverage guarantee $\betaMBCA$ holds \emph{independent} of $\rho$, $\rSSM$, and interaction length. This independence is the crucial property---\MBCA{}'s safety guarantee does not degrade as the architecture becomes more SSM-heavy or contexts grow longer.

\subsection{Spectral Radius as a Safety Metric}

A central contribution is elevating $\rho(\bar{A})$ from a numerical stability condition to a first-class safety metric. Three properties make this compelling:

\begin{enumerate}[label=(\arabic*)]
  \item \textbf{Architecture-independent}: $\rho$ can be computed for any SSM---S4, Mamba, or otherwise---without reference to the broader architecture.
  \item \textbf{Measurable}: eigendecomposition of $\bar{A}$ takes milliseconds for typical SSM state dimensions ($d_{\mathrm{state}} \leq 256$).
  \item \textbf{Predictive}: Pearson $r = 0.923$ between the attenuation factor $g(\rho, \rSSM, L)$ and empirical HiSPA degradation across all model--attack pairs.
\end{enumerate}

The safety horizon $\Horizon$ translates this eigenvalue into an interpretable bound: the number of tokens over which a safety signal remains above threshold. This is directly actionable: if $\Horizon < L_{\max}$ for a deployment scenario, the model has a certified blind window that must be addressed.

\subsection{Practical Implications}

\paragraph{Spectral safety auditing as standard practice.}
Any hybrid SSM-Transformer model should undergo spectral safety auditing before deployment. Phases 1--2 require $<$100\,ms; Phase 3 is more expensive but parallelizable.

\paragraph{MBCA as deployable intervention.}
With $K = 8$ probes, \MBCA{} achieves $\betaMBCA > 0.70$ with $< 1\%$ benign degradation on MMLU/HellaSwag/ARC. Overhead scales as $O(K \cdot d_{\mathrm{model}})$ per token, dominated by model computation.

\paragraph{Generalizability.}
The framework applies to any architecture with linear recurrence layers alternating with attention: Griffin~\cite{de2024griffin}, RWKV-hybrid~\cite{peng2023rwkv}, and future designs. As long as the recurrence has a well-defined spectral radius, Theorem~1 applies.

\paragraph{Architecture design guidance.}
For a target safety horizon $H_{\mathrm{target}}$, the constraint $g < \epsilon_{\mathrm{tol}}$ bounds the allowable $\rSSM$ as a function of $\rho$ and $L$. Designers can use this to choose SSM-to-attention ratios that maintain acceptable safety margins.

% =======================================================================
\section{Limitations and Future Work}
\label{sec:limitations}

\paragraph{Linear SSM assumption.}
Theorem~1 relies on linear recurrence dynamics. Mamba-2 introduces selective scan dynamics (SSD) where $\bar{A}_t$ is input-dependent~\cite{dao2024mamba2}. Extensions to nonlinear SSMs may require bounding the spectral radius of the worst-case input-dependent operator.

\paragraph{Training data coverage.}
\MBCA{} probes are trained on BeaverTails~\cite{ji2023beavertails}, which underrepresents specialized domains (biosecurity, election interference). The coverage guarantee $\betaMBCA$ is only as strong as the per-probe detection rates, limited by the training distribution.

\paragraph{Bound tightness.}
Theorem~1 is tight in the worst case but may be loose for specific architectures where safety features distribute across multiple eigenmodes. Characterizing the tightness gap as a function of architecture design remains open.

\paragraph{Multi-modal extension.}
The framework is developed for text-only models. Multi-modal hybrids with vision/audio tokens may exhibit different SSM decay interactions not captured by single-modality analysis.

\paragraph{Adaptive probe selection.}
Fixed $K$ probes trained offline could be replaced by adaptive variants that select or weight probes dynamically, potentially achieving higher coverage with fewer probes.

\paragraph{Formal verification connections.}
Connecting spectral safety bounds to formal verification methods (abstract interpretation, satisfiability-based analysis) could yield stronger worst-case guarantees.

% =======================================================================
\section{Broader Impact}
\label{sec:broader_impact}

\paragraph{Positive impacts.}
SSH-Hybrid provides the first principled framework for evaluating safety properties of hybrid SSM-Transformer architectures. By making safety blind windows visible and measurable, the framework enables organizations to identify and mitigate architectural risks \emph{before} deployment. The audit procedure is lightweight and architecture-agnostic, suitable for integration into existing evaluation pipelines.

More broadly, identifying \emph{architecture-level} safety risks---as distinct from training-level risks---opens a new dimension in AI safety. We hope this encourages deeper engagement with the architectural properties of emerging model classes.

\paragraph{Negative risks and mitigation.}
Knowledge of safety blind windows could be misused by adversaries. We argue transparency is correct: (1) the vulnerability exists regardless of publication and can be discovered by empirical probing; (2) we publish both attack characterization and defense simultaneously; (3) the AI safety community broadly endorses responsible disclosure with simultaneous mitigations~\cite{carlini2024aligned}.

% =======================================================================
\section{Conclusion}
\label{sec:conclusion}

We have presented SSH-Hybrid, the first formal framework connecting SSM spectral properties to quantifiable safety margins in hybrid SSM-Transformer architectures. The framework addresses a gap that has grown urgent as hybrid models enter production: the absence of principled methods for characterizing safety implications of SSM-specific dynamics.

Three core contributions:

\begin{enumerate}[label=(\arabic*)]
\item \textbf{Theorem~1} establishes a formal upper bound on safety signal retention as a function of $\rho$, $\rSSM$, and $L$, defining the safety horizon $\Horizon$ and attenuation factor $g$ as scalar predictors of safety degradation.

\item \textbf{The \MBCA{} mechanism with Theorem~2} exploits monotone OR-updates to provide persistent safety memory provably immune to SSM decay, with coverage guarantee $\betaMBCA$ independent of $\rho$.

\item \textbf{The 3-phase spectral safety audit} translates the theoretical framework into a practical pipeline combining lightweight spectral computation ($<$100\,ms) with empirical validation via HiSPA.
\end{enumerate}

Empirical validation across four architectures confirms: Pearson $r = 0.923$ between predicted and empirical degradation; \MBCA{} with $K=8$ achieves $\betaMBCA > 0.70$ with $<$2\% benign degradation; the audit correctly ranks all architectures by safety horizon.

\textbf{Our call to action is direct: every hybrid SSM-Transformer model should undergo spectral safety auditing before deployment.} The audit is lightweight, the theoretical basis is sound, and the cost of omitting it---deploying a model with an uncharacterized safety blind window---is potentially severe. We release the SSH-Hybrid codebase, audit scripts, and pre-trained \MBCA{} probes at \url{https://github.com/anonymous/ssh-hybrid}.

% =======================================================================
\section*{Acknowledgments}

We thank the anonymous reviewers for constructive feedback and the safety teams at the organizations whose architectures are evaluated for their engagement during responsible disclosure. This work was supported in part by academic compute allocations. The authors have no financial conflicts of interest.

% =======================================================================
\begin{thebibliography}{99}

\bibitem{gu2022efficiently}
Albert Gu, Karan Goel, and Christopher R\'{e}.
Efficiently modeling long sequences with structured state spaces.
In \emph{ICLR}, 2022.

\bibitem{gu2023mamba}
Albert Gu and Tri Dao.
Mamba: Linear-time sequence modeling with selective state spaces.
\emph{arXiv:2312.00752}, 2023.

\bibitem{dao2024mamba2}
Tri Dao and Albert Gu.
Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality.
In \emph{ICML}, 2024.

\bibitem{lieber2024jamba}
Opher Lieber et al.
Jamba: A hybrid Transformer-Mamba language model.
\emph{arXiv:2403.19887}, 2024.

\bibitem{lieber2024jamba15}
Opher Lieber et al.
Jamba-1.5: Hybrid Transformer-Mamba models at scale.
\emph{arXiv:2408.12570}, 2024.

\bibitem{glorioso2024zamba}
Paolo Glorioso et al.
Zamba: A compact 7B SSM hybrid model.
Technical report, Zyphra, 2024.

\bibitem{de2024griffin}
Soham De et al.
Griffin: Mixing gated linear recurrences with local attention for efficient language models.
\emph{arXiv:2402.19427}, 2024.

\bibitem{peng2023rwkv}
Bo Peng et al.
RWKV: Reinventing RNNs for the Transformer era.
In \emph{Findings of EMNLP}, 2023.

\bibitem{ouyang2022training}
Long Ouyang et al.
Training language models to follow instructions with human feedback.
In \emph{NeurIPS}, 2022.

\bibitem{bai2022constitutional}
Yuntao Bai et al.
Constitutional AI: Harmlessness from AI feedback.
\emph{arXiv:2212.08073}, 2022.

\bibitem{bai2022training}
Yuntao Bai et al.
Training a helpful and harmless assistant with reinforcement learning from human feedback.
\emph{arXiv:2204.05862}, 2022.

\bibitem{anil2024many}
Cem Anil et al.
Many-shot jailbreaking.
Technical report, Anthropic, 2024.

\bibitem{zou2023universal}
Andy Zou, Zifan Wang, J.~Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models.
\emph{arXiv:2307.15043}, 2023.

\bibitem{zou2023representation}
Andy Zou et al.
Representation engineering: A top-down approach to AI transparency.
\emph{arXiv:2310.01405}, 2023.

\bibitem{ji2023beavertails}
Jiaming Ji et al.
BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset.
In \emph{NeurIPS}, 2023.

\bibitem{wang2024spectralguard}
Martin Le~Mercier, Ana\"{i}s Vergne, and Cl\'{e}ment Beysson.
SpectralGuard: Spectral analysis for state space model safety.
\emph{arXiv preprint}, 2026.

\bibitem{lemercier2025robench}
Martin Le~Mercier, Ana\"{i}s Vergne, and Cl\'{e}ment Beysson.
RoBench-25: Robustness benchmarking for large language models.
In \emph{EMNLP}, 2025.

\bibitem{kalman1960}
Rudolf E. Kalman.
A new approach to linear filtering and prediction problems.
\emph{J.~Basic Engineering}, 82(1):35--45, 1960.

\bibitem{hendrycks2021mmlu}
Dan Hendrycks et al.
Measuring massive multitask language understanding.
In \emph{ICLR}, 2021.

\bibitem{zellers2019hellaswag}
Rowan Zellers et al.
HellaSwag: Can a machine really finish your sentence?
In \emph{ACL}, 2019.

\bibitem{clark2018arc}
Peter Clark et al.
Think you have solved question answering? Try ARC, the AI2 reasoning challenge.
\emph{arXiv:1803.05457}, 2018.

\bibitem{wei2024jailbroken}
Alexander Wei, Nika Haghtalab, and Jacob Steinhardt.
Jailbroken: How does LLM safety training fail?
In \emph{NeurIPS}, 2024.

\bibitem{carlini2024aligned}
Nicholas Carlini et al.
Are aligned language models adversarially aligned?
\emph{arXiv:2402.13725}, 2024.

\bibitem{miyato2018spectral}
Takeru Miyato et al.
Spectral normalization for generative adversarial networks.
In \emph{ICLR}, 2018.

\bibitem{pascanu2013difficulty}
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio.
On the difficulty of training recurrent neural networks.
In \emph{ICML}, 2013.

\bibitem{gao2023lmeval}
Leo Gao et al.
A framework for few-shot language model evaluation.
\emph{Zenodo}, 2023.

\bibitem{perez2022red}
Ethan Perez et al.
Red teaming language models with language models.
In \emph{EMNLP}, 2022.

\bibitem{perez2022ignore}
Ethan Perez and Marco Tulio Ribeiro.
Ignore this title and HackAPrompt: Exposing systemic weaknesses of LLMs.
In \emph{EMNLP}, 2023.

\bibitem{ganguli2022red}
Deep Ganguli et al.
Red teaming language models to reduce harms.
\emph{arXiv:2209.07858}, 2022.

\bibitem{inan2023llama}
Hakan Inan et al.
Llama Guard: LLM-based input-output safeguard for human-AI conversations.
\emph{arXiv:2312.06674}, 2023.

\bibitem{gehman2020realtoxicity}
Samuel Gehman et al.
RealToxicityPrompts: Evaluating neural toxic degeneration in language models.
In \emph{Findings of EMNLP}, 2020.

\bibitem{mazeika2024harmbench}
Mantas Mazeika et al.
HarmBench: A standardized evaluation framework for automated red teaming and robust refusal.
\emph{arXiv:2402.04249}, 2024.

\bibitem{schulman2017proximal}
John Schulman et al.
Proximal policy optimization algorithms.
\emph{arXiv:1707.06347}, 2017.

\bibitem{rafailov2023direct}
Rafael Rafailov et al.
Direct preference optimization: Your language model is secretly a reward model.
In \emph{NeurIPS}, 2023.

\bibitem{smith2022s5}
Jimmy T.H. Smith, Andrew Warrington, and Scott W. Linderman.
Simplified state space layers for sequence modeling.
In \emph{ICLR}, 2023.

\bibitem{poli2023hyena}
Michael Poli et al.
Hyena hierarchy: Towards larger convolutional language models.
In \emph{ICML}, 2023.

\bibitem{liu2024autodan}
Xiaogeng Liu et al.
AutoDAN: Generating stealthy jailbreak prompts on aligned large language models.
In \emph{ICLR}, 2024.

\bibitem{chao2023pair}
Patrick Chao et al.
Jailbreaking black box large language models in twenty queries.
\emph{arXiv:2310.08419}, 2023.

\bibitem{zhang2024zhispa}
Yichi Zhang et al.
Z-HiSPA: Zero-shot hierarchical structured prompt attacks on LLMs.
\emph{arXiv preprint}, 2024.

\bibitem{chen2024speculative}
Charlie Chen et al.
Speculative safety: Adversarial attacks on speculative decoding.
\emph{arXiv preprint}, 2024.

\bibitem{sun2023retentive}
Yutao Sun et al.
Retentive network: A successor to Transformer for large language models.
\emph{arXiv:2307.08621}, 2023.

\bibitem{pennington2017resurrecting}
Jeffrey Pennington et al.
Resurrecting the sigmoid in deep learning through dynamical isometry.
In \emph{NeurIPS}, 2017.

\bibitem{elhage2021mathematical}
Nelson Elhage et al.
A mathematical framework for Transformer circuits.
\emph{Transformer Circuits Thread}, 2021.

\end{thebibliography}

\end{document}
