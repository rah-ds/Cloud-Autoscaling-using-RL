from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


rates: List[float] = [
    5,
    50,
    200,
]  # lambdas for the Poisson process in different states (idle, normal, busy)

transitions: List[List[float]] = [
    [0.90, 0.09, 0.01],  # from idle to idle, normal, busy
    [0.15, 0.80, 0.05],  # from normal to idle, normal, busy
    [0.05, 0.15, 0.80],  # from busy to idle, normal, busy
]


def simulate_mmpp(
    rates: Optional[List[float]] = None,
    transitions: Optional[List[List[float]]] = None,
    duration: int = 1,
    print_: bool = False,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Simulate a Markov-Modulated Poisson Process (MMPP).

    Args:
        rates: list of lambda (arrivals per time step) per state.
        transitions: state transition probability matrix (rows sum to 1).
        duration: number of discrete time steps to simulate.
        print_: whether to print per-step info.
        seed: optional random seed for reproducibility.

    Returns:
        arrivals: list of ints, number of arrivals at each time step.
        states: list of ints, state index at each time step (before transition).
    """
    # Use module-level defaults if not provided
    if rates is None:
        rates = [5, 50, 200]
    if transitions is None:
        transitions = [
            [0.90, 0.09, 0.01],
            [0.15, 0.80, 0.05],
            [0.05, 0.15, 0.80],
        ]
    
    if seed is not None:
        np.random.seed(seed)

    state = 0
    states_ = []
    arrivals = []

    for t in range(duration):
        lam = rates[state]
        # number of arrivals in this time step
        count = np.random.poisson(lam)
        arrivals.append(count)
        states_.append(state)

        if print_:
            print(f"t={t}, state={state}, lambda={lam}, arrivals={count}")

        # transition to next state
        state = np.random.choice(len(rates), p=transitions[state])

    return arrivals, states_


def plot_mmpp_simulation(arrivals: List[int], states_: List[int]) -> None:
    t = np.arange(len(arrivals))
    fig, axs = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # arrivals over time
    axs[0].bar(t, arrivals, color="C0", alpha=0.8)
    axs[0].set_ylabel("Arrivals per step")
    axs[0].set_title("MMPP arrivals and state over time")
    axs[0].grid(True)

    # state over time
    axs[1].step(t, states_, where="post", marker="o")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("State")
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(["idle", "normal", "busy"])
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# test the simulation and plotting
# if __name__ == "__main__":
#     arrivals, states_ = simulate_mmpp(duration=200, seed=42)
#     plot_mmpp_simulation(arrivals, states_)
