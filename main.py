import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Beschrijving van Blackjack Gymnasium https://gymnasium.farama.org/environments/toy_text/blackjack/
# Prima methode: https://www.javatpoint.com/q-learning-in-python
# Reinforcement learning document p. 131 voor Q-learning


def train_rl(env, episodes, alpha, gamma):

    random.seed(1)

    S1_max = 21
    S2_max = 10
    S3_max = 1
    S_max = [S1_max, S2_max, S3_max]

    # Setting up the Q-table: important for reinforcement learning using the Q-learning method.
    # Q-table layout: 2 columns for both actions, number of rows according to possible states in sections of
    # S3, then sections of S2, and then S1 that changes constantly
    Q_table = np.zeros((S1_max * S2_max * (S3_max+1), 2))

    # Iterations table: not necessary for the program, but serves as a check whether all observations and action
    # combinations have been visited frequently.
    its_table = np.zeros((S1_max * S2_max * (S3_max+1)))

    # Observations table: again, not necessary, but served as a check whether the translation from the three different
    # sets of observations to the one array went well.
    obs_table = np.zeros((S1_max * S2_max * (S3_max+1), 3))

    for episode in range(episodes):

        # Start of game: e.g. drawing hands.
        observations, _ = env.reset()
        S1, S2, S3 = observations

        # Setting the current hand as the 'previous state' for the Q-learning function.
        prev_state = S3 * (S1_max * S2_max) + (S2 - 1) * S1_max + (S1-1)

        # Used to serve as a check, does not really serve a purpose now.
        obs_table[prev_state] = observations

        terminated = False

        # Terminates if player exceeds 21 or decides to stick.
        while not terminated:

            # Randomised action selection.
            action = random.choice([0, 1])
            observations, reward, terminated, _, _ = env.step(action)
            S1, S2, S3 = observations

            # Translation of observations of new step to the single array /  Q-table index.
            state = S3 * (S1_max * S2_max) + (S2 - 1) * S1_max + (S1-1)

            # Q-learning function
            Q_table[prev_state, action] = Q_table[prev_state, action] + alpha * (reward + gamma * max(Q_table[state, :]) - Q_table[prev_state, action])

            # Used to be handy, now not per se.
            its_table[prev_state] += 1

            # Preparing for the next Q-learning function.
            prev_state = state

    return Q_table, S_max


def plot_rl(Q_table, S_max, plot_hos=True):
    # Hit-or-stick matrix
    S1_max, S2_max, S3_max = S_max
    hos = np.zeros(Q_table.shape[0])
    for i in range(len(hos)):
        if Q_table[i, 0] > Q_table[i, 1]:
            hos[i] = 0  # Stick
        elif Q_table[i, 0] < Q_table[i, 1]:
            hos[i] = 1  # Hit
        else:
            hos[i] = -1  # No answer

    if plot_hos:

        hos_no_ace = hos[:S1_max * S2_max].reshape((S1_max, S2_max), order="F")
        hos_ace = hos[S1_max * S2_max:].reshape((S1_max, S2_max), order="F")

        plt.subplot(1, 2, 1)
        plt.imshow(hos_no_ace, cmap='brg', interpolation='nearest', extent=[0.5, 10.5, 21.5, 0.5])
        ax = plt.gca()
        plt.ylabel("Player's hand")
        plt.xlabel("Dealer's visible card")
        ax.set_xticks(np.arange(1, 11))
        ax.set_xticks(np.arange(0.5, 10.5, 0.5), minor=True)
        ax.set_yticks(np.arange(1, 22))
        ax.set_yticks(np.arange(0.5, 21.5, 0.5), minor=True)
        ax.grid(which='minor', color='grey')
        plt.title("No usable ace")

        plt.subplot(1, 2, 2)
        plt.imshow(hos_ace, cmap='brg', interpolation='nearest', extent=[0.5, 10.5, 21.5, 0.5])
        ax = plt.gca()
        plt.ylabel("Player's hand")
        plt.xlabel("Dealer's visible card")
        ax.set_xticks(np.arange(1, 11))
        ax.set_xticks(np.arange(0.5, 10.5, 0.5), minor=True)
        ax.set_yticks(np.arange(1, 22))
        ax.set_yticks(np.arange(0.5, 21.5, 0.5), minor=True)
        ax.grid(which='minor', color='grey')
        plt.title("Usable ace")

        plt.show()

        return


def test_rl(Q_table, S_max):

    random.seed(2)

    test_episodes = 1000
    score = 0
    S1_max, S2_max, S3_max = S_max

    for test_episode in range(test_episodes):

        observations, _ = env.reset()
        S1, S2, S3 = observations
        state = S3 * (S1_max * S2_max) + (S2 - 1) * S1_max + (S1-1)

        terminated = False
        while not terminated:

            # Determining action using the Q-table
            if Q_table[state, 0] > Q_table[state, 1]:
                # STICK
                action = 0
            elif Q_table[state, 0] < Q_table[state, 1]:
                # HIT
                action = 1
            else:
                # print("Action not determined for this combination, action is performed at random.")
                action = random.choice([0, 1])



            observations, reward, terminated, _, _ = env.step(action)
            S1, S2, S3 = observations
            state = S3 * (S1_max * S2_max) + (S2 - 1) * S1_max + (S1-1)

            # print("Current hand: ", S1, "with ", S3,"aces. Dealer card: ", S2, "\n We choose ", action, "and the reward is ", reward)

            score += reward

    win_percentage = (score + test_episodes) / (test_episodes * 2) * 100

    return win_percentage


def sensitivity_analysis(env, episodes, alpha_min, alpha_max, alpha_step, gamma_min, gamma_max, gamma_step, global_ranges=False, logarithmic=True):

    if logarithmic:
        alpha_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    else:
        alpha_range = np.round(np.arange(alpha_min/alpha_step, (alpha_max+alpha_step)/alpha_step, 1) * alpha_step, decimals=2)

    gamma_range = np.round(np.arange(gamma_min/gamma_step, (gamma_max+gamma_step)/gamma_step, 1) * gamma_step, decimals=2)

    wins_array = np.zeros((len(alpha_range), len(gamma_range)))

    i=0
    for alpha in alpha_range:
        j=0
        for gamma in gamma_range:
            Q_table, S_max = train_rl(env, episodes, alpha, gamma)
            win_percentage = test_rl(Q_table, S_max)
            wins_array[i, j] = win_percentage
            print("For alpha = ", alpha, " and gamma = ", gamma, ", win percentage = ", win_percentage, "\n")
            j += 1
        i += 1

    print("Highest win percentage: ", np.max(wins_array))

    plt.imshow(wins_array, cmap='PRGn', interpolation='nearest', extent=[gamma_min-0.5*gamma_step, gamma_max+0.5*gamma_step, alpha_min-0.5*alpha_step, alpha_max+0.5*alpha_step], origin='lower')
    ax = plt.gca()
    if global_ranges:
        mult = 2
    else:
        mult = 5

    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$\alpha$')
    ax.set_xticks(np.round(np.arange(gamma_min-gamma_step+mult*gamma_step, gamma_max+gamma_step*mult, gamma_step*mult), decimals=2))
    #ax.set_yticks(np.round(np.arange(alpha_min-alpha_step+mult*alpha_step, alpha_max+alpha_step*mult, alpha_step*mult)[:-1], decimals=2))
    plt.title(r'Win percentages for focused ranges of $\alpha$ and $\gamma$')
    plt.colorbar()

    plt.show()


#### MAIN ####

env = gym.make('Blackjack-v1', render_mode=None, natural=True, sab=False)

sens = False

if not sens:
    alph = 0.1  # Update rate: low because of many iterations
    gamm = 0.9  # Discount rate
    train_episodes = 100000

    Q_table, S_max = train_rl(env, train_episodes, alph, gamm)
    plot_rl(Q_table, S_max)
    win_percentage = test_rl(Q_table, S_max)

    print("Win percentage: ", win_percentage, "\n")

else:
    alphagamma_sens = False
    episodes_sens = True

    if alphagamma_sens:

        sens_episodes = 100000

        gamma_step = 0.02
        gamma_min = gamma_step
        gamma_max = 0.4
        gammas = [gamma_min, gamma_max, gamma_step]

        alpha_step = 0.02
        alpha_min = alpha_step
        alpha_max = 0.3
        alphas = [alpha_min, alpha_max, alpha_step]
        sensitivity_analysis(env, sens_episodes, alpha_min, alpha_max, alpha_step, gamma_min, gamma_max, gamma_step)

    elif episodes_sens:

        episodes_list = [100, 1000, 10000, 100000, 1000000, 10000000]
        alph = 0.01
        gamm = 0.08

        for train_episodes in episodes_list:
            t = time.time()
            Q_table, S_max = train_rl(env, train_episodes, alph, gamm)
            print("For ", train_episodes, " episodes, the time is ", time.time()-t)
            plot_rl(Q_table, S_max)
            win_percentage = test_rl(Q_table, S_max)

            print("For ", train_episodes, ", the win percentage = ", win_percentage, "\n")

    else:
        print("Select type of sensitivity analysis.")


env.close()




