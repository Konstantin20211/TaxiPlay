import numpy as np
import random
from math import fabs
from environment import TaxiEnv
from IPython.display import clear_output
from visual import print_frames
from constants import Q_LEARNING_PARAMS as QP

env = TaxiEnv()
q_table1 = np.zeros([env.states_num, env.actions_num])


def learn(table, print_or_not_print, whether_save_history):
    epochs_required_for_100 = 0
    history = [[], [], []]
    epochs_required = 0
    delta_q = 0

    for i in range(1, QP['max_games'] + 1):
        state = env.reset()

        # Init Vars
        epochs, reward = 0, 0
        done = False

        while not done and epochs < QP['max_epochs']:
            epochs_required_for_100 += 1
            if random.uniform(0, 1) < QP['epsilon']:
                # Check the action space
                action = random.randint(0, env.actions_num - 1)
            else:
                # Check the learned values
                action = np.argmax(table[state])

            next_state, reward, done = env.step(action)

            old_value = table[state, action]
            next_max = np.max(table[next_state])

            # Update the new value
            new_value = (1 - QP['alpha']) * old_value + QP['alpha'] * \
                (reward + QP['gamma'] * next_max)
            delta_q += fabs(new_value-old_value)
            table[state, action] = new_value

            state = next_state
            epochs += 1

        if i % 100 == 0:
            print_information(print_or_not_print, whether_save_history,
                              epochs_required_for_100, delta_q,
                              epochs_required, history, i)
            epochs_required_for_100 = 0
            if delta_q < QP['delta']:
                break
            delta_q = 0

    if print_or_not_print == 'print':
        print(f"Training finished. It required {epochs_required} epochs.\n")
    return epochs_required, history


def print_information(print_or_not_print, whether_save_history,
                      epochs_required_for_100, delta_q,
                      epochs_required, history, i):
    clear_output(wait=True)
    if print_or_not_print == 'print':
        print(f"Episode: {i}, "
              f"Required: {epochs_required_for_100} epochs and Q changed by {delta_q}")
    epochs_required += epochs_required_for_100
    if whether_save_history:
        history[0].append(epochs_required_for_100)
        history[1].append(i)
        history[2].append(delta_q)


def play_and_print(table):
    frames = []
    done = False
    state = env.reset()
    while not done:
        action = np.argmax(table[state])
        state, reward, done = env.step(action)

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
        }
        )

    print_frames(frames)


def play_all(table, print_or_not_print):
    counter = 0
    for el in env.initial_states_list:
        env.s = el
        state = el
        epochs = 0
        done = False
        while not done and epochs < 100:
            epochs += 1
            action = np.argmax(table[state])
            state, reward, done = env.step(action)
        if epochs >= 100:
            counter += 1
    if print_or_not_print == 'print':
        print(f"Failed {counter} of {len(env.initial_states_list)} initial states")
    return counter


def learn_and_write(table, filename):
    learn(table, 'print', False)
    file = open(filename, 'w')
    for el in q_table1:
        file.write(str(el) + '\n')
    file.close()


def read_data(filename):
    data = [[], [], []]
    array = []
    with open(filename, 'r') as file:
        for line in file:
            array.extend(float(x) for x in line.split(' '))
    for i in range(int(len(array)/3)):
        data[0].append(array[i * 3])
        data[1].append(array[i * 3 + 1])
        data[2].append(array[i * 3 + 2])
    return data


if __name__ == '__main__':
    learn_and_write(q_table1, 'Q_TABLE')
    play_and_print(q_table1)
    # play_all(q_table1, 'print')
