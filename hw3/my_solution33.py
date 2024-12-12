import numpy as np
import logging

def read_state_weights(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    state_weights = {}
    num_states, default_weight = lines[1].strip().split()
    default_weight = float(default_weight)
    for line in lines[2:]:
        state, weight = line.strip().split()
        state = state.strip('"')
        state_weights[state] = float(weight)
    return state_weights

def read_state_action_state_weights(file_path, use_actions):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    if use_actions:
        num_triples, num_states, num_actions, default_weight = lines[1].strip().split()
        num_triples = int(num_triples)
        num_states = int(num_states)
        num_actions = int(num_actions)
    else:
        num_triples, num_states, default_weight = lines[1].strip().split()
        num_triples = int(num_triples)
        num_states = int(num_states)
        num_actions = 0  # No actions
    default_weight = float(default_weight)
    transition_weights = {}
    states = set()
    actions = set()
    for line in lines[2:]:
        parts = line.strip().split()
        if use_actions:
            s_prev, action, s_next, weight = parts
            s_prev = s_prev.strip('"')
            action = action.strip('"')
            s_next = s_next.strip('"')
            weight = float(weight)
            transition_weights[(s_prev, action, s_next)] = weight
            actions.add(action)
        else:
            s_prev, s_next, weight = parts
            s_prev = s_prev.strip('"')
            s_next = s_next.strip('"')
            weight = float(weight)
            transition_weights[(s_prev, s_next)] = weight
        states.update([s_prev, s_next])
    # Normalize to get transition probabilities
    transition_probs = {}
    if use_actions:
        for s_prev in states:
            for action in actions:
                total_weight = sum(transition_weights.get((s_prev, action, s_next), default_weight)
                                   for s_next in states)
                for s_next in states:
                    weight = transition_weights.get((s_prev, action, s_next), default_weight)
                    prob = weight / total_weight if total_weight > 0 else 0
                    transition_probs[(s_prev, action, s_next)] = prob
    else:
        for s_prev in states:
            total_weight = sum(transition_weights.get((s_prev, s_next), default_weight)
                               for s_next in states)
            for s_next in states:
                weight = transition_weights.get((s_prev, s_next), default_weight)
                prob = weight / total_weight if total_weight > 0 else 0
                transition_probs[(s_prev, s_next)] = prob
    return transition_probs, states, actions

def read_state_observation_weights(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    num_pairs, num_states, num_observations, default_weight = lines[1].strip().split()
    num_pairs = int(num_pairs)
    num_states = int(num_states)
    num_observations = int(num_observations)
    default_weight = float(default_weight)
    observation_weights = {}
    states = set()
    observations = set()
    for line in lines[2:]:
        state, observation, weight = line.strip().split()
        state = state.strip('"')
        observation = observation.strip('"')
        weight = float(weight)
        observation_weights[(state, observation)] = weight
        states.add(state)
        observations.add(observation)
    # Normalize to get P(o | s)
    observation_probs = {}
    for state in states:
        total_weight = sum(observation_weights.get((state, o), default_weight)
                           for o in observations)
        for observation in observations:
            weight = observation_weights.get((state, observation), default_weight)
            prob = weight / total_weight if total_weight > 0 else 0
            observation_probs[(state, observation)] = prob
    return observation_probs, observations

def read_observation_actions(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    num_pairs = int(lines[1].strip())
    observation_actions = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) == 2:
            observation, action = parts
            observation = observation.strip('"')
            action = action.strip('"')
            if action == "N":
                action = None
            observation_actions.append((observation, action))
        elif len(parts) == 1:
            observation = parts[0].strip('"')
            observation_actions.append((observation, None))
    return observation_actions

def write_states(file_path, states_sequence):
    with open(file_path, 'w') as f:
        f.write('states\n')
        f.write(f'{len(states_sequence)}\n')
        for state in states_sequence:
            f.write(f'"{state}"\n')

def viterbi_algorithm(state_weights, transition_probs, observation_probs, observation_actions, states, actions, use_actions):
    T = len(observation_actions)
    states = list(states)
    N = len(states)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    state_indices = {state: i for i, state in enumerate(states)}

    logging.info("States: %s", states)
    logging.info("Observation Actions: %s", observation_actions)

    # Initialization
    initial_observation, initial_action = observation_actions[0]
    logging.info("Initial Observation: %s", initial_observation)
    for i, state in enumerate(states):
        delta[0, i] = state_weights.get(state, 0) * observation_probs.get((state, initial_observation), 0)
        psi[0, i] = 0
        logging.debug("delta[0][%s] = %f", state, delta[0, i])
    # Normalize delta[0]
    if np.sum(delta[0]) > 0:
        delta[0] /= np.sum(delta[0])

    logging.info("Delta after initialization: %s", delta[0])

    # Recursion
    for t in range(1, T):
        observation, current_action = observation_actions[t]
        previous_action = observation_actions[t-1][1]  # Action that caused transition to current state
        logging.info("Time Step %d: Observation = %s, Previous Action = %s", t, observation, previous_action)
        for j, s_j in enumerate(states):
            max_prob = 0
            max_state = 0
            for i, s_i in enumerate(states):
                if use_actions and previous_action is not None:
                    trans_prob = transition_probs.get((s_i, previous_action, s_j), 0)
                else:
                    # Use transition probabilities without actions
                    trans_prob = transition_probs.get((s_i, s_j), 0)
                prob = delta[t-1, i] * trans_prob
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            delta[t, j] = max_prob * observation_probs.get((s_j, observation), 0)
            psi[t, j] = max_state
            logging.debug("delta[%d][%s] = %f (prev state: %s)", t, s_j, delta[t, j], states[max_state])
        # Normalize delta[t]
        if np.sum(delta[t]) > 0:
            delta[t] /= np.sum(delta[t])
        logging.info("Delta at time %d: %s", t, delta[t])

    # Termination
    last_state = np.argmax(delta[T-1])
    states_sequence = [states[last_state]]
    logging.info("Starting Backtracking from state: %s", states[last_state])

    # Backtracking
    for t in range(T-1, 0, -1):
        last_state = psi[t, last_state]
        states_sequence.insert(0, states[last_state])
        logging.info("Backtracked to state: %s at time %d", states[last_state], t-1)

    logging.info("Most probable states sequence: %s", states_sequence)
    return states_sequence

def main():
    # Set up logging
    logging.basicConfig(
        filename='viterbi_log.txt',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

    # Determine whether to use actions based on the content of the observation_actions.txt file
    observation_actions = read_observation_actions('observation_actions.txt')
    use_actions = any(action is not None for _, action in observation_actions)
    logging.info("Using actions: %s", use_actions)

    state_weights = read_state_weights('state_weights.txt')
    logging.info("State Weights: %s", state_weights)

    transition_probs, states, actions = read_state_action_state_weights('state_action_state_weights.txt', use_actions)
    logging.info("Transition Probabilities:")
    for key, value in transition_probs.items():
        if use_actions:
            logging.debug("P(%s | %s, %s) = %f", key[2], key[0], key[1], value)
        else:
            logging.debug("P(%s | %s) = %f", key[1], key[0], value)

    observation_probs, observations_set = read_state_observation_weights('state_observation_weights.txt')
    logging.info("Observation Probabilities:")
    for key, value in observation_probs.items():
        logging.debug("P(%s | %s) = %f", key[1], key[0], value)

    states_sequence = viterbi_algorithm(
        state_weights, transition_probs, observation_probs, observation_actions, states, actions, use_actions)

    write_states('states.txt', states_sequence)

if __name__ == "__main__":
    main()
