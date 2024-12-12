import numpy as np

def read_state_weights(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    state_weights = {}
    line1 = []
    line1 = lines[1].strip().split()
    if len(line1) == 2:
      default_weight = line1[1]
    else:
      default_weight = 0
    default_weight = float(default_weight)
    for line in lines[2:]:
        state, weight = line.strip().split()
        state = state.strip('"')
        state_weights[state] = float(weight)
    return state_weights

def read_state_action_state_weights(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_triples, num_states, num_actions, default_weight = lines[1].strip().split()
    num_triples = int(num_triples)
    num_states = int(num_states)
    num_actions = int(num_actions)
    default_weight = float(default_weight)
    transition_weights = {}
    states = set()
    actions = set()
    for line in lines[2:]:
        s_prev, action, s_next, weight = line.strip().split()
        s_prev = s_prev.strip('"')
        action = action.strip('"')
        s_next = s_next.strip('"')
        weight = float(weight)
        if action == "N":
            action = None
        transition_weights[(s_prev, action, s_next)] = weight
        states.update([s_prev, s_next])
        actions.add(action)
    # Normalize to get transition probabilities
    transition_probs = {}
    for s_prev in states:
        for action in actions:
            total_weight = sum(transition_weights.get((s_prev, action, s_next), default_weight)
                               for s_next in states)
            for s_next in states:
                weight = transition_weights.get((s_prev, action, s_next), default_weight)
                prob = weight / total_weight if total_weight > 0 else 0
                transition_probs[(s_prev, action, s_next)] = prob
    return transition_probs, states, actions

def read_state_observation_weights(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
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

def viterbi_algorithm(state_weights, transition_probs, observation_probs, observation_actions, states):
    T = len(observation_actions)
    states = list(states)
    N = len(states)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    # Initialize
    initial_observation, initial_action = observation_actions[0]
    for i, state in enumerate(states):
        delta[0, i] = state_weights.get(state, 0) * observation_probs.get((state, initial_observation), 0)
        psi[0, i] = 0
    # Normalize delta[0]
    if np.sum(delta[0]) > 0:
        delta[0] /= np.sum(delta[0])
    # Run Algo
    for t in range(1, T):
        observation, current_action = observation_actions[t]
        previous_action = observation_actions[t-1][1]
        if previous_action == "N":
            previous_action = None
        for j, s_j in enumerate(states):
            max_prob = 0
            max_state = 0
            for i, s_i in enumerate(states):
                trans_prob = transition_probs.get((s_i, previous_action, s_j), 0)
                prob = delta[t-1, i] * trans_prob
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            delta[t, j] = max_prob * observation_probs.get((s_j, observation), 0)
            psi[t, j] = max_state
        # Normalize delta[t]
        if np.sum(delta[t]) > 0:
            delta[t] /= np.sum(delta[t])
    # Terminate
    last_state = np.argmax(delta[T-1])
    states_sequence = [states[last_state]]
    # Backtrack
    for t in range(T-1, 0, -1):
        last_state = psi[t, last_state]
        states_sequence.insert(0, states[last_state])
    return states_sequence

def main():
    observation_actions = read_observation_actions('observation_actions.txt')
    state_weights = read_state_weights('state_weights.txt')
    transition_probs, states, actions = read_state_action_state_weights('state_action_state_weights.txt')
    observation_probs, observations_set = read_state_observation_weights('state_observation_weights.txt')
    states_sequence = viterbi_algorithm(
        state_weights, transition_probs, observation_probs, observation_actions, states)
    write_states('states.txt', states_sequence)

if __name__ == "__main__":
    main()
