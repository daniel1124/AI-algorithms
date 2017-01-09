
import numpy as np

def viterbi(evidence_vector, prior, states, evidence_variables, transition_probs, emission_probs):
    """
        This method takes as input the following:
        evidence_vector: A list of dictionaries mapping evidence variables to their values
        prior: A dictionary corresponding to the prior distribution over states
        states: A list of all possible system states
        evidence_variables: A list of all valid evidence variables
        transition_probs: A dictionary mapping states onto dictionaries mapping states onto probabilities
        emission_probs: A dictionary mapping states onto dictionaries mapping evidence variables onto 
                        probabilities for their possible values
        This method returns:
            A list of states that is the most likely sequence of states explaining the evidence
    """
    e = evidence_variables
    p_trellis = {e: [dict(prior)]}
    
    states = list(states)
    states.remove('End')
    predecessor = []
    
    for i in range(len(evidence_vector)):
        o = evidence_vector[i][e]
        p_step = {}
        pred = {}
        
        for new in states:
            probs = [emission_probs[new][e][o] * transition_probs[old][new] * p_trellis[e][i][old] for old in states]
            p_step[new] = np.max(probs)
            pred[new] = states[np.argmax(probs)]
            
        predecessor.append(pred)
        p_trellis[e].append(p_step)
    
    prev = 'Low'
    seq = [prev]
    for i in range(len(predecessor) - 1, 0, -1):
        prev = predecessor[i][prev]
        seq.append(prev)
    
    seq.reverse()
    return seq

# Umbrella World & Forward-backward
def forward_backward(evidence_vector, prior, states, evidence_variables, transition_probs, emission_probs):
    
    """
        This method takes as input the following:
        evidence_vector: A list of dictionaries mapping evidence variables to their values at each time step
        prior: A dictionary corresponding to the prior distribution over states
        states: A list of all possible system states
        evidence_variables: A list of all valid evidence variables
        transition_probs: A dictionary mapping states onto dictionaries mapping states onto probabilities
        emission_probs: A dictionary mapping states onto dictionaries mapping evidence variables onto 
                    a list of probabilities for their possible values
                    
        This method returns:
            A list of dictionaries giving the distribution of possible states at each time step
    """
    
    prodf = lambda x, y: x * y
    
    # forward
    alpha = [dict(prior)]
    
    for o in evidence_vector:
        new_a = {}
        for s in states:
            new_a[s] = sum([reduce(prodf, [emission_probs[s][e][o[e]] for e in evidence_variables]) * 
                            alpha[-1][prev_s] * transition_probs[prev_s][s] for prev_s in states])   
        alpha.append(new_a)
    
    alpha = alpha[1:]

    # backward
    beta = [{}]
    for s in states:
        beta[0][s] = 1
    
    posterior = []
    for i, o in enumerate(reversed(evidence_vector)):
        new_b = {}
        
        posterior.insert(0, {})
        
        for s in states:
            new_b[s] = sum([reduce(prodf, [emission_probs[next_s][e][o[e]] for e in evidence_variables]) * 
                            beta[0][next_s] * transition_probs[s][next_s] for next_s in states])
            
        scale = sum([new_b[s] * alpha[-i - 1][s] for s in states])
        
        for s in states:
            
            posterior[0][s] = new_b[s] * alpha[-i - 1][s] / scale
        
        beta.insert(0, new_b)
    
    return posterior

