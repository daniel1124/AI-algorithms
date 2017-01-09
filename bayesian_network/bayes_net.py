import sys
if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine
from Inference import EnumerationEngine
import random
import math
from random import randint,uniform


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    t_node = BayesNode(0, 2, name='temperature')
    fg_node = BayesNode(1, 2, name='faulty gauge')
    g_node = BayesNode(2, 2, name='gauge')
    fa_node = BayesNode(3, 2, name='faulty alarm')
    a_node = BayesNode(4, 2, name='alarm')
    
    t_node.add_child(fg_node)
    fg_node.add_parent(t_node)
    t_node.add_child(g_node)
    g_node.add_parent(t_node)
    fg_node.add_child(g_node)
    g_node.add_parent(fg_node)
    g_node.add_child(a_node)
    a_node.add_parent(g_node)
    fa_node.add_child(a_node)
    a_node.add_parent(fa_node)
    
    nodes = [t_node, fg_node, g_node, fa_node, a_node]
    
    return BayesNet(nodes)


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""
    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    T_dist = DiscreteDistribution(T_node)
    index = T_dist.generate_index([],[])
    T_dist[index] = [0.8, 0.2]
    T_node.set_dist(T_dist)
    
    F_A_dist = DiscreteDistribution(F_A_node)
    index = F_A_dist.generate_index([],[])
    F_A_dist[index] = [0.8, 0.2]
    F_A_node.set_dist(F_A_dist)
    
    F_G_dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)   #Note the order of G_node, A_node
    F_G_dist[0, :] = [0.95, 0.05]
    F_G_dist[1, :] = [0.2, 0.8]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node], table=F_G_dist)
    F_G_node.set_dist(F_G_distribution)
    
    G_dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    G_dist[0, 0, :] = [0.95, 0.05]
    G_dist[0, 1, :] = [0.2, 0.8]
    G_dist[1, 0, :] = [0.05, 0.95]
    G_dist[1, 1, :] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=G_dist)
    G_node.set_dist(G_distribution)
    
    A_dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    A_dist[0, 0, :] = [0.9, 0.1]
    A_dist[0, 1, :] = [0.55, 0.45]
    A_dist[1, 0, :] = [0.1, 0.9]
    A_dist[1, 1, :] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=A_dist)
    A_node.set_dist(A_distribution)
    
    return bayes_net


def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    A_node = bayes_net.get_node_by_name("alarm")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings],range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    G_node = bayes_net.get_node_by_name("gauge")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot],range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the probability of the 
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""

    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    engine.evidence[A_node] = True
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot],range(Q.nDims))
    temp_prob = Q[index]
    
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    
    a_node = BayesNode(0, 4, name='A')
    b_node = BayesNode(1, 4, name='B')
    c_node = BayesNode(2, 4, name='C')
    ab_node = BayesNode(3, 3, name='AvB')
    bc_node = BayesNode(4, 3, name='BvC')
    ca_node = BayesNode(5, 3, name='CvA')
    
    a_node.add_child(ab_node)
    b_node.add_child(ab_node)
    ab_node.add_parent(a_node)
    ab_node.add_parent(b_node)
    
    c_node.add_child(bc_node)
    b_node.add_child(bc_node)
    bc_node.add_parent(c_node)
    bc_node.add_parent(b_node)
    
    a_node.add_child(ca_node)
    c_node.add_child(ca_node)
    ca_node.add_parent(a_node)
    ca_node.add_parent(c_node)
    
    a_dist = DiscreteDistribution(a_node)
    index = a_dist.generate_index([], [])
    a_dist[index] = [0.15, 0.45, 0.3, 0.1]
    a_node.set_dist(a_dist)
    
    b_dist = DiscreteDistribution(b_node)
    index = b_dist.generate_index([], [])
    b_dist[index] = [0.15, 0.45, 0.3, 0.1]
    b_node.set_dist(b_dist)
    
    c_dist = DiscreteDistribution(c_node)
    index = c_dist.generate_index([],[])
    c_dist[index] = [0.15, 0.45, 0.3, 0.1]
    c_node.set_dist(c_dist)

    prob = {0: [0.1, 0.1, 0.8],
            1: [0.2, 0.6, 0.2],
            2: [0.15, 0.75, 0.1],
            3: [0.05, 0.9, 0.05],
            -1: [0.6, 0.2, 0.2],
            -2: [0.75, 0.15, 0.1],
            -3: [0.9, 0.05, 0.05]}
    
    ab_dist = zeros([a_node.size(), b_node.size(), ab_node.size()], dtype=float32)
    for i in range(a_node.size()):
        for j in range(b_node.size()):
            ab_dist[i, j, :] = prob[j - i]
    ab_distribution = ConditionalDiscreteDistribution(nodes=[a_node, b_node, ab_node], table=ab_dist)
    ab_node.set_dist(ab_distribution)
    
    bc_dist = zeros([b_node.size(), c_node.size(), bc_node.size()], dtype=float32)
    for i in range(b_node.size()):
        for j in range(c_node.size()):
            bc_dist[i, j, :] = prob[j - i]
    bc_distribution = ConditionalDiscreteDistribution(nodes=[b_node, c_node, bc_node], table=bc_dist)
    bc_node.set_dist(bc_distribution)
    
    ca_dist = zeros([c_node.size(), a_node.size(), ca_node.size()], dtype=float32)
    for i in range(c_node.size()):
        for j in range(a_node.size()):
            ca_dist[i, j, :] = prob[j - i]
    ca_distribution = ConditionalDiscreteDistribution(nodes=[c_node, a_node, ca_node], table=ca_dist)
    ca_node.set_dist(ca_distribution)
    
    nodes = [a_node, b_node, c_node, ab_node, bc_node, ca_node]

    return BayesNet(nodes)


def calculate_posterior(games_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    
    bc_node = games_net.get_node_by_name("BvC")
    ab_node = games_net.get_node_by_name("AvB")
    ac_node = games_net.get_node_by_name("CvA")
    
    # b_node = games_net.get_node_by_name("B")
    # c_node = games_net.get_node_by_name("C")
    
    engine = EnumerationEngine(games_net)
    
    # engine.evidence[b_node] = 2
    # engine.evidence[c_node] = 0
    
    engine.evidence[ab_node] = 0
    engine.evidence[ac_node] = 2

    Q = engine.marginal(bc_node)[0]
    idx = Q.generate_index([], [])
    posterior = list(Q[idx])
        
    return posterior


def Gibbs_sampler(games_net, initial_value, number_of_teams=5, evidence=None):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_value is a list of length 10 where: 
    index 0-4: represent skills of teams T1, .. ,T5 (values lie in [0,3] inclusive)
    index 5-9: represent results of matches T1vT2,...,T5vT1 (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 10. 
    Return the sample as a tuple.
    """
    global idx
    
    A = games_net.get_node_by_name("A")      
    AvB = games_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    n = number_of_teams
    l1 = A.size()
    l2 = AvB.size()
        
    if initial_value is None or len(initial_value) == 0:
        initial_value = [random.randint(0, l1 - 1) for i in range(n)] + \
            [random.randint(0, l2 - 1) for i in range(n, 2 * n)]
        return tuple(initial_value)

    if evidence is None or len(evidence) != 4:
        idx = random.randint(0, 2 * n - 1)
    else:
        initial_value = initial_value[:5] + evidence + initial_value[-1:]
        cand_set = range(5) + [9]
        idx = random.sample(cand_set, 1)[0]
        
    prob = []
    if idx < n:
        for i in range(l1):
            prob.append(team_table[i] * 
                        match_table[i, initial_value[(idx + 1) % n], initial_value[idx + n]] * 
                        match_table[initial_value[(idx - 1) % n], i, initial_value[(idx - 1) % n + n]])
    else:
        prob = match_table[initial_value[idx - n], initial_value[(idx + 1 - n) % n], :]

    prob = [p / sum(prob) for p in prob]
    
    new_value = list(initial_value)
    x = random.random()
    cum = 0
    for i in range(len(prob)):
        cum += prob[i]
        if x < cum:
            new_value[idx] = i
            break
    sample = tuple(new_value)
    return sample

import math
def converge_count_Gibbs(bayes_net, initial_state, match_results, number_of_teams=5):
    """Calculate number of iterations for Gibbs sampling to converge to a stationary distribution. 
    And return the likelihoods for the last match. """
    count = 0
    n = 100
    delta = 1e-2
    burn = 1000
    interval = 10
    kld = float('Inf')
    total = 100000

    def kl_divergence(p1, p2):
        res = float('Inf')
        if len(p1) != len(p2):
        	return res

        for i in range(len(p1)):
            if p2[i] < 1e-5 and p1[i] < 1e-5:
                continue
            elif p2[i] < 1e-5:
                return float('Inf') 
            elif p1[i] < 1e-5:
                return float('-Inf')
            else:
                res += p1[i] * math.log(p1[i] / p2[i])

        return res
    
    # burn-in
    for i in range(burn):
        sample = Gibbs_sampler(bayes_net, initial_state, number_of_teams, match_results)
        initial_state = list(sample)
    count += burn
    posterior = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    
    res = [0, 0, 0]
    while abs(kld) > delta and count < total:
        res_ = [0, 0, 0]
        for i in range(n * interval):
            sample = Gibbs_sampler(bayes_net, initial_state, number_of_teams, match_results)
            initial_state = list(sample)
            if (i + 1) % interval == 0:
                res[sample[-1]] += 1
                res_[sample[-1]] += 1
        count += n
        
        new_posterior = [1.0 * res_[i] / sum(res_) for i in range(len(res_))]
        kld = kl_divergence(posterior, new_posterior)

        posterior = [1.0 * res[i] / sum(res) for i in range(len(res))]

    return count, posterior


def MH_sampler(games_net, initial_value, n=5, evidence=None):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_value is a list of length 10 where: 
    index 0-4: represent skills of teams T1, .. ,T5 (values lie in [0,3] inclusive)
    index 5-9: represent results of matches T1vT2,...,T5vT1 (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 10. 
    """
    A = games_net.get_node_by_name("A")      
    AvB = games_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    l1 = A.size()
    l2 = AvB.size()
        
    if initial_value is None:
        initial_value = [random.randint(0, l1 - 1) for i in range(n)] + \
            [random.randint(0, l2 - 1) for i in range(n, 2 * n)]
        return tuple(initial_value)
    
    if evidence is not None and len(evidence) == 4:
        initial_value = initial_value[:5] + evidence + initial_value[-1:]
        cand_set = range(5) + [9]
    else:
        cand_set = range(len(initial_value))
        
    new_value = list(initial_value)
    for i in cand_set:
        if i < n:
            maxn = l1
        else:
            maxn = l2
        delta = random.randint(0, 1) * 2 - 1
        new_value[i] = (initial_value[i] + delta) % maxn

    p_old, p_new = 1, 1
    for i in range(n):
        p_old *= team_table[initial_value[i]]
        p_new *= team_table[new_value[i]]
    for i in range(n, 2 * n):
        p_old *= match_table[initial_value[i - n], initial_value[(i + 1 - n) % n], initial_value[i]]
        p_new *= match_table[new_value[i - n], new_value[(i + 1 - n) % n], new_value[i]]
    
    alpha = p_new / p_old
    
    u = random.random()
    
    if u < alpha:
        sample = tuple(new_value)
    else:
        sample = tuple(initial_value)
        
    return sample


def converge_count_MH(bayes_net, initial_state, match_results, number_of_teams=5):
    """Calculate number of iterations for MH sampling to converge to any stationary distribution. 
    And return the likelihoods for the last match. """
    count=0
    n = 100
    delta = 1e-2
    burn = 1000
    interval = 10
    total = 100000
    kld = float('Inf')
    
    def kl_divergence(p1, p2):
        res = float('Inf')
        if len(p1) != len(p2):
        	return res

        for i in range(len(p1)):
            if p2[i] < 1e-5 and p1[i] < 1e-5 and count < total:
                continue
            elif p2[i] < 1e-5:
                return float('Inf') 
            elif p1[i] < 1e-5:
                return float('-Inf')
            else:
                res += p1[i] * math.log(p1[i] / p2[i])

        return res

    # burn-in
    for i in range(burn):
        sample = MH_sampler(bayes_net, initial_state, number_of_teams, match_results)
        initial_state = list(sample)
    count += burn
    posterior = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    
    res = [0, 0, 0]
    while abs(kld) > delta and count < total:
        res_ = [0, 0, 0]
        for i in range(n * interval):
            sample = MH_sampler(bayes_net, initial_state, number_of_teams, match_results)
            initial_state = list(sample)
            if (i + 1) % interval == 0:
                res[sample[-1]] += 1
                res_[sample[-1]] += 1
        count += n
        
        new_posterior = [1.0 * res_[i] / sum(res_) for i in range(len(res_))]
        kld = kl_divergence(posterior, new_posterior)
        
        posterior = [1.0 * res[i] / sum(res) for i in range(len(res))]
    
    return count, posterior


def compare_sampling(bayes_net,initial_state, match_results, n):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge 
    to the provided posterior."""
    Gibbs_count, _ = converge_count_Gibbs(bayes_net, initial_state, match_results, n)
    MH_count, _ = converge_count_MH(bayes_net, initial_state, match_results, n)
    
    return Gibbs_count, MH_count
