
from __future__ import division
import math
from osm2networkx import *
import random
import pickle
import sys

# A heapq backed priority queue
import heapq

class PriorityQueue():
    """Implementation of a priority queue 
    to store nodes during search."""
    
    def __init__(self):
        self.queue = []
        self.current = 0    

    def next(self):
        if self.current >=len(self.queue):
            self.current
            raise StopIteration
    
        out = self.queue[self.current]
        self.current += 1

        return out

    def pop(self):
        return heapq.heappop(self.queue)
        
    def remove(self, nodeId):
        for i in range(self.size()):
            if self.queue[i][1] == nodeId:
                self.queue = self.queue[:i] + self.queue[i+1:]
                heapq.heapify(self.queue)
                break

    def __iter__(self):
        return self

    def __str__(self):
        return 'PQ:[%s]'%(', '.join([str(i) for i in self.queue]))

    def append(self, node):
        heapq.heappush(self.queue, node)
        
    def __contains__(self, key):
        self.current = 0
        return key in [n for v,n in self.queue]

    def __eq__(self, other):
        return self == other

    def size(self):
        return len(self.queue)
    
    def clear(self):
        self.queue = []
        
    def top(self):
        return self.queue[0]

    __next__ = next

def breadth_first_search(graph, start, goal):
    """Run a breadth-first search from start
    to goal and return the path."""
    if start == goal:
        return []
    
    frontier = [start]
    prev = {}
    
    def get_path():
        path = []
        prev_node = goal
        while prev_node != start:
            path = [prev_node] + path
            prev_node = prev[prev_node]
        path = [start] + path
        return path
    
    while len(frontier) > 0:
        node = frontier.pop(0)
        
        for neighbor in graph.neighbors(node):
            if neighbor not in graph.get_explored_nodes() and neighbor not in frontier:
                prev[neighbor] = node
                if neighbor == goal:
                    return get_path()
                frontier.append(neighbor)
    
    return get_path()

def uniform_cost_search(graph, start, goal):
    """Run uniform-cost search from start
    to goal and return the path"""
    
    if start == goal:
        return []
    
    costs ={}
    frontier = PriorityQueue()
    frontier.append((0, start))
    costs[start] = 0
    
    prev = {}
    
    while frontier.size() > 0:
        cost, node = frontier.pop()
        
        if node == goal:
            break
        
        for neighbor in graph.neighbors(node):
            if neighbor not in graph.get_explored_nodes():
                if neighbor not in frontier:
                    costs[neighbor] = cost + graph.edge[node][neighbor]['weight']
                    frontier.append((costs[neighbor], neighbor))
                    prev[neighbor] = node
                if neighbor in frontier and cost + graph.edge[node][neighbor]['weight'] < costs[neighbor]:
                    frontier.remove(neighbor)
                    costs[neighbor] = cost + graph.edge[node][neighbor]['weight']
                    frontier.append((costs[neighbor], neighbor))
                    prev[neighbor] = node

    path = []
    prev_node = goal
    while prev_node != start:
        path = [prev_node] + path
        prev_node = prev[prev_node]
    path = [start] + path
    return path

def null_heuristic(graph, v, goal ):
    return 0

def euclidean_dist_heuristic(graph, v, goal):
    """Return the Euclidean distance from
    node v to the goal."""
    
    if 'pos' in graph.node[v]:
        pos = 'pos'
    else:
        pos = 'position'
        
    pos1 = graph.node[v][pos]
    pos2 = graph.node[goal][pos]
    
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def a_star(graph, start, goal, heuristic):
    """Run A* search from the start to
    goal using the specified heuristic
    function, and return the final path."""
    
    if start == goal:
        return []
    
    hcosts = {}
    vcosts = {}
    frontier = PriorityQueue()
    hcost = heuristic(graph, start, goal)
    frontier.append((hcost, start))
    hcosts[start] = hcost
    vcosts[start] = 0
    
    prev = {}
    
    while frontier.size() > 0:
        fcost, node = frontier.pop()
        if node == goal:
            break
        
        for neighbor in graph.neighbors(node):
            if neighbor not in graph.get_explored_nodes():
                if neighbor not in frontier:
                    vcosts[neighbor] = vcosts[node] + graph.edge[node][neighbor]['weight']
                    hcosts[neighbor] = heuristic(graph, neighbor, goal)
                    frontier.append((vcosts[neighbor] + hcosts[neighbor], neighbor))
                    prev[neighbor] = node
                if neighbor in frontier and vcosts[node] + graph.edge[node][neighbor]['weight'] < vcosts[neighbor]:
                    frontier.remove(neighbor)
                    vcosts[neighbor] = vcosts[node] + graph.edge[node][neighbor]['weight']
                    hcosts[neighbor] = heuristic(graph, neighbor, goal)
                    frontier.append((vcosts[neighbor] + hcosts[neighbor], neighbor))
                    prev[neighbor] = node

    path = []
    prev_node = goal
    while prev_node != start:
        path = [prev_node] + path
        prev_node = prev[prev_node]
    path = [start] + path
    return path


def bidirectional_ucs(graph, start, goal):
    """Run bidirectional uniform-cost search
    between start and goal"""
    if start == goal:
        return []
    
    frt1 = PriorityQueue()
    frt2 = PriorityQueue()
    
    prev1 = {}
    prev2 = {}
    
    explored1 = set()
    explored2 = set()
    
    costs1 = {start: 0}
    costs2 = {goal: 0}
    
    frt1.append((0, start))
    frt2.append((0, goal))
    
    def get_path(prev, start, goal, reverse=False):
        path = []
        if start == goal:
            return path
        if not reverse:
            prev_node = goal
            while prev_node != start:
                path = [prev_node] + path
                prev_node = prev[prev_node]
            path = [start] + path
        else:
            next_node = start
            while next_node != goal:
                path.append(next_node)
                next_node = prev[next_node]
            path.append(goal)
        return path
        
    def combine_path(node):
        if node == start:
            return get_path(prev2, node, goal, True)
        if node == goal:
            return get_path(prev1, start, node)
        path1 = get_path(prev1, start, node)
        path2 = get_path(prev2, node, goal, True)
        return path1[:-1] + path2
        
    min_cost = float('Inf')
    path = []
    
    def add_neighbors(node, cost, frt, explored, other_explored, costs, other_costs, prev, min_cost, path):

        for ngbr in graph.neighbors(node):
            if ngbr not in explored:
                if ngbr not in frt:
                    costs[ngbr] = cost + graph.edge[node][ngbr]['weight']
                    frt.append((costs[ngbr], ngbr))
                    prev[ngbr] = node
                if ngbr in frt and cost + graph.edge[node][ngbr]['weight'] < costs[ngbr]:
                    frt.remove(ngbr)
                    costs[ngbr] = cost + graph.edge[node][ngbr]['weight']
                    frt.append((costs[ngbr], ngbr))
                    prev[ngbr] = node
                
                if ngbr in other_explored and costs[ngbr] + other_costs[ngbr] < min_cost:
                    min_cost = costs[ngbr] + other_costs[ngbr]
                    path = combine_path(ngbr)
        
        return min_cost, path
        
    while frt1.size() > 0 and frt2.size() > 0:
        cost1, node1 = frt1.pop()
        cost2, node2 = frt2.pop()
        
        if cost1 + cost2 >= min_cost:
            break
        
        explored1.add(node1)
        explored2.add(node2)
        
        min_cost, path = add_neighbors(node1, cost1, frt1, explored1, explored2, costs1, costs2, prev1, min_cost, path)
        min_cost, path = add_neighbors(node2, cost2, frt2, explored2, explored1, costs2, costs1, prev2, min_cost, path)

    return path


def bidirectional_a_star(graph, start, goal, heuristic):
    """Run bidirectional A* search between
    start and goal."""

    if start == goal:
        return []
    
    frt1 = PriorityQueue()
    frt2 = PriorityQueue()
    
    prev1 = {}
    prev2 = {}
    
    explored1 = set()
    explored2 = set()
    
    vcosts1 = {start: 0}
    vcosts2 = {goal: 0}
    
    tot = heuristic(graph, start, goal)
    
    def h(v, forward=True):
        fv, rv = heuristic(graph, v, goal), heuristic(graph, start, v)
        if forward:
            return 0.5 * (fv - rv) + 0.5 * tot
        else:
            return 0.5 * (rv - fv) + 0.5 * tot
    
    hcosts1 = {start: h(start)}
    hcosts2 = {goal: h(goal, False)}

    frt1.append((0, start))
    frt2.append((0, goal))
    
    def get_path(prev, start, goal, reverse=False):
        path = []
        if start == goal:
            return path
        if not reverse:
            prev_node = goal
            while prev_node != start:
                path = [prev_node] + path
                prev_node = prev[prev_node]
            path = [start] + path
        else:
            next_node = start
            while next_node != goal:
                path.append(next_node)
                next_node = prev[next_node]
            path.append(goal)
        return path
        
    def combine_path(node):
        if node == start:
            return get_path(prev2, node, goal, True)
        if node == goal:
            return get_path(prev1, start, node)
        path1 = get_path(prev1, start, node)
        path2 = get_path(prev2, node, goal, True)
        return path1[:-1] + path2
        
    min_cost = float('Inf')
    path = []
    
    def add_neighbors(node, frt, explored, other_explored, vcosts, other_vcosts, hcosts, prev, min_cost, path, forward=True):

        for ngbr in graph.neighbors(node):
            if ngbr not in explored:
                if ngbr not in frt:
                    vcosts[ngbr] = vcosts[node] + graph.edge[node][ngbr]['weight']
                    hcosts[ngbr] = h(ngbr, forward)
                    frt.append((vcosts[ngbr] + hcosts[ngbr], ngbr))
                    prev[ngbr] = node
                if ngbr in frt and vcosts[node] + graph.edge[node][ngbr]['weight'] < vcosts[ngbr]:
                    frt.remove(ngbr)
                    vcosts[ngbr] = vcosts[node] + graph.edge[node][ngbr]['weight']
                    hcosts[ngbr] = h(ngbr, forward)
                    frt.append((vcosts[ngbr] + hcosts[ngbr], ngbr))
                    prev[ngbr] = node
                    
                if ngbr in other_explored and vcosts[ngbr] + other_vcosts[ngbr] < min_cost:
                    min_cost = vcosts[ngbr] + other_vcosts[ngbr]
                    path = combine_path(ngbr)
        return min_cost, path
        
    while frt1.size() > 0 and frt2.size() > 0:
        fcost1, node1 = frt1.pop()
        fcost2, node2 = frt2.pop()
        
        if fcost1 + fcost2 >= min_cost + tot:
            break
        
        explored1.add(node1)
        explored2.add(node2)
        
        min_cost, path = add_neighbors(node1, frt1, explored1, explored2, vcosts1, vcosts2, hcosts1, prev1, min_cost, path)
        min_cost, path = add_neighbors(node2, frt2, explored2, explored1, vcosts2, vcosts1, hcosts2, prev2, min_cost, path, False)

    return path

def tridirectional_search(graph, goals):
    """Run tridirectional uniform-cost search
    between the goals and return the path."""
    
    for i in range(3):
        if goals[i % 3] == goals[(i + 1) % 3]:
            return []
    
    def sum_weight(g, path):
        pairs = zip(path, path[1:])
        return sum([g.get_edge_data(a, b)['weight'] for a, b in pairs])

    paths, frts, prev, explored, costs, min_costs = [], [], [], [], [], []
    for i in range(3):
        paths.append([])
        frts.append([PriorityQueue(), PriorityQueue()])
        frts[i][0].append((0, goals[i]))
        frts[i][1].append((0, goals[(i + 1) % 3]))
        prev.append([{}, {}])
        explored.append([set(), set()])
        costs.append([{goals[i]: 0}, {goals[(i + 1) % 3]: 0}])
        min_costs.append(float('Inf'))
    
    def get_path(prev, start, goal, reverse=False):
        path = []
        if start == goal:
            return path
        if not reverse:
            prev_node = goal
            while prev_node != start:
                path = [prev_node] + path
                prev_node = prev[prev_node]
            path = [start] + path
        else:
            next_node = start
            while next_node != goal:
                path.append(next_node)
                next_node = prev[next_node]
            path.append(goal)
        return path
        
    def combine_path(prev1, prev2, node, start, goal):
        if node == start:
            return get_path(prev2, node, goal, True)
        if node == goal:
            return get_path(prev1, start, node)
        path1 = get_path(prev1, start, node)
        path2 = get_path(prev2, node, goal, True)
        return path1[:-1] + path2
    
    def add_neighbors(node, cost, i, j):

        for ngbr in graph.neighbors(node):
            if ngbr not in explored[i][j]:
                if ngbr not in frts[i][j]:
                    costs[i][j][ngbr] = cost + graph.edge[node][ngbr]['weight']
                    frts[i][j].append((costs[i][j][ngbr], ngbr))
                    prev[i][j][ngbr] = node
                if ngbr in frts[i][j] and cost + graph.edge[node][ngbr]['weight'] < costs[i][j][ngbr]:
                    frts[i][j].remove(ngbr)
                    costs[i][j][ngbr] = cost + graph.edge[node][ngbr]['weight']
                    frts[i][j].append((costs[i][j][ngbr], ngbr))
                    prev[i][j][ngbr] = node
                    
                if ngbr in explored[i][1 - j] and costs[i][j][ngbr] + costs[i][j - 1][ngbr] < min_costs[i]:
                    min_costs[i] = costs[i][j][ngbr] + costs[i][j - 1][ngbr]
                    paths[i] = combine_path(prev[i][0], prev[i][1], ngbr, goals[i], goals[(i + 1) % 3])
    
    stop = [0, 0, 0]
    mask = True
    for i in range(3):
        for j in range(2):
            mask = mask and frts[i][j].size() > 0
    
    while mask and reduce(lambda x, y: x + y, stop) < 3:
            
        for i in range(3):
            if stop[i] == 0:

                cost0, node0 = frts[i][0].pop()
                cost1, node1 = frts[i][1].pop()
                
                if cost0 + cost1 >= min_costs[i]:
                    stop[i] = 1
                    continue

                explored[i][0].add(node0)
                explored[i][1].add(node1)

                add_neighbors(node0, cost0, i, 0)
                add_neighbors(node1, cost1, i, 1)
    
    sums = []
    for i in range(3):
        sums.append(sum_weight(graph, paths[i]) + sum_weight(graph, paths[(i + 1) % 3]))
    
    if sums[0] <= sums[1] and sums[0] <= sums[2]:
        return paths[0][:-1] + paths[1]
    elif sums[1] <= sums[0] and sums[1] <= sums[2]:
        return paths[1][:-1] + paths[2]
    else:
        return paths[2][:-1] + paths[0]


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    for i in range(3):
        if goals[i % 3] == goals[(i + 1) % 3]:
            return []
    
    def sum_weight(g, path):
        pairs = zip(path, path[1:])
        return sum([g.get_edge_data(a, b)['weight'] for a, b in pairs])

    def h(v, start, goal, forward=True):
        fv, rv, tot_ = heuristic(graph, v, goal), heuristic(graph, start, v), heuristic(graph, start, goal)
        if forward:
            return 0.5 * (fv - rv) + 0.5 * tot_
        else:
            return 0.5 * (rv - fv) + 0.5 * tot_
        
    paths, frts, prev, explored, vcosts, hcosts, min_costs, tot = [], [], [], [], [], [], [], []
    for i in range(3):
        paths.append([])
        frts.append([PriorityQueue(), PriorityQueue()])
        frts[i][0].append((0, goals[i]))
        frts[i][1].append((0, goals[(i + 1) % 3]))        
        prev.append([{}, {}])
        explored.append([set(), set()])
        vcosts.append([{goals[i]: 0}, {goals[(i + 1) % 3]: 0}])
        h1 = h(goals[i], goals[i], goals[(i + 1) % 3])
        h2 = h(goals[(i + 1) % 3], goals[i], goals[(i + 1) % 3], False)
        hcosts.append([{goals[i]: h1}, {goals[(i + 1) % 3]: h2}])
        tot.append(heuristic(graph, goals[i], goals[(i + 1) % 3]))
        min_costs.append(float('Inf'))
    
    def get_path(prev, start, goal, reverse=False):
        path = []
        if start == goal:
            return path
        if not reverse:
            prev_node = goal
            while prev_node != start:
                path = [prev_node] + path
                prev_node = prev[prev_node]
            path = [start] + path
        else:
            next_node = start
            while next_node != goal:
                path.append(next_node)
                next_node = prev[next_node]
            path.append(goal)
        return path
        
    def combine_path(prev1, prev2, node, start, goal):
        if node == start:
            return get_path(prev2, node, goal, True)
        if node == goal:
            return get_path(prev1, start, node)
        path1 = get_path(prev1, start, node)
        path2 = get_path(prev2, node, goal, True)
        return path1[:-1] + path2
    
    def add_neighbors(node, start, goal, frt, explored, vcosts, hcosts, prev, forward=True):
        
        for ngbr in graph.neighbors(node):
            if ngbr not in explored:
                if ngbr not in frt:
                    vcosts[ngbr] = vcosts[node] + graph.edge[node][ngbr]['weight']
                    hcosts[ngbr] = h(ngbr, start, goal, forward)
                    frt.append((vcosts[ngbr] + hcosts[ngbr], ngbr))
                    prev[ngbr] = node
                if ngbr in frt and vcosts[node] + graph.edge[node][ngbr]['weight'] < vcosts[ngbr]:
                    frt.remove(ngbr)
                    vcosts[ngbr] = vcosts[node] + graph.edge[node][ngbr]['weight']
                    hcosts[ngbr] = h(ngbr, start, goal, forward)
                    frt.append((vcosts[ngbr] + hcosts[ngbr], ngbr))
                    prev[ngbr] = node
    
    stop = [0, 0, 0]
    mask = True
    for i in range(3):
        for j in range(2):
            mask = mask and frts[i][j].size() > 0
    
    while mask and reduce(lambda x, y: x + y, stop) < 3:
            
        for i in range(3):
            if stop[i] == 0:
                
                fcost0, node0 = frts[i][0].pop()
                fcost1, node1 = frts[i][1].pop()
        
                if fcost0 + fcost1 >= min_costs[i] + tot[i]:
                    stop[i] = 1                                                   
                                                                                    
                explored[i][0].add(node0)
                explored[i][1].add(node1)

                if node0 in explored[i][1] and vcosts[i][0][node0] + vcosts[i][1][node0] < min_costs[i]:
                    min_costs[i] = vcosts[i][0][node0] + vcosts[i][1][node0]
                    paths[i] = combine_path(prev[i][0], prev[i][1], node0, goals[i], goals[(i + 1) % 3])

                if node1 in explored[i][0] and vcosts[i][1][node1] + vcosts[i][0][node1] < min_costs[i]:
                    min_costs[i] = vcosts[i][1][node1] + vcosts[i][0][node1]
                    paths[i] = combine_path(prev[i][0], prev[i][1], node1, goals[i], goals[(i + 1) % 3])

                add_neighbors(node0, goals[i], goals[(i + 1) % 3], frts[i][0], explored[i][0], vcosts[i][0], hcosts[i][0], prev[i][0])
                add_neighbors(node1, goals[i], goals[(i + 1) % 3], frts[i][1], explored[i][1], vcosts[i][1], hcosts[i][1], prev[i][1], False)
    
    sums = []
    for i in range(3):
        sums.append(sum_weight(graph, paths[i]) + sum_weight(graph, paths[(i + 1) % 3]))
    
    if sums[0] <= sums[1] and sums[0] <= sums[2]:
        return paths[0][:-1] + paths[1]
    elif sums[1] <= sums[0] and sums[1] <= sums[2]:
        return paths[1][:-1] + paths[2]
    else:
        return paths[2][:-1] + paths[0]

