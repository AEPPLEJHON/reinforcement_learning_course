# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.pacman.gamestate import GameState

east = """ 
%%%%%%%%
% P   .%
%%%%%%%% """ 

east2 = """
%%%%%%%%
%    P.%
%%%%%%%% """

SS2tiny = """
%%%%%%
%.P  %
% GG.%
%%%%%%
"""

SS0tiny = """
%%%%%%
%.P  %
%   .%
%%%%%%
"""

SS1tiny = """
%%%%%%
%.P  %
%  G.%
%%%%%%
"""

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

# TODO: 30 lines missing.
#raise NotImplementedError("Put your own code here")

class ShortestPathDPModel(DPModel): #structure copied from exercise 2
    def __init__(self, x0: GameState, N: int):
        super().__init__(N=N)
        self._S = get_future_states(x0, N)

    def S(self, k: int):
        return self._S[k]

    def A(self, x: GameState, k: int):
        if x.is_won():
            return {"Stop"}
        return set(x.A())

    def Pw(self, x: GameState, u, k: int):
        if x.is_won():
            return {x: 1.0}
        return p_next(x, u)

    def f(self, x, u, w, k: int):
        return w

    def g(self, x: GameState, u, w: GameState, k: int) -> float:
        return 0.0 if x.is_won() else 1.0 #each step costs 1

    def gN(self, x: GameState) -> float:
        return 0.0 if x.is_won() else 1e9 #punish loss heavily


def p_next(x : GameState, u: str): 
    """ Given the agent is in GameState x and takes action u, the game will transition to a new state xp.
    The state xp will be random when there are ghosts. This function should return a dictionary of the form

    {..., xp: p, ...}

    of all possible next states xp and their probability -- you need to compute this probability.

    Hints:
        * In the above, xp should be a GameState, and p will be a float. These are generated using the functions in the GameState x.
        * Start simple (zero ghosts). Then make it work with one ghosts, and then finally with any number of ghosts.
        * Remember the ghosts move at random. I.e. if a ghost has 3 available actions, it will choose one with probability 1/3
        * The slightly tricky part is that when there are multiple ghosts, different actions by the individual ghosts may lead to the same final state
        * Check the probabilities sum to 1. This will be your main way of debugging your code and catching issues relating to the previous point.
    """
    xp = x.f(u)


    #handle edge cases:
    if xp.is_won() or xp.is_lost():
        return {xp: 1.0}

    if xp.players() == 1:
        return {xp: 1.0}
    

    states = {xp: 1.0}

    for _ in range(xp.players() - 1): #loop for each ghost
        new_states = {}

        for x, prob in states.items(): #the states are children of a parent state (prev. ghost)
            g_actions = list(x.A())
            p = prob / len(g_actions) # weight by probability of parent state

            for v in g_actions:
                xk1 = x.f(v)
                new_states[xk1] = new_states.get(xk1, 0.0) + p

        states = new_states

    return states

def go_east(map): 
    """ Given a map-string map (see examples in the top of this file) that can be solved by only going east, this will return
    a list of states Pacman will traverse. The list it returns should therefore be of the form:

    [s0, s1, s2, ..., sn]

    where each sk is a GameState object, the first element s0 is the start-configuration (corresponding to that in the Map),
    and the last configuration sn is a won GameState obtained by going east.

    Note this function should work independently of the number of required east-actions.

    Hints:
        * Use the GymPacmanEnvironment class. The report description will contain information about how to set it up, as will pacman_demo.py
        * Use this environment to get the first GameState, then use the recommended functions to go east
    """
    env = PacmanEnvironment(layout_str=map)
    x, _ = env.reset() #initialize env
    S = [x] #create list
    while not x.is_won(): #go east until victory state
        x = x.f("East")
        S.append(x)
    env.close()
    return S

def get_future_states(x, N): 
    # TODO: 4 lines missing.
    #raise NotImplementedError("return a list-of-list of future states [S_0, ... ,S_N].
    #Each S_k is a state space, i.e. a list of GameState objects.")

    state_spaces = [[x]]

    for k in range(N):
        S = set()
        for x in state_spaces[k]:
            for u in x.A():
                for xp in p_next(x,u).keys():
                    S.add(xp)
        state_spaces.append(list(S))

    return state_spaces

def win_probability(map, N=10): 
    """ Assuming you get a reward of -1 on wining (and otherwise zero), the win probability is -J_pi(x_0). """
    # TODO: 5 lines missing.
    raise NotImplementedError("Return the chance of winning the given map within N steps or less.")
    return win_probability

def shortest_path(map, N=10): 
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    # TODO: 4 lines missing.
    #raise NotImplementedError("Return the cost of the shortest path, the list of actions taken, and the list of states.")
    
    env = PacmanEnvironment(layout_str=map)
    x0, _ = env.reset()

    model = ShortestPathDPModel(x0, N)
    _, pi = DP_stochastic(model)

    actions = []
    states = [x0]
    x = x0

    for k in range(N):  
        if x.is_won():
            break
        u = pi[k][x]
        actions.append(u)
        pw = model.Pw(x, u, k)
        w = next(iter(pw))
        x = model.f(x, u, w, k)
        states.append(x)

    env.close()
    return actions, states


def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    # This function contains examples of calling your functions. However, you should use unitgrade to verify correctness.

    ## Problem 7: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    ## Problem 8: try the p_next function for a few empty environments. Does the result look sensible?
    x, _ = PacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map: 'east'")
    print(x)
    print(p_next(x, action))  # use str(state) to get a nicer representation.

    print(f"Transitions when taking action {action} in map: 'east2'")
    x, _ = PacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Problem 9
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x, _ = PacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]: # Print all elements in S_1.
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Problem 10
    N = 20  # Planning horizon
    action, states = shortest_path(east, N)
    print("east: Optimal action sequence:", action)

    action, states = shortest_path(datadiscs, N)
    print("datadiscs: Optimal action sequence:", action)

    action, states = shortest_path(SS0tiny, N)
    print("SS0tiny: Optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this tends to increase with planning depth
    wp = []
    for n in range(10):
        wp.append(win_probability(SS1tiny, N=n))
    print(wp)
    print("One ghost:", win_probability(SS1tiny, N=12))


def two_ghosts():
    # Win probability when planning using two ghosts
    print("Two ghosts:", win_probability(SS2tiny, N=12))

if __name__ == "__main__":
    no_ghosts()
    one_ghost()
    two_ghosts()
