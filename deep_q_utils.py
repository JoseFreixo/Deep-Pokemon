import numpy as np

def get_alive_own_pokemon(state):
    n_alive = 0
    if state[0] > 0:
        n_alive += 1
    for i in range(5):
        if state[8 + 7 * i] > 0:
            n_alive += 1
    return n_alive

def get_alive_opp_pokemon(state):
    n_alive = 0
    if state[43] > 0:
        n_alive += 1
    for i in range(5):
        if state[51 + 7 * i] > 0:
            n_alive += 1
    return n_alive

def damage_taken(state, new_state):
    return state[0] - new_state[0]

def damage_dealt(state, new_state):
    return state[43] - new_state[43]

##########################################################################
#                                                                        #
#                            REWARD FUNCTION                             #
#                                                                        #
##########################################################################
def get_reward(state, new_state):
    if np.array_equal(state, new_state):
        print("SAME STATE")
        return -1
    own_lost_poke = get_alive_own_pokemon(state) - get_alive_own_pokemon(new_state)
    opp_lost_poke = (get_alive_opp_pokemon(state) - get_alive_opp_pokemon(new_state)) * 1.5
    return (opp_lost_poke - own_lost_poke + damage_dealt(state, new_state) * 1.5 - damage_taken(state, new_state))