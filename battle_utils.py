from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle
from poke_env.environment.effect import Effect
from poke_env.environment.pokemon_type import PokemonType

import numpy as np

from typing import Dict

def getPokemonStatus(pokemon: Pokemon):
    status = pokemon.status
    if status == 1: # Burned
        return [0., 0., 0., 0., 1.]
    elif status == 3: # Frozen
        return [0., 0., 0., 1., 0.]
    elif status == 4: # Paralyzed
        return [1., 0., 0., 0., 0.]
    elif status == 5: # Poisoned
        return [0., 1., 0., 0., 0.]
    elif status == 6: # Asleep
        return [0., 0., 1., 0., 0.]
    elif status == 7: # Badly poisoned
        return [0., 2., 0., 0., 0.]
    else: # No status
        return [0, 0, 0, 0, 0]

def getPokemonConfusion(pokemon: Pokemon):
    if len(pokemon.effects.intersection([6])) == 1:
        return [1]
    return [0]

def getPokemonFainted(pokemon: Pokemon):
    status = pokemon.status
    if status == 2: # Fainted
        return [1]
    return [0]

def get_stat_multiplier(poke, stat): # TODO: TEST THIS
    value = poke._boosts.get(stat)
    if value < 0:
        return 1 / (1 + 0.5 * -value)
    else: # value >= 0:
        return 1 + 0.5 * value

def getTeamFeatures(team: Dict, active_poke: Pokemon):
    state = np.array([])

    level = active_poke.level
    # hp, atk, def, spa, spd, spe
    speed = active_poke.base_stats.get("spe") * get_stat_multiplier(active_poke, "spe")
    # speed = ((2 * speed + 31) * level / 100 + 5) * 1.1 * 1.1 + 200
    state = np.append(state, [active_poke.current_hp_fraction])
    state = np.append(state, [speed])
    state = np.append(state, getPokemonStatus(active_poke))
    state = np.append(state, getPokemonConfusion(active_poke))
    
    for key in team:
        if team[key].active:
            continue
        state = np.append(state, [team[key].current_hp_fraction])
        state = np.append(state, getPokemonStatus(team[key]))
        state = np.append(state, getPokemonFainted(team[key]))

    missing = 6 - len(team)
    for i in range(missing):
        state = np.append(state, [1, 0, 0, 0, 0, 0, 0])
    return state

def get_type_multiplier(move_type: PokemonType, pokemon: Pokemon):
    if pokemon.type_2:
        return move_type.damage_multiplier(pokemon.type_1, pokemon.type_2)
    else:
        return move_type.damage_multiplier(pokemon.type_1)

def getMovesInfo(own_poke: Pokemon, opp_poke: Pokemon, battle):
    state = np.array([])
    moves = own_poke.moves

    for key in battle.team:
        if own_poke == battle.team[key]:
            new_moves = {}
            for move in battle.available_moves:
                for key2 in moves:
                    if move == moves[key2]:
                        new_moves[key2] = move
                        break
            moves = new_moves
            break

    if own_poke.species == "Ditto" and len(moves) > 4:
        moves.pop('transform')

    for key in moves:
        damage_mult = get_type_multiplier(moves[key].type, opp_poke)
        if (moves[key].category.name == "PHYSICAL"):
            # print("Effectiveness: " + str(damage_mult))            
            power = (damage_mult * moves[key].base_power * own_poke.base_stats.get("atk") * get_stat_multiplier(own_poke, "atk")
                * moves[key].accuracy / (opp_poke.base_stats.get("def") * get_stat_multiplier(own_poke, "def")))
            secondary = 0.0
            if (moves[key].secondary != None):
                secondary = moves[key].secondary.get("chance") / 100
            state = np.append(state, [power, secondary, moves[key].priority])
        
        elif (moves[key].category.name == "SPECIAL"):
            # print("Effectiveness: " + str(damage_mult))     
            power = (damage_mult * moves[key].base_power * own_poke.base_stats.get("spa") * get_stat_multiplier(own_poke, "spa")
                * moves[key].accuracy / (opp_poke.base_stats.get("spd") * get_stat_multiplier(own_poke, "spd")))
            secondary = 0.0
            if (moves[key].secondary != None):
                secondary = moves[key].secondary.get("chance") / 100
            state = np.append(state, [power, secondary, moves[key].priority])
        
        elif (moves[key].category.name == "STATUS"):
            state = np.append(state, [0, moves[key].accuracy, moves[key].priority])

    missing = 4 - len(moves)
    for i in range(missing):
        state = np.append(state, [0, 0, 0])

    return state
        