from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle
from poke_env.environment.effect import Effect

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

def getTeamFeatures(team: Dict, active_poke: Pokemon):
    state = np.array([])

    level = active_poke.level
    # hp, atk, def, spa, spd, spe
    speed = active_poke.base_stats.get("spe")
    speed = ((2 * speed + 31) * level / 100 + 5) * 1.1 * 1.1 + 200
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
    return state