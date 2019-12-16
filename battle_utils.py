from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle

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