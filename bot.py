import asyncio
import time
import battle_utils

import numpy as np

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.trainable_player import TrainablePlayer
from poke_env.player.utils import cross_evaluate
from poke_env.environment.battle import Battle
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration

from typing import Dict

from keras.models import Sequential
from keras.layers import Dense

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

class PokeAgent(TrainablePlayer):
    def init_model(self):
        self.model = Sequential()
        self.model.add(Dense(94, input_dim=94))
        self.model.add(Dense(94))
        self.model.add(Dense(94))
        self.model.add(Dense(9))

    def action_to_move(self, action, battle: Battle):
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def battle_to_state(self, battle: Battle):
        level = battle.active_pokemon.level
        # hp, atk, def, spa, spd, spe
        speed = battle.active_pokemon.base_stats.get("spe")
        speed = ((2 * speed + 31) * level / 100 + 5) * 1.1 * 1.1 + 200
        
        state = np.array([])
        state = np.append(state, [battle.active_pokemon.current_hp_fraction])
        state = np.append(state, [speed])
        state = np.append(state, battle_utils.getPokemonStatus(battle.active_pokemon))
        return None
        

    def state_to_action(self, state: np.array, battle: Battle):
        pass

    def replay(self, battle_history: Dict):
        print(battle_history)

async def main():
    start = time.time()

    # We define two player configurations.
    player_1_configuration = PlayerConfiguration("Agent player", None)
    player_2_configuration = PlayerConfiguration("Max damage player", None)

    # We create the corresponding players.
    agent_player = PokeAgent(
        player_configuration=player_1_configuration,
        battle_format="gen7letsgorandombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    max_damage_player = MaxDamagePlayer(
        player_configuration=player_2_configuration,
        battle_format="gen7letsgorandombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    await agent_player.train_against(max_damage_player, 1)

    # Now, let's evaluate our player
    # cross_evaluation = await cross_evaluate(
    #     [agent_player, max_damage_player], n_challenges=1
    # )

    # print(
    #     "Max damage player won %d / 100 battles [this took %f seconds]"
    #     % (
    #         cross_evaluation[max_damage_player.username][agent_player.username] * 1,
    #         time.time() - start,
    #     )
    # )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())