import asyncio
import time
import battle_utils as bu

import numpy as np

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.trainable_player import TrainablePlayer
from poke_env.player.utils import cross_evaluate
from poke_env.environment.battle import Battle
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration, ServerConfiguration

from typing import Dict, Optional

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from tensorflow.python.keras import backend as K
import tensorflow as tf

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
    def __init__(
        self,
        player_configuration: PlayerConfiguration,
        *,
        avatar: Optional[int] = None,
        battle_format: str,
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        model=None,
        server_configuration: ServerConfiguration,
        start_listening: bool = True,
    ) -> None:
    
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)
        self.graph = tf.compat.v1.get_default_graph()
        K.set_session(self.session)
        with self.graph.as_default():
            with self.session.as_default():
                self.graph.finalize()

        super(PokeAgent, self).__init__(
            player_configuration=player_configuration, 
            avatar=avatar, 
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=max_concurrent_battles,
            model=model, 
            server_configuration=server_configuration,
            start_listening=start_listening
        )

    @staticmethod
    def init_model():
        model = Sequential()
        model.add(Dense(110, activation="relu", input_shape=(110,)))
        model.add(Dense(110, activation="relu"))
        model.add(Dense(9, activation="softmax"))
        model._make_predict_function()
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return model

    def action_to_move(self, action, battle: Battle):
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def battle_to_state(self, battle: Battle):
        state = np.array([])
        # ----- Add own team info ----- #
        state = np.append(state, bu.getTeamFeatures(battle.team, battle.active_pokemon))
        # ----- Add opponent team info ----- #
        state = np.append(state, bu.getTeamFeatures(battle.opponent_team, battle.opponent_active_pokemon))
        # ----- Add own moves info ----- #
        state = np.append(state, bu.getMovesInfo(battle.active_pokemon, battle.opponent_active_pokemon))
        # ----- Add opponent moves info ----- #
        state = np.append(state, bu.getMovesInfo(battle.opponent_active_pokemon, battle.active_pokemon))
        return state
    
    def state_to_action(self, state: np.array, battle: Battle):
        print("Antes do predict")
        with self.graph.as_default():
            with self.session.as_default():
                actions = self.model.predict(state)
        print("----------- ACTIONS HERE -----------")
        print(actions)
        pass

    def replay(self, battle_history: Dict):
        # print(battle_history)
        pass

async def main():
    start = time.time()

    # We define two player configurations.
    player_1_configuration = PlayerConfiguration("Agent player", None)
    player_2_configuration = PlayerConfiguration("Random player", None)
    
    # We create the corresponding players.
    agent_player = PokeAgent(
        player_configuration=player_1_configuration,
        battle_format="gen7letsgorandombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    random_player = RandomPlayer(
        player_configuration=player_2_configuration,
        battle_format="gen7letsgorandombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    await agent_player.train_against(random_player, 1)

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