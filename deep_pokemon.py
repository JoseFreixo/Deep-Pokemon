import asyncio
import time
import battle_utils as bu

import sys
import os.path

import numpy as np

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.trainable_player import TrainablePlayer
from poke_env.player.utils import cross_evaluate
from poke_env.environment.battle import Battle
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration, ServerConfiguration

from typing import Dict, Optional

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from tensorflow.python.keras import backend as K
import tensorflow as tf

from multiprocessing import Pool, Pipe

class AggressivePlayer(Player):

    def choose_best_safe_switch(self, battle):
        poke_id = 0
        best_value = 0
        for i in range(len(battle.available_switches)):
            moves = self.dict_moves_to_list(battle.available_switches[i].moves)
            best_move = max(moves, key=lambda move: move.base_power * bu.get_type_multiplier(move.type, battle.opponent_active_pokemon))
            move_value = best_move.base_power * bu.get_type_multiplier(best_move.type, battle.opponent_active_pokemon)
            if move_value > best_value:
                best_value = move_value
                poke_id = i
        
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power * bu.get_type_multiplier(move.type, battle.opponent_active_pokemon))
            move_value = best_move.base_power * bu.get_type_multiplier(best_move.type, battle.opponent_active_pokemon)
            if len(battle.available_switches) == 0 or move_value > best_value:
                # No switches available
                # print("MDP: Chose: " + self.create_order(best_move))
                return self.create_order(best_move, mega=battle.can_mega_evolve)
        
        # print("MDP: Chose: " + self.create_order(battle.available_switches[poke_id]))
        return self.create_order(battle.available_switches[poke_id])

    def get_switch_moves(self, battle, moves):
        switch_moves = []
        for move in moves:
            if move.self_switch:
                switch_moves.append(move)
        
        if len(switch_moves) > 0:
            switch_moves.sort(reverse=True, key=lambda mov: mov.base_power * bu.get_type_multiplier(mov.type, battle.opponent_active_pokemon))
            if bu.get_type_multiplier(switch_moves[0].type, battle.opponent_active_pokemon) == 0:
                return []

        return switch_moves

    def dict_moves_to_list(self, dictionary):
        arr = []
        if dictionary == None:
            return arr
        for key in dictionary:
            arr.append(dictionary[key])
        return arr

    def use_setup_move(self, battle, default_move, default_damage):
        moves = battle.available_moves
        best_boost = 0
        move_id = 0
        for i in range(len(moves)):
            boost = sum(self.dict_moves_to_list(moves[i].self_boost))
            if boost > best_boost:
                best_boost = boost
                move_id = i
        # See if it is able to setup
        if best_boost > 0:
            # print("MDP: Chose: " + self.create_order(moves[move_id]))
            return self.create_order(moves[move_id], mega=battle.can_mega_evolve)
        # See if it has a good enough move
        if default_damage > 90:
            # print("MDP: Chose: " + self.create_order(default_move))
            return self.create_order(default_move, mega=battle.can_mega_evolve)
        # See if it has a switch move
        switch_moves = self.get_switch_moves(battle, moves)
        if len(switch_moves) > 0:
            # print("MDP: Chose: "+ self.create_order(switch_moves[0]))
            return self.create_order(switch_moves[0], mega=battle.can_mega_evolve)
        # Switch
        return self.choose_best_safe_switch(battle)


    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power * bu.get_type_multiplier(move.type, battle.opponent_active_pokemon))
            move_damage = best_move.base_power * bu.get_type_multiplier(best_move.type, battle.opponent_active_pokemon)

            # See if it is able to setup
            if battle.active_pokemon.current_hp_fraction >= 0.74:
                return self.use_setup_move(battle, best_move, move_damage)

            # See if it has a good enough move
            if move_damage > 90:
                # print("MDP: Chose: " + self.create_order(best_move))
                return self.create_order(best_move, mega=battle.can_mega_evolve)

            # See if it has a switch move
            switch_moves = self.get_switch_moves(battle, battle.available_moves)
            if len(switch_moves) > 0:
                # print("MDP: Chose: "+ self.create_order(switch_moves[0]))
                return self.create_order(switch_moves[0], mega=battle.can_mega_evolve)

            # Switch
            return self.choose_best_safe_switch(battle)


        # If no attack is available, the "best safe" switch will be made
        else:
            return self.choose_best_safe_switch(battle)

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
        conn=None,
        train=True
    ) -> None:
        self.train = train
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.001
        self.conn = conn
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
        pass

    def action_to_move(self, action, battle: Battle):
        orders = []
        for move in battle.available_moves:
            if battle.can_mega_evolve:
                orders.append(self.create_order(move, mega=True))
            else:
                orders.append(self.create_order(move))

        for key in battle.team:
            if battle.team[key].active:
                continue
            orders.append(self.create_order(battle.team[key]))
        if not battle.available_moves:
            action -= 4
        if len(battle.available_moves) == 1 and not battle.available_switches:
            action = 0
        if action >= len(orders):
            action = np.random.randint(0, len(orders))
        # print("Chose: " + str(orders[action]))
        return orders[action]


    def battle_to_state(self, battle: Battle):
        state = np.array([])
        # ----- Add own team info ----- #
        state = np.append(state, bu.getTeamFeatures(battle.team, battle.active_pokemon))
        # ----- Add opponent team info ----- #
        state = np.append(state, bu.getTeamFeatures(battle.opponent_team, battle.opponent_active_pokemon))
        # ----- Add own moves info ----- #
        state = np.append(state, bu.getMovesInfo(battle.active_pokemon, battle.opponent_active_pokemon, battle))
        # ----- Add opponent moves info ----- #
        state = np.append(state, bu.getMovesInfo(battle.opponent_active_pokemon, battle.active_pokemon, battle))
        return state
    
    def state_to_action(self, state: np.array, battle: Battle):
        self.conn.send(state)
        actions = self.conn.recv()[0]
        action = 0
        if self.train:
            if self.epsilon < np.random.random():
                rn = np.random.random()
                act_sum = 0
                for act in actions:
                    if rn <= act + act_sum:
                        break
                    act_sum += act
                    action += 1
            else:
                action = np.random.randint(0, 9)
        else:
            if np.random.random() > 0.05:
                rn = np.random.random()
                act_sum = 0
                for act in actions:
                    if rn <= act + act_sum:
                        break
                    act_sum += act
                    action += 1
            else:
                action = np.random.randint(0, 9)

        return action

    def replay(self, battle_history: Dict):
        message = [-2]
        for key in battle_history:
            for turn in range(len(battle_history[key])):
                if (turn + 1 == len(battle_history[key])):
                    break
                state = battle_history[key][turn][0]
                action = battle_history[key][turn][1]
                new_state = battle_history[key][turn + 1][0]
                message.append([state, action, new_state])  
        self.conn.send(message)
        # ack = self.conn.recv()

##########################################################################

async def evaluating(future, child, opponent):
    # We define two player configurations.
    player_1_configuration = PlayerConfiguration("Agent player", None)
    player_2_configuration = PlayerConfiguration("Enemy player", None)
    
    # We create the corresponding players.
    agent_player = PokeAgent(
        player_configuration=player_1_configuration,
        battle_format="gen7letsgorandombattle",
        server_configuration=LocalhostServerConfiguration,
        conn=child,
        train=False
    )
    if opponent == "random":
        enemy_player = RandomPlayer(
            player_configuration=player_2_configuration,
            battle_format="gen7letsgorandombattle",
            server_configuration=LocalhostServerConfiguration,
        )
    else:
        enemy_player = AggressivePlayer(
            player_configuration=player_2_configuration,
            battle_format="gen7letsgorandombattle",
            server_configuration=LocalhostServerConfiguration,
        )

    n_battles = 100
    won_battles = 0
    while n_battles > 0:
        cross_evaluation = await cross_evaluate(
            [agent_player, enemy_player], n_challenges=1
        )
        n_battles -= 1
        won_battles += cross_evaluation[agent_player.username][enemy_player.username]


    print("Agent won {} / 100 battles".format(won_battles))
    future.set_result("I'm done!")
    agent_player.conn.send([-1])
    
async def training(future, child, opponent):

    # We define two player configurations.
    player_1_configuration = PlayerConfiguration("Agent player", None)
    player_2_configuration = PlayerConfiguration("Enemy player", None)
    
    # We create the corresponding players.
    agent_player = PokeAgent(
        player_configuration=player_1_configuration,
        battle_format="gen7letsgorandombattle",
        server_configuration=LocalhostServerConfiguration,
        conn=child
    )
    if opponent == "random":
        enemy_player = RandomPlayer(
            player_configuration=player_2_configuration,
            battle_format="gen7letsgorandombattle",
            server_configuration=LocalhostServerConfiguration,
        )
    else:
        enemy_player = AggressivePlayer(
            player_configuration=player_2_configuration,
            battle_format="gen7letsgorandombattle",
            server_configuration=LocalhostServerConfiguration,
        )

    episodes = 800
    while episodes > 0:
        await agent_player.train_against(enemy_player, 1)
        episodes -=1
        if agent_player.epsilon > agent_player.min_epsilon:
            agent_player.epsilon = max(agent_player.epsilon * agent_player.epsilon_decay, agent_player.min_epsilon)
        if (800 - episodes == 1 or 800 - episodes == 100 or 800 - episodes == 200 
                or 800 - episodes == 300 or 800 - episodes == 400 or 800 - episodes == 500
                or 800 - episodes == 600 or 800 - episodes == 700 or 800 - episodes == 800):
            print("Fiz " + str(800 - episodes) + " batalhas - SAVING MODEL")
            agent_player.conn.send([-3, 800 - episodes])
    print("Terminei")
    future.set_result("I'm done!")
    agent_player.conn.send([-1])

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

def get_reward(state, new_state):
    if np.array_equal(state, new_state):
        print("SAME STATE")
        return -1
    own_lost_poke = get_alive_own_pokemon(state) - get_alive_own_pokemon(new_state)
    opp_lost_poke = (get_alive_opp_pokemon(state) - get_alive_opp_pokemon(new_state)) * 1.5
    return (opp_lost_poke - own_lost_poke + damage_dealt(state, new_state) * 1.5 - damage_taken(state, new_state)) # * 10



def startPSthread(child, mode, opponent):
    loop = asyncio.get_event_loop()
    future = asyncio.Future()
    if mode == "train":
        asyncio.ensure_future(training(future, child, opponent))
    else:
        asyncio.ensure_future(evaluating(future, child, opponent))
    loop.run_until_complete(future)
    loop.close()

def arguments_error(reason):
    print(reason + ': Usage is python <script path> <mode> <opponent> [model path]')
    print('Mode is either \"train\" or \"evaluate\"')
    print('Opponent is either \"random\" or \"aggressive\"')
    print('If mode is \"evaluate\": \"model path\" is required')
    sys.exit()



if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        arguments_error("Wrong arguments")

    if sys.argv[1] != "train" and sys.argv[1] != "evaluate":
        arguments_error("Wrong mode")

    if sys.argv[1] == "evaluate" and len(sys.argv) != 4:
        arguments_error("No model specified")

    if sys.argv[2] != "random" and sys.argv[2] != "aggressive":
        arguments_error("Wrong opponent")

    gamma = 0.95
    
    parent, child = Pipe()

    pool = Pool(processes=1)
    result = pool.apply_async(startPSthread, (child, sys.argv[1], sys.argv[2],))

    if sys.argv[1] == "train" and len(sys.argv) < 4:
        model = Sequential()
        model.add(Dense(50, activation="elu", input_shape=(110,)))
        model.add(Dense(50, activation="elu"))
        model.add(Dense(25, activation="elu"))
        model.add(Dense(25, activation="elu"))
        model.add(Dense(9, activation="softmax"))
        model._make_predict_function()
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    else:
        if os.path.isfile(sys.argv[3]):
            model = load_model(sys.argv[3])
        else:
            arguments_error("Wrong model path")

    while True:
        state = parent.recv()
        
        # All battles ended
        if state[0] == -1:
            print("BATTLES ARE OVER")
            break

        if state[0] == -3:
            if sys.argv[2] == "random":
                model.save("models\\model{:03d}.h5".format(state[1]))
            else:
                model.save("models\\model{:03d}_agg.h5".format(state[1]))
            continue
        
        # Onde battle ended, training the network
        if state[0] == -2:
            print("TRAINING TIME")
            state.pop(0)

            current_states = np.array([transition[0] for transition in state])
            current_qs_list = model.predict(current_states)

            new_current_states = np.array([transition[2] for transition in state])
            future_qs_list = model.predict(new_current_states)
            
            X = []
            y = []
            
            for turn in range(len(state)):
                # print(state[turn])
                curr_state = state[turn][0]
                action = state[turn][1]
                new_state = state[turn][2]
                reward = get_reward(curr_state, new_state)
                
                # TOTO: Fit network here
                if turn == len(state) - 1:
                    new_q = reward
                else:
                    max_future_q = np.max(future_qs_list[turn])
                    new_q = reward + gamma * max_future_q

                current_qs = current_qs_list[turn]
                current_qs[action] = new_q

                X.append(curr_state)
                y.append(current_qs)

            model.fit(np.array(X), np.array(y), verbose=1)
            continue
        
        actions = model.predict(np.array([state]))
        parent.send(actions)
