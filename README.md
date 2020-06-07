# Deep Pokemon

Deep Pokémon is a Pokémon Battler Artificial Intelligence based on Deep-Q Learning, and is run on the Pokémon Battle Simulator - Pokémon Showdown!

## Setup

### Install poke-env
https://github.com/hsahovic/poke-env
```
pip install poke-env
```
This requires python >= 3.6

### Clone this Pokemon Showdown fork and install it
```
git clone https://github.com/hsahovic/Pokemon-Showdown.git
cd Pokemon-Showdown
git checkout 87d8912dc63ac410797944087d5391b206b00b83
./pokemon-showdown
```

## Running

### Start a Pokemon Showdown local server
Run the following command on the Pokemon Showdown fork:
```
node pokemon-showdown
```
This requires Node.js v10+

### Run Deep Pokemon
For training, run:
```
python deep_pokemon.py train <opponent> [model_path]
```

For evaluating, run:
```
python deep_pokemon.py evaluate <opponent> <model_path>
```

Opponent should be either 'random' or 'aggressive'

To battle against the agent, run:
```
python deep_pokemon.py human <model_path>
```
