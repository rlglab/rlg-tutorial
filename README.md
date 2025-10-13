# Tutorials for Reinforcement Learning in Games

This repository provides starter code and instructions for two tutorials on learning reinforcement learning (RL) in games.
The tutorials break down the essential parts of the RL algorithms into key TODO sections, allowing you to complete them and run the AI program without needing to handle other routine elements like game tasks.

### TD Learning for 2048 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlglab/rlg-tutorial/blob/main/tdl2048_tutorial.ipynb?hl=en)
You will implement the **temporal difference (TD) learning** algorithm to train a value function for the game [**2048**](https://en.wikipedia.org/wiki/2048_(video_game)).  
The goal is to train an agent that can successfully merge a 2048-tile in the game.  
<!-- The task involves completing key TODO sections and experimenting with value function approximation using **N-tuple networks**.   -->

### AlphaZero for TicTacToe / Connect4 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlglab/rlg-tutorial/blob/main/alphazero_tutorial.ipynb?hl=en)
You will implement **Monte Carlo Tree Search** with **PUCT** selection in an **AlphaZero** framework.  
The goal is to train the agents to play [**Connect4**](https://en.wikipedia.org/wiki/Connect_Four) and [**TicTacToe**](https://en.wikipedia.org/wiki/Tic-tac-toe) against a random agent baseline.  
<!-- The task involves completing key TODO sections and experimenting with different network architectures and feature designs to improve the agent's performance.   -->


## 🧠 What You Will Learn

- **Temporal Difference Learning** and afterstate updates  
- **Value function approximation** using tuple networks  
- **Monte Carlo Tree Search** for policy improvement  
- **AlphaZero training pipeline** (self-play, optimization)


## 🧩 Assignment 1: TD Learning for 2048

### 1. Objectives
Train a strong 2048 player using the TD(0) learning algorithm.

### 2. Required TODOs
#### Implement the best-move selection:
```python
def select_best_move(self, b : board) -> move:
    # ============== TODO ==============
    # hint: use self.estimate(b) to retrieve V(b)
    moves = [ move(b, opcode) for opcode in range(4) ]
    random.shuffle(moves)
    for mv in moves:
        if mv.is_valid():
            return mv # select a legal move randomly
    return move() # no legal move
```
- Iterate over four possible move directions.
- Exclude illegal moves.
- Return the move with the highest $r + V(s')$.
- Expected result:
    - Average score > 1100.
    - Max tiles should be 512-tile or 1024-tile.

#### Implement TD(0) updates:
```python
def learn_from_episode(self, path : list[move], alpha : float = 0.1) -> None:
    # ============== TODO ==============
    # hint: use self.estimate(b) to retrieve V(b);
    # use self.update(b, u) to update V(b) with an error u
```
- For each afterstate $s'$, update:
    - $V(s'_t) ← V(s'_t) + α (r_{t+1} + V(s'_{t+1}) − V(s'_t))$
- For last afterstate:
    - $V(s'_{T−1}) ← V(s'_{T−1}) + α (0 − V(s'_{T−1}))$
- Expected result:
    - Average score > 3000 after 100 trained games.
    - 2048-tile should appear within 2000 trained games.

### 3. Advanced Topics
Beyond the tutorials, you might want to dive deeper and explore more:  
- **Features of N-Tuple Network**: experiment with different tuple architectures to improve performance.
- **Expectimax Search**: implement a lookahead search procedure for better action decision.

## 🧩 Assignment 2: AlphaZero for TicTacToe / Connect4

### 1. Objectives
Train an strong AlphaZero-based agent for Connect4 and TicTacToe using MCTS with PUCT.

### 2. Required TODOs
#### Implement child node selection by PUCT:
```python
def select_child(self, parent: Node) -> Node:
    # ============== TODO ==============
    # hint: select the child with the highest PUCT score
    # hint: self.PUCT_C1 and self.PUCT_C2 are PUCT constants
    best_child = np.random.choice(parent.children)
    return best_child
```
- Select the best child node by $\arg\max_{a}(Q(s,a) + U(s,a))$.
    - $Q(s,a)=
    \begin{cases}
    -1& \text{if } N(s,a)=0\\
    Q(s,a)& \text{if } N(s,a)\neq0
    \end{cases}$
    - $U(s,a)=P(s,a) \frac{\sqrt{\sum{_b}N(s,b)}}{1+N(s,a)}[c_1+\log(\frac{\sum{_b}N(s,b)+c_2+1}{c_2})]$
- $N(s,a)$ represents the visit count of node $s$ when taking action $a$.
- $\sum{_b}N(s,b)$ represents the total visit count of the child nodes $b$ of node $s$, which typically refers to the visit count of the parent node.
- If multiple child nodes have the same $Q(s,a) + U(s,a)$ score, select the one
with the highest policy $P(s,a)$.
- Expected result:
    - For the TicTacToe, after about 50 iterations, draw should be the most frequent outcome.

### 3. Advanced Topics
Beyond the tutorials, you might want to dive deeper and explore more:  
- **Network architecture**: add convolutional or residual layers.
- **Feature design**: add history channels or new board encodings.

## 🔗 References
- [M. Szubert and W. Jaśkowski, "Temporal difference learning of N-tuple networks for the game 2048," CIG 2014.](https://doi.org/10.1109/CIG.2014.6932907)  
- [I-C. Wu, K.-H. Yeh, C.-C. Liang, C.-C. Chang, and H. Chiang, "Multi-stage temporal difference learning for 2048," TAAI 2014.](https://doi.org/10.1007/978-3-319-13987-6_34)  
- [K. Matsuzaki, "Systematic selection of N-tuple networks with consideration of interinfluence for game 2048," TAAI 2016](https://doi.org/10.1109/TAAI.2016.7880154).  
- [D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," Science 362, 2018.](https://doi.org/10.1126/science.aar6404)  
