import copy
import random
import math
import numpy as np
from evaluate import eval

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

def normalize(values):
  min_val = min(values)
  max_val = max(values)

  if max_val == min_val:
      return [0] * len(values)

  return [(2 * (x - min_val) / (max_val - min_val)) - 1 for x in values]
# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node,sim_env):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        if node.untried_actions:
          return node
        else:
          child_reward=[child.total_reward for child in node.children.values()]
          child_reward=normalize(child_reward)
          child_visits=[child.visits for child in node.children.values()]
          # print(f"child_reward:{child_reward}, child_visits:{child_visits}")
          complmentary=[self.c * np.sqrt(np.log(node.visits) / child_visits[i] ) for i in range(len(child_reward))]

          ucb1_scores =[child_reward[i]  + self.c * np.sqrt(np.log(node.visits) / child_visits[i] ) for i in range(len(child_reward))]
          # print(f"ucb1:{ucb1_scores},complmentary:{complmentary}")
          # print(np.argmax(ucb1_scores))
          _,_,_,_=sim_env.step(np.argmax(ucb1_scores))
          return self.select_child(node.children[np.argmax(ucb1_scores)],sim_env)
    def expansion(self,sim_env,node):
        if not sim_env.is_game_over:
          return node
        item =node.untried_actions.pop(0)
        state, reward, done, _ =sim_env.step(item)
        node.children[item] = TD_MCTS_Node(state.copy(), reward, node, item)
        expanded_node = node.children[item]
        return expanded_node
    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        if depth == 0 or sim_env.is_game_over():
          return self.approximator.value(sim_env.board)
        rand_num=np.random.rand()
        best_reward=-10000
        legal_moves=[a for a in range(4) if sim_env.is_move_legal(a)]
        if rand_num < 0.1:
          action = random.choice([a for a in range(4) if sim_env.is_move_legal(a)])
        else:
           for i in range(4):
              moved,sim_state,sim_score=eval(sim_env.board,sim_env.score,i)
              if not moved:
                 continue
              total_reward=sim_score+self.gamma*self.approximator.value(sim_state)
              if total_reward>best_reward:
                  if i in legal_moves:
                    best_reward=total_reward
                    action=i
        score=self.approximator.value(sim_env.board)
        state, reward, done, _ =sim_env.step(action)
        return score+self.gamma*self.rollout(sim_env, depth - 1)

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        node.visits += 1
        node.total_reward += (reward-node.total_reward)/node.visits
        if node.parent:
          self.backpropagate(node.parent, reward)
    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        selected_node=self.select_child(node,sim_env)

        # TODO: Expansion: if the node has untried actions, expand one.
        expanded_node=self.expansion(sim_env,selected_node)

        # Rollout: Simulate a random game from the expanded node.
        depth=self.rollout_depth
        rollout_reward = self.rollout(sim_env, depth)

        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(expanded_node, rollout_reward)

    def best_action_distribution(self, env,root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        legal_moves=[action for action in range(4) if env.is_move_legal(action)]
        child_visits= [child.visits for child in root.children.values()]
        total_visits = sum(child_visits[v] for v in legal_moves)
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if action not in legal_moves:
              distribution[action] = 0
            else:
              distribution[action] = child.visits / total_visits if total_visits > 0 else 1/len(legal_moves)
              if child.visits > best_visits:
                  best_visits = child.visits
                  best_action = action
        return best_action, distribution
