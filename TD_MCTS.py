import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, env, parent=None, action=None):
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
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.max=float('-inf')
        self.min=float('inf')

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env
    def normalize(self,values):
        min_val = self.min
        max_val = self.max
        if max_val == min_val:
            return [0] * len(values)
        return [(2 * (x - min_val) / (max_val - min_val)) - 1 for x in values]
    def select_child(self, node,sim_env):
        if sim_env.is_game_over():
          return node
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        if node.untried_actions:
          return self.expansion(sim_env,node)
        else:
          child_reward = [sum(child.total_reward for child in child_list) / len(child_list)
                for child_list in node.children.values()]
          child_reward=self.normalize(child_reward)

          child_visits=[sum(child.visits for child in child_list)
                for child_list in node.children.values()]
          
          complmentary=[self.c * np.sqrt(np.log(node.visits) / child_visits[i] ) for i in range(len(child_reward))]

          ucb1_scores =[child_reward[i]  + self.c * np.sqrt(np.log(node.visits) / child_visits[i] ) for i in range(len(child_reward))]
          for act in range(4):
             if act not in node.children.keys():
                ucb1_scores.insert(act,-1.01)
        #   print(f"ucb1:{ucb1_scores},complmentary:{complmentary}")
        #   print(f"child_reward:{child_reward}, child_visits:{child_visits}")
          state,reward,_,_=sim_env.step(np.argmax(ucb1_scores))
          try:
            for index,child in enumerate(node.children[np.argmax(ucb1_scores)]):
              if np.array_equal(child.state, state):
                  #print(f"state:{state},child.state:{child.state}")
                  return self.select_child(node.children[np.argmax(ucb1_scores)][index],sim_env)
              
            node.children[np.argmax(ucb1_scores)].append(TD_MCTS_Node(state.copy(), reward, sim_env, node, np.argmax(ucb1_scores)))
            expanded_node = node.children[np.argmax(ucb1_scores)][-1]
            return expanded_node
          except:
            print(ucb1_scores)
    def expansion(self,sim_env,node):
        if sim_env.is_game_over():
          return node
        item =node.untried_actions.pop(0)
        state, reward, done, _ =sim_env.step(item)
        if item not in node.children.keys():
          node.children[item] = []
        node.children[item].append(TD_MCTS_Node(state.copy(), reward, sim_env, node, item))
        expanded_node = node.children[item][-1]
        return expanded_node
    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        rand_num=np.random.rand()
        # if sim_env.is_game_over():
        #   return 0
        if rand_num<1.0 or depth==0 or sim_env.is_game_over():
          #print(self.approximator.value(sim_env.board))
          return self.approximator.value(sim_env.afterstate_board)
        
        action = random.choice([a for a in range(4) if sim_env.is_move_legal(a)])
        state, reward, done, _ =sim_env.step(action)
        
        return self.rollout(sim_env, depth - 1)

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        node.visits += 1
        node.total_reward += (reward-node.total_reward)/node.visits
        if node.total_reward>self.max:
            self.max=node.total_reward
        if node.total_reward<self.min:  
            self.min=node.total_reward
        if node.parent:
          self.backpropagate(node.parent,node.score-node.parent.score+self.gamma*reward)
    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        expanded_node=self.select_child(node,sim_env)

        # TODO: Expansion: if the node has untried actions, expand one.
        

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
        child_reward = [sum(child.total_reward for child in child_list) / len(child_list)
                for child_list in root.children.values()]
        for act in range(4):
             if act not in root.children.keys():
               child_reward.insert(act,-1.01)
        child_visits=[sum(child.visits for child in child_list)
                for child_list in root.children.values()]
        for act in range(4):
             if act not in root.children.keys():
               child_visits.insert(act,0)
        total_visits = sum(child_visits[v] for v in legal_moves)
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action in root.children.keys():
            if action not in legal_moves:
              distribution[action] = 0
            else:
              distribution[action] = child_visits[action] / total_visits if total_visits > 0 else 1/len(legal_moves)
              if child_visits[action] > best_visits:
                  best_visits = child_visits[action]
                  best_action = action
        return best_action, distribution, child_reward, child_visits
