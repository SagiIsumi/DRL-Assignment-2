import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import time
from evaluate import eval


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
def rot90(pattern, board_size):
    """
    90 degree counterclockwise rotation
    """
    return [(board_size - 1 - y, x) for x, y in pattern]

def rot180(pattern, board_size):
    """
    180 degree counterclockwise rotation
    """
    return [(board_size - 1 - x, board_size - 1 - y) for x, y in pattern]

def rot270(pattern, board_size):
    """
    270 degree counterclockwise rotation
    """
    return [(y, board_size - 1 - x) for x, y in pattern]
def reflect(pattern, board_size):
    """
    horizontal reflection
    """
    return [(board_size - 1 - x, y) for x, y in pattern]
def sort_parts(pattern):
  pattern=sorted(pattern, key=lambda x: (x[0], x[1]))
  return pattern
# -------------------------------

def plot_mean_scores(final_scores,stage):
    final_average_scores=[np.mean(final_scores[i-100:i]) for i in range(100,len(final_scores))]
    plt.figure(figsize=(10, 6))
    plt.plot(final_average_scores)
    plt.xlabel("Episodes")
    plt.ylabel("final_average_scores")
    plt.title("Scores of TA-learning methods of 2048")
    plt.legend()
    plt.savefig(f"final_average_scores{stage}.jpg")



class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        
        def default_value():
            return 500
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(sort_parts(syms_))
        print(f"symmetry_patterns:{self.symmetry_patterns}")

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        symmetries = []
        symmetries.append(pattern)  # original
        symmetries.append(rot90(pattern,self.board_size))  # 90 degree rotation
        symmetries.append(rot180(pattern,self.board_size))  # 180 degree rotation
        symmetries.append(rot270(pattern,self.board_size))  # 270 degree rotation
        symmetries.append(reflect(pattern,self.board_size))  # horizontal reflection
        symmetries.append(rot90(reflect(pattern,self.board_size),self.board_size))  # 90 degree rotation
        symmetries.append(rot180(reflect(pattern,self.board_size),self.board_size))  # 180 degree rotation
        symmetries.append(rot270(reflect(pattern,self.board_size),self.board_size))  # 270 degree rotation

        return symmetries

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, pattern):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        feature=[self.tile_to_index(board[x,y]) for x,y in pattern]
        return tuple(feature)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        value = 0
        for i , pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            index=i//8
            value += self.weights[index][feature]
        value=value
        return value
    def update(self, board, td_error, alpha):
        
        # print(self.value(board))
        # TODO: Update weights based on the TD error.
        for i , pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            index=i//8
            self.weights[index][feature] += alpha * (td_error)/len(self.symmetry_patterns)
        
        # print(self.value(board))
        # time.sleep(2)


def create_env_from_state(env):
    """
    Creates a deep copy of the environment with a given board state and score.
    """
    new_env = copy.deepcopy(env)
    return new_env
def td_learning(env, approximator, num_episodes=50000, alpha=0.1, gamma=0.99,stage:str="",stage_record=[]):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    stage_next_board=[]
    final_scores = []
    success_flags = []
    epsilon=0.0
    if stage_record:
        num_episodes=len(stage_record)
    for episode in range(num_episodes):
        state = env.reset().copy()
        if stage_record:
            env.board=stage_record[episode][0]
            state=stage_record[episode][0]
            env.score=stage_record[episode][1]
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)
        score_threshold=4000
        stage_next_board_trigger=True
        stopper=0
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            rand_num=np.random.rand()
            if rand_num < epsilon:
                action = random.choice(legal_moves)
            else:
                expected_reward=-100000
                for action in range(4):
                    if action in legal_moves:
                        _,sim_state,sim_score=eval(env.board,env.score,action)
                        total_reward=sim_score-previous_score+gamma*approximator.value(sim_state)
                        if total_reward>expected_reward:                          
                            expected_reward=total_reward
                            best_action=action
                action=best_action
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            _,next_state,new_score=eval(env.board,env.score,action)
            incremental_reward = new_score - previous_score
            trajectory.append((state, action, incremental_reward, next_state))
            _, _, done, _ = env.step(action)
            if env.score>score_threshold:
                print(f"episode:{episode},score:{env.score}")
                score_threshold+=4000
          
            # TODO: Store trajectory or just update depending on the implementation
            previous_score = new_score
            state = next_state
            max_tile = max(max_tile, np.max(next_state))

            if stage=="stage1":
                if max_tile>=8192 and stage_next_board_trigger:
                    stage_next_board.append((state,new_score))
                    stage_next_board_trigger=False
            if stage=="stage2":
                if max_tile>=16384 and stage_next_board_trigger:
                    stage_next_board.append((state,new_score))
                    stage_next_board_trigger=False
        #epsilon=epsilon*0.995
        # if episode>8000:
        #     alpha=max(alpha*0.999995,0.15)
        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        for state, action, incremental_reward, next_state in reversed(trajectory):
            #print(f"state:{state}, action:{action}, incremental_reward:{incremental_reward}, next_state:{next_state}")
            value = approximator.value(state)
            next_value = approximator.value(next_state)
            td_error = incremental_reward + gamma * next_value - value
            # if stopper<2:
            #     print(f"state:{state}, action:{action}, \n next_state:{next_state}")
            #     print(f"value:{value}, next_value:{next_value},incremental_reward:{incremental_reward},td_error:{td_error}")
            #     print(f"td_error:{td_error}")
            #     stopper+=1
            # time.sleep(0.1)
            approximator.update(state, td_error, alpha)
        # print(f"weights1:{approximator.weights[0][(0,0,0,0,0,0)]}\nweights2:{approximator.weights[1][(0,0,0,0,0,0)]}")
        # print(f"weights1:{approximator.weights[0][(1,1,0,0,0,0)]}\nweights2:{approximator.weights[1][(1,1,0,0,0,0)]}")
        # time.sleep(2)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}, epsilon:{epsilon}")
            print(f"value:{value}, next_value:{next_value},incremental_reward:{incremental_reward},td_error:{td_error}")
            # print(f"dictionary: {approximator.weights[0]}")
            # sorted_weights=sorted(approximator.weights[0].items(), key=lambda x: x[1], reverse=True)
            # print(f"sorted_weights:{sorted_weights}")
        if len(stage_next_board)>=20000:
            return final_scores,stage_next_board
    return final_scores,stage_next_board

if __name__=="__main__":
    from student_agent import Game2048Env
    # TODO: Define your own n-tuple patterns
    #patterns = [[(0,0),(1,0),(2,0),(3,0),(2,1),(3,1)],[(0,1),(1,1),(2,1),(3,1),(2,2),(3,2)],[(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)],[(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]]
    patterns = [[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)],[(0,0),(1,0),(2,0),(3,0),(0,1),(1,1)],[(0,0),(1,0),(2,0),(3,0),(2,1),(3,1)]]
    # patterns = [[(0,0),(0,1),(1,0),(1,1)],[(0,0),(0,1),(0,2),(1,0)]]
    approximator_stage1 = NTupleApproximator(board_size=4, patterns=patterns)
    approximator_stage2 =  NTupleApproximator(board_size=4, patterns=patterns)
    env = Game2048Env()

    # Run TD-Learning training
    # Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
    # However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
    final_scores,stage_next_board = td_learning(env, approximator_stage1, num_episodes=12000, 
                                                alpha=0.16, gamma=0.99,stage="stage1")
    plot_mean_scores(final_scores=final_scores,stage=1)
    with open('stage_1.pkl', 'wb') as f:
        pickle.dump(approximator_stage1.weights, f)
    print(stage_next_board)

    final_scores ,stage_next_board = td_learning(env, approximator_stage2, num_episodes=20000, 
                            alpha=0.2, gamma=0.99,stage="stage2",stage_record=stage_next_board)
    plot_mean_scores(final_scores=final_scores,stage=2)
    with open('stage_2.pkl', 'wb') as f:
        pickle.dump(approximator_stage2.weights, f)
    print(stage_next_board)