import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import time
import re
import sys
import itertools

class TD_learn():
    def __init__(self):
        self.size=19
        feature_size=16
        self.turn=1
        self.game_over=False
        self.board = np.zeros((self.size, self.size), dtype=int)
        alpha=0.0001
        self.model=nn.Sequential(
            nn.Linear(feature_size,28),
            nn.ReLU(),
            nn.Linear(28,56),
            nn.ReLU(),
            nn.Linear(56,28),
            nn.ReLU(),
            nn.Linear(28,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,1),
        )
        self.loss_func=nn.MSELoss()
        self.optimizer=optim.Adam(self.model.parameters(),lr=alpha)
        self.trajectory=[]
        #self.weights_func=torch.ones([feature_size])/100
    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    
    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            self.game_over=True
                            return current_color
                
        return 0
    def get_act_zone(self,r=2):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        act_zone=set()
        radius=r
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    for dr, dc in directions:
                        for rad in range(-radius,radius+1):
                            nr=r+rad*dr
                            nc=c+rad*dc
                            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == 0:
                                    act_zone.add((nr, nc))
        if act_zone ==set():
            act_zone.add((self.size//2,self.size//2))
        return act_zone
    
    def check_line(self,state, x, y,color,debug=False):
        features=[0,0,0,0,0,0,0,0]
        if color =='B':
            attacker=1
            defender=2
        else:
            attacker=2
            defender=1
        directions = [(0, 1), (1, 0), (1, 1),(1,-1)]
        for dx, dy in directions:
            end=[-4,7]
            count=0
            connect_count=0
            connect_counting=0
            zero_count=0
            accumulated_zero=0
            added=False
            for index in range(-3, 7):  # 向前2和向後6個位置
                nx, ny = x + index * dx, y + index * dy
                if index<0:
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if state[nx,ny]==attacker:
                            break
                        if state[nx,ny]==defender:
                            end[0]=index
                            continue
                        if not (0 <= nx < self.size and 0 <= ny < self.size):
                            end[0]=index
                            continue
                    else:
                        end[0]=index
                        continue# 超出邊界
                elif 0 <= nx < self.size and 0 <= ny < self.size:
                    if state[nx,ny]==attacker: 
                        count+=1
                        connect_counting+=1
                        zero_count=0
                        if connect_counting>connect_count and index==6:
                            connect_count=connect_counting                     
                    elif state[nx,ny]==defender:
                        end[1]=index
                        # print(nx,ny,state[nx,ny])
                        break
                    else:
                        if connect_counting>connect_count:
                            connect_count=connect_counting
                        connect_counting=0
                        zero_count+=1
                        accumulated_zero+=1
                        if zero_count==3:
                            end[1]=index+1
                            break
                else:
                    end[1]=index
                    break# 超出邊界
            interval=end[1]-end[0]-1
            # print(state[x,y],x,y,interval,count,connect_count,zero_count,end)
            # print(features)
            # time.sleep(3)
            if connect_count>=6:
                features[7]+=1
            if interval==5:
                if 0 <= x+5*dx < self.size and 0 <= y+5*dy <self.size:
                    if count==2 and end[0]==-1 and state[x+5*dx,y+5*dy]==0:
                        features[5]+=1
                if 0 <=  x-4*dx < self.size and 0 <= y-4*dy < self.size:    
                    if count == 2 and end[1]==2 and state[x-4*dx,y-4*dy]==0:
                        features[5]+=1
            elif interval==6:
                if count==2:
                    features[5]+=1
                elif count==3:
                    features[3]+=1
                elif count>=4:
                    features[0]+=1
                    
            elif interval==7:
                if count==1 and zero_count==3:
                    if 0 <=  x-4*dx < self.size and 0 <= y-4*dy <self.size and 0 <= x+4*dx < self.size and 0<= y+4*dy <self.size:
                        if state[x-4*dx,y-4*dy]==0 and state[x+4*dx,y+4*dy]==0:
                            features[6]+=1
                elif count==2:
                    features[5]+=1
                elif count==3:
                    features[3]+=1
                elif connect_count==4 and zero_count==3:
                    features[0]+=1
                elif count==4 and (accumulated_zero-zero_count)>2:
                    features[2]+=1
                    added =True

                if connect_count==5 and end[0]==-2:
                    features[1]+=1
                    added =True
                elif 0 <= x+7*dx < self.size and 0 <= y+7*dy <self.size:
                    if connect_count==5 and state[x+1*dx,y+1*dy]==0 and state[x+7*dx,y+7*dy]==0:
                        features[1]+=1
                        added =True
                    elif connect_count==4 and end[0]==-2 and state[x+6*dx,y+6*dy]==0 and state[x+7*dx,y+7*dy]==0:
                        features[1]+=1
                        added =True
                if 0 <= x+8*dx < self.size and 0 <= y+8*dy <self.size:
                    if connect_count==4 and end[0]==-2 and state[x+7*dx,y+7*dy]==0 and state[x+8*dx,y+8*dy]==0:
                        features[1]+=1
                        added =True
                if count>=4 and not added:
                    features[0]+=1
                    
            elif interval==8:
                if count==2:
                    features[4]+=1
                elif count==3 and zero_count>=2 and end[0]<-2:
                    features[2]+=1
                elif count==3 and (accumulated_zero-zero_count)>2:
                    features[4]+=1
                elif count==3:
                    features[3]+=1
                elif connect_count==4 and end[0]<=-3 and zero_count>=2:
                    features[1]+=1 
                    added =True
                elif count==4 and (accumulated_zero-zero_count)==3:
                    features[2]+=1
                    added =True
                elif 0 <= x+7*dx < self.size and 0 <= y+7*dy <self.size:
                    if connect_count==4 and end[0]==-2 and state[x+6*dx,y+6*dy]==0 and state[x+7*dx,y+7*dy]==0 and zero_count!=3:
                        features[1]+=1
                        added =True
                    elif connect_count==5 and state[x+1*dx,y+1*dy]==0 and state[x+7*dx,y+7*dy]==0:
                        features[1]+=1
                        added =True
                if 0 <= x+8*dx < self.size and 0 <= y+8*dy <self.size:
                    if connect_count==4 and end[0]==-2 and state[x+7*dx,y+7*dy]==0 and state[x+8*dx,y+8*dy]==0 and zero_count!=3:
                        features[1]+=1
                        added =True
                if 0 <=  x-1*dx < self.size and 0 <= y-1*dy < self.size and 0 <= x+5*dx < self.size and 0 <= y+5*dy <self.size:
                    if connect_count==5 and state[x-1*dx,y-1*dy]==0 and state[x+5*dx,y+5*dy]==0:
                        features[1]+=1
                        added =True
                
                if count>=4 and not added:
                    features[0]+=1
                    
                
            elif interval>8:
                if debug:
                    print(f"debug: {x},{y},dx:{dx},dy:{dy}")
                    print(f'interval:{interval},count:{count}, connect:{connect_count},zero_count:{zero_count},accumulated_zero:{accumulated_zero},end:{end}')
                if count==2:
                    features[4]+=1
                elif count==3 and (accumulated_zero-zero_count)>2:
                    features[5]+=1
                elif count == 3:
                    features[2]+=1
                elif connect_count==4 and end[0]<=-3 and zero_count>=2:
                    features[1]+=1  
                    added =True
                elif count==4 and (accumulated_zero-zero_count)==3:
                    features[2]+=1
                    added =True
                elif x+7*dx < self.size and y+7*dy <self.size:
                    if connect_count==4  and state[x+6*dx,y+6*dy]==0 and state[x+7*dx,y+7*dy]==0 and zero_count!=3:
                        features[1]+=1
                        added =True
                    elif connect_count==5 and state[x+1*dx,y+1*dy]==0 and state[x+7*dx,y+7*dy]==0:
                        features[1]+=1
                        added =True
                if x+8*dx < self.size and y+8*dy <self.size:
                    if connect_count==4 and state[x+7*dx,y+7*dy]==0 and state[x+8*dx,y+8*dy]==0 and zero_count!=3:
                        features[1]+=1
                        added =True
                if x+5*dx < self.size and y+5*dy <self.size:
                    if connect_count==5 and state[x-1*dx,y-1*dy]==0 and state[x+5*dx,y+5*dy]==0:
                        features[1]+=1
                        added =True
                
                if count>=4 and not added:
                    features[0]+=1
                    

        return features
    
    def get_feature(self,state,mycolor,debug=False):
        features=[0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0]
        mycolor=mycolor.upper()
        for r in range(self.size):
            for c in range(self.size):
                if state[r, c] == 1:
                    feature=self.check_line(state,r,c,color='B',debug=debug)
                    for index,count in enumerate(feature):
                        if mycolor == 'B':
                            features[index]+=count
                        else:
                            features[index+8]+=count
                elif state[r, c] == 2:
                    feature=self.check_line(state,r,c,color='W', debug=debug)
                    for index,count in enumerate(feature):
                        if mycolor == 'W':
                            features[index]+=count
                        else:
                            features[index+8]+=count
            
        return features
    
    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')
    def greedy(self,epsilon):
        pass
    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                print(stone)
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        return positions

    def undo(self,positions):
        for row, col in positions:
            self.board[row, col] = 0

    def update(self,state_feature,next_state_feature,result,end):
        gamma=0.99
        if result >=1 and end:
            final_feature=torch.tensor(next_state_feature,dtype=torch.float32,requires_grad=True)
            final_value=self.model(final_feature)
            reward = torch.tensor([2.0], dtype=torch.float32)
            final_loss=self.loss_func(final_value,reward)
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
        elif end:
            final_feature=torch.tensor(next_state_feature,dtype=torch.float32,requires_grad=True)
            final_value=self.model(final_feature)
            reward = torch.tensor([-2.0], dtype=torch.float32)
            final_loss=self.loss_func(final_value,reward)
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
        # elif end:
        #     final_feature=torch.tensor(next_state_feature,dtype=torch.float32,requires_grad=True)
        #     final_value=self.model(final_feature)
        #     reward = torch.tensor(0.5, dtype=torch.float32)
        #     final_loss=self.loss_func(final_value,reward)
        #     self.optimizer.zero_grad()
        #     final_loss.backward()
        #     self.optimizer.step()

        feature=torch.tensor(state_feature, dtype=torch.float32, requires_grad=True)
        next_feature=torch.tensor(next_state_feature,dtype=torch.float32)
        value=self.model(feature)   
        next_value=self.model(next_feature)*gamma

        loss=self.loss_func(value,next_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self,feature):
        feature_tensor=torch.tensor(feature, dtype=torch.float32)
        # value=self.model(feature_tensor).detach().numpy()
        weights=[48.0,100.0,5.0,3.2,2.5,1.0,0.1,1500.0,-300.0,-600.0,-12.0,-5.0,-2.8,-1.2,-0.1,-900.0]
        weights=torch.tensor(weights, dtype=torch.float32)
        value=torch.dot(feature_tensor,weights)
        if feature[0]>2:
            value+=50
        if feature[0]>0 and feature[1]>0:
            value+=50
        if feature[1]>1:
            value+=50

        return value


    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)
    def onpolicy_train(self):
        print("onpolicy train",file=sys.stderr)
        for state,next_state,result,end,color in reversed(self.trajectory):
            state_feat=self.get_feature(state,color)
            next_state_feat=self.get_feature(next_state,color)
            #print(f"state_feat:{state_feat},next_state_feat:{next_state_feat},result:{result},end:{end}",file=sys.stderr)
            if color =='B':
                result = -1 if result==2 else result 
            else:
                result = -1 if result==1 else result
            self.update(state_feat,next_state_feat,result,end)
        self.trajectory=[]
        print("train done",file=sys.stderr)
        self.save_model_weights()
    def selfplay(self,dataset):
        for content in dataset:
            attacker=content["winner"]
            self.reset_board()
            b_trajectory=[]
            w_trajectory=[]
            b_state=self.board.copy()
            w_state=self.board.copy()
            self.game_over=False
            b_end=True
            w_end=True
            for move in content['moves']:
                print(move)
                positions = self.play_move('B',move[0])
                b_next_state=self.board.copy()
                _=self.check_win()
                if len(move)==2:
                    positions = self.play_move('W',move[1])
                    w_next_state=self.board.copy()
                b_trajectory.append((b_state,b_next_state,self.check_win()))
                w_trajectory.append((w_state,w_next_state,self.check_win()))
                b_state=b_next_state
                w_state=w_next_state
            
            for state,next_state,result in reversed(b_trajectory):
                state_feat=self.get_feature(state,'b')
                next_state_feat=self.get_feature(next_state,'b')
                result = -1 if result==2 else result 
                self.update(state_feat,next_state_feat,result,b_end)
                b_end=False
            for state,next_state,result in reversed(w_trajectory):
                state_feat=self.get_feature(state,'w')
                next_state_feat=self.get_feature(next_state,'w')
                result = -1 if result==1 else result 
                self.update(state_feat,next_state_feat,result,w_end)
                w_end=False

    @staticmethod
    def load_json_file():
        file='./C6_train/'
        All_file=[]
        for content in Path(file).glob('*.json'):
            with open(content,'r') as f: 
                chess_manual=json.load(f)
            All_file.extend(chess_manual)
        # print(All_file[0]['moves'])
        # print(All_file[0]['header']['players'][0]['result'])
        return All_file
    @staticmethod
    def load_sgf_file():
        file='./C6_train/'
        All_file=[]
        for content in Path(file).glob('*.txt'):
            with open(content,'r',encoding='utf-8') as f: 
                sgf_data=f.read()
                sgf_data=sgf_data.split("\n(;")
                for game in sgf_data:
                    if game=='':
                        continue
                    moves = re.findall(r';([BW])\[(.*?)\]', game)
                    formatted_moves = []
                    for i in range(0, len(moves), 2):
                        black_move = moves[i][1]
                        white_move = moves[i+1][1] if i+1 < len(moves) else None
                        black_move = re.sub(r"(\d)(\D)", r"\1,\2", black_move)
                        if white_move:
                            white_move = re.sub(r"(\d)(\D)", r"\1,\2", white_move)
                            formatted_moves.append([black_move, white_move])
                        else:
                            formatted_moves.append([black_move])
                    if white_move:
                        winner='W'
                    else:
                        winner='B'
                    game_info={"winner":winner,"moves":formatted_moves}
                    All_file.append(game_info)
        print(All_file[0])
        return All_file
    def load_model_weights(self):
        self.model.load_state_dict(torch.load('C6_weights.pth'))

    def save_model_weights(self):
        torch.save(self.model.state_dict(),'C6_weights.pth')


class alpha_beta_MCTS_Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.move_str=[]
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = self.get_children(state)  # 初始未展開的動作
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    def get_children(self,state,r=1):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        act_zone=[]
        radius=r
        for r in range(19):
            for c in range(19):
                if state[r, c] != 0:
                    for dr, dc in directions:
                        for rad in range(-radius,radius+1):
                            nr=r+rad*dr
                            nc=c+rad*dc
                            if 0 <= nr < 19 and 0 <= nc < 19 and state[nr][nc] == 0:
                                if (nr,nc) not in act_zone:
                                    act_zone.append((nr, nc))
        
        if act_zone==[]:
            act_zone.append((19//2,19//2))
        # else:
        #     act_zone = list(itertools.combinations(act_zone, 2))
        return act_zone


class alpha_beta_MCTS:
    def __init__(self, value_func, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.value_func = value_func
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.undo_positions = []
        self.attacker_color=None
    
    def normalize(self,values):
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [0] * len(values)

        return [(2 * (x - min_val) / (max_val - min_val)) - 1 for x in values]
    def select_child(self, node ):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        if node.untried_actions:
            return node
        else:
            if (self.timer%4)<2:
                color='B' if self.attacker_color=='B' else 'W'
            else:
                color='W' if self.attacker_color=='B' else 'B'
            
            child_reward=[child.total_reward for child in node.children]
            child_reward=self.normalize(child_reward)
            child_visits=[child.visits for child in node.children]
            #print(f"child_reward:{child_reward}, child_visits:{child_visits}")
            complmentary=[self.c * np.sqrt(np.log(node.visits) / child_visits[i] ) for i in range(len(child_reward))]
            ucb1_scores =[child_reward[i]  + self.c * np.sqrt(np.log(node.visits) / child_visits[i] ) for i in range(len(child_reward))]
            #print(f"ucb1:{ucb1_scores},complmentary:{complmentary}")
            # print(np.argmax(ucb1_scores))
            self.undo_positions.append(self.value_func.play_move(color,node.move_str[np.argmax(ucb1_scores)]))
            self.timer+=1
            return self.select_child(node.children[np.argmax(ucb1_scores)])
    def expansion(self,node):
        if (self.timer%4)<2:
            color='B' if self.attacker_color=='B' else 'W'
        else:
            color='W' if self.attacker_color=='B' else 'B'
        if self.value_func.check_win():
            return node
        empty_positions = [(r, c) for r in range(19) for c in range(19) if self.value_func.board[r, c] == 0]
        while True:
            action = node.untried_actions.pop()
            # if len(action) == 2:
            #     if action[0] in empty_positions and action[1] in empty_positions:
            #         break
            # else:
            if action in empty_positions:
                break        
        r, c =action
        move_str = f"{self.value_func.index_to_label(c)}{r+1}"
        #print("move_str",move_str)
        self.undo_positions.append(self.value_func.play_move(color,move_str))
        child_node = alpha_beta_MCTS_Node(self.value_func.board.copy(), parent=node)
        #print(f"move_str:{move_str},node_str:{node.move_str}",file=sys.stderr)
        node.children.append(child_node)
        node.move_str.append(move_str)
        self.timer+=1
        return child_node
    
    def alpha_beta(self, depth, alpha, beta):
        if depth == 0 or self.value_func.check_win()!=0:
            feature=self.value_func.get_feature(self.value_func.board,self.attacker_color)
            # print(self.value_func.board,file=sys.stderr)
            #print(f"feature:{feature}",file=sys.stderr)
            #time.sleep(2)
            value=self.value_func.eval(feature)
            #print(f"value:{value}",file=sys.stderr)
            self.value_func.game_over=False
            #print(self.value_func.board)
            return value

        act_zone = self.value_func.get_act_zone(r=1)
        empty_positions = [(r, c) for r in range(19) for c in range(19) if self.value_func.board[r, c] == 0]
        # if len(act_zone) >1:
        #     act_zone = list(itertools.combinations(act_zone, 2))
        if (self.timer%4)<2:
            color='B' if self.attacker_color=='B' else 'W'
        else:
            color='W' if self.attacker_color=='B' else 'B'
        
        if (self.timer%4)<2:
            value = float('-inf')
            for act in act_zone:
                #print(f"act_max:{act}",file=sys.stderr)
                # if len(act) == 2:
                #     if act[0] in empty_positions and act[1] in empty_positions:
                #         move_str = ",".join(f"{self.value_func.index_to_label(c)}{r+1}" for r, c in act)
                # else:
                if act in empty_positions:       
                    r, c =act
                    move_str = f"{self.value_func.index_to_label(c)}{r+1}"
                    self.undo_positions.append(self.value_func.play_move(color,move_str))
                    self.timer+=1
                    value = max(value, self.alpha_beta(depth - 1, alpha, beta))
                    alpha = max(alpha, value)
                    self.value_func.undo(self.undo_positions.pop())
                    self.timer-=1
                    if alpha >= beta:
                        break
            return value
        else:
            value = float('inf')
            for act in act_zone:
                #print(f"act_min:{act}",file=sys.stderr)
                # if len(act) == 2:
                #     if act[0] in empty_positions and act[1] in empty_positions:
                #         move_str = ",".join(f"{self.value_func.index_to_label(c)}{r+1}" for r, c in act)
                # else:
                if act in empty_positions:       
                    r, c =act
                    move_str = f"{self.value_func.index_to_label(c)}{r+1}"
                    #print(f"color:{color},self.timer:{self.timer}",file=sys.stderr)
                    self.undo_positions.append(self.value_func.play_move(color,move_str))
                    self.timer+=1
                    value = min(value, self.alpha_beta(depth - 1, alpha, beta))
                    beta = min(beta, value)
                    self.value_func.undo(self.undo_positions.pop())
                    self.timer-=1
                    if beta <= alpha:
                        break
            return value
    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        node.visits += 1
        node.total_reward += (reward-node.total_reward)/node.visits
        if node.parent:
          self.backpropagate(node.parent, reward)
    def stimulation(self,root, color,timer):
        self.attacker_color=color
        if len(root.untried_actions) >self.iterations:
            self.iterations=len(root.untried_actions)
        for _ in range(self.iterations):
            node = root
            self.timer=timer
            #print(self.value_func.board,file=sys.stderr)
            selected_node=self.select_child(node)

            # TODO: Expansion: if the node has untried actions, expand one.
            expanded_node=self.expansion(selected_node)
            #print(f"expanded_node:{self.value_func.board}",file=sys.stderr)
            # Rollout: Simulate a random game from the expanded node.
            depth=self.rollout_depth
            #color=='W' if color=='B' else 'B'
            value = self.alpha_beta(
                depth=depth,
                alpha=float('-inf'),
                beta=float('inf'),
            )

            #print(f"move_str:{expanded_node.parent.move_str[-1]}rollout value:{value}",file=sys.stderr)
            # Backpropagation: Update the tree with the rollout reward.
            self.backpropagate(expanded_node, value)
            while self.undo_positions:
                pos=self.undo_positions.pop()
                self.value_func.undo(pos)
            # --- Simulation (using Alpha-Beta) ---

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_rewards = sum(abs(child.total_reward) for child in root.children)
        distribution = np.zeros(len(root.children))
        best_visits = -10000
        best_action = None
        for action, child in enumerate(root.children):
            distribution[action] = child.total_reward / total_rewards if total_rewards > 0 else 0
            if child.total_reward > best_visits:
                best_visits = child.total_reward
                best_action = root.move_str[action]
        return best_action, distribution
    




if __name__=="__main__":
    train_process=TD_learn()
    dataset=train_process.load_sgf_file()
    # train_process.load_json_file()
    train_process.selfplay(dataset)
    train_process.save_model_weights()