import sys
import numpy as np
import random
from train_c6 import TD_learn


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def get_rel_zone(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        relevance_zone=set()
        radius=3
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    for dr, dc in directions:
                        for rad in range(-radius,radius+1):
                            nr=r+rad*dr
                            nc=c+rad*dc
                            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == 0:
                                relevance_zone.add((nr, nc))
        return relevance_zone
    def check_line(self, x, y,color):
        line = []
        features=[0,0,0,0,0,0,0]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            dir_feature=[]
            for i in range(-4, 4):  # 向前和向後檢查最多5個位置
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    dir_feature.append(self.board[nx][ny])
                else:
                    dir_feature.append('2')  # 超出邊界
            line.append(dir_feature)


        # 定義各種棋形的模式

        threat1_patterns = ['2011110','101110','110110','111010','2111100',
                          '011101','101101','110101','111001','111001',
                          '2111110','2101111','2110111','2111011','2111101']
        threat2_patterns=['00111100','0111110','10111100','10111101']
        live3_patterns = ['011100','010110']
        dead3_patterns = ['2111000', '2110100','2101100','2100110']
        live2_patterns = ['001100', '010100',]
        dead2_patterns = ['2110000', '2011000', '2001100', '2000110']
        live1_patterns = ['0100000', '0010000', '0001000'] 
        for dir in line:       
        # 將檢測到的線轉換為字符串，方便匹配模式
            if color=='B':
                line_str = ''.join(['1' if cell == 1 else '0' if cell == 0 else '2' for cell in dir])
            else:
                line_str = ''.join(['1' if cell == 2 else '0' if cell == 0 else '1' for cell in dir])

            if any(pattern in line_str for pattern in threat1_patterns):
                features[0]+=1
            if any(pattern[::-1] in line_str for pattern in threat1_patterns):
                features[0]+=1
            if any(pattern in line_str for pattern in threat2_patterns):
                features[1]+=1
            if any(pattern[::-1] in line_str for pattern in threat2_patterns):
                features[1]+=1
            if any(pattern in line_str for pattern in live3_patterns):
                features[2]+=1
            if any(pattern[::-1] in line_str for pattern in live3_patterns):
                features[2]+=1
            if any(pattern in line_str for pattern in dead3_patterns):
                features[3]+=1
            if any(pattern[::-1] in line_str for pattern in dead3_patterns):
                features[3]+=1
            if any(pattern in line_str for pattern in live2_patterns):
                features[4]+=1
            if any(pattern[::-1] in line_str for pattern in live2_patterns):
                features[4]+=1            
            if any(pattern in line_str for pattern in dead2_patterns):
                features[5]+=1
            if any(pattern[::-1] in line_str for pattern in dead2_patterns):
                features[5]+=1
            if any(pattern in line_str for pattern in live1_patterns):
                features[6]+=1
            if any(pattern[::-1] in line_str for pattern in live1_patterns):
                features[6]+=1
        return features
    def get_feature(self):
        features=[0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 1:
                    feature=self.check_line(r,c,color='B')
                    for index,count in enumerate(feature):
                        features[index]+=count
                elif self.board[r, c] == 2:
                    feature=self.check_line(r,c,color='W')
                    for index,count in enumerate(feature):
                        features[index+7]+=count
        return features

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
                            return current_color
        return 0
    
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
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        selected = random.sample(empty_positions, 1)
        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
