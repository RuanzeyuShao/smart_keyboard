import numpy as np 
from functools import lru_cache
import time 

class Go(): 
    """
    illegal moves 
    -1: out of bounds 
    -2: existing stone 
    -3: ko 
    -4: suicide 
    -5: game end 
    """
    
    def __init__(self, end_num = 1000): 
        self.board = np.zeros([19,19]).astype(int)
        self.turn = 1 #黑1 白0
        self.end_num = end_num 
        self.prevpass = False 
        self.borders = 19 
        self.visit = np.zeros([19,19]).astype(int) #调用liberties函数之前一定要先重置visit
        self.last = np.zeros([19,19]).astype(int) 
        self.last2 = np.zeros([19,19]).astype(int) 
        self.game_end = False 
        self.hist = np.array([[-5,-5]]).astype(int) 
        self.board_visit = np.zeros([19,19]).astype(int)
        
    def place(self, cord): #cord [1,19]
        self.cord1 = np.array([cord[0]-1, cord[1]-1]) 
        if self.game_end == True: 
            return -5 
        
        if self.cord1[0] > 18 or self.cord1[1] > 18 or self.cord1[0] < 0 or self.cord1[1] < 0: 
        
            if np.all(cord == np.array([-1,-1])): #Pass
                self.turn = 1-self.turn 
                if self.prevpass: 
                    self.score() 
                self.prevpass = True 
                return 0 

            return -1 
        
        elif self.board[self.cord1[0], self.cord1[1]] != 0: 
            return -2
        
        else: 
            self.capture_any = False #whether if any is captured 
            self.board[self.cord1[0], self.cord1[1]] = self.turn*2-1 
            self.prevpass = False
            # update board first 
            self.captive = np.zeros([19,19]).astype(int) 
            nb = self.neighbour(self.cord1)
            for i in range(nb.shape[0]): 
                
                if self.board[nb[i][0], nb[i][1]] == -self.board[self.cord1[0], self.cord1[1]]: 
                    self.visit = np.zeros([19,19]).astype(int)
                    self.capture_neighbour, self.captured_pieces = self.isalive(nb[i])
                    
                    if self.capture_neighbour == False: # whether if the neighbour we are looking at is captured 
                        self.capture_any = True 
                        self.captive = np.min([np.ones([19,19]).astype(int),self.captive + self.captured_pieces], axis = 0)
            
            if self.capture_any: 
                self.temp_board = self.board - self.captive*(1-self.turn*2) 
                if self.superko(): 
                    #capture+ko 
                    #ko appears after all pieces have captured 
                    self.turn = 1-self.turn
                    self.board = self.temp_board.astype(int) 
                    self.last2 = np.copy(self.last.astype(int)) 
                    self.last = np.copy(self.board.astype(int)) 
                    self.hist = np.append(self.hist, [[self.cord1[0]+1, self.cord1[1]+1]], axis = 0) 
                    return 0
                
                else: 
                    self.board[self.cord1[0], self.cord1[1]] = 0
                    return -3 
            
            elif self.hist.shape[0] > self.end_num: 
                self.score() 
                return 0 
                
            else: 
                self.visit = np.zeros([19,19]).astype(int) 
                nb = self.neighbour(self.cord1)

                if np.any(self.board[nb[:,0],nb[:,1]] == 0):
                    self.this_liberties = True 
                else: 
                    self.this_liberties, _ = self.isalive(self.cord1)
                if self.this_liberties == False:
                    self.board[self.cord1[0], self.cord1[1]] = 0 
                    #take history 
                    return -4 
                
                else: 
                    self.last2 = np.copy(self.last.astype(int)) 
                    self.last = np.copy(self.board.astype(int))
                    self.hist = np.append(self.hist, [[self.cord1[0]+1, self.cord1[1]+1]], axis = 0) 
                    self.turn = 1-self.turn
                    return 0 


    def superko(self): 
        no_ko = True 
        if np.all(self.temp_board == self.last2): 
            no_ko = False 
                
        return no_ko 
    
    def score(self, komi = 7.5, estimate = False): 
        #数子规则 贴7.5目，原因是懒得算提子数 
        self.board_visit = np.zeros([19,19]).astype(int) 
        self.score_black = np.zeros([19,19]).astype(int) 
        self.score_white = np.zeros([19,19]).astype(int) 
        self.visit = np.zeros([19,19]).astype(int) 

        for x in range(19): 
            for y in range(19): 
                if self.board_visit[x, y] == 0 and self.board[x, y] == 1: 
                    self.score_black[x, y] = 1 
                    self.board_visit[x, y] = 1
                    
                elif self.board_visit[x, y] == 0 and self.board[x, y] == -1: 
                    self.score_white[x, y] = 1
                    self.board_visit[x, y] = 1
                    
                elif self.board_visit[x, y] == 0 and self.board[x, y] == 0: 
                    self.visit = np.zeros([19,19]).astype(int) 
                    self.influence_black, self.influence_white, self.local_visit = self.influence(np.array([x,y])) 
                    self.board_visit = self.board_visit.astype(int) + self.local_visit.astype(int) 
                    if self.influence_black == True and self.influence_white == True:
                        pass 
                    elif self.influence_black == True and self.influence_white == False: 
                        self.score_black = self.score_black.astype(int)  + self.local_visit.astype(int) 
                    elif self.influence_black == False and self.influence_white == True: 
                        self.score_white = self.score_white.astype(int)  + self.local_visit.astype(int) 

        self._sum_w = np.sum(self.score_white.astype(int)) + komi 
        self._sum_b = np.sum(self.score_black.astype(int))
        
        #print("Black: {}, White: {}".format(self._sum_b, self._sum_w))

        if self._sum_b > self._sum_w: 
            #print("Black Wins") 
            self.wl = 1
        elif self._sum_b < self._sum_w: 
            #print("White Wins") 
            self.wl = 0 
        if estimate == False:
            self.game_end = True 
    
    def influence(self, cord, b = False, w = False): 
        self.visit[cord[0], cord[1]] = 1         
        if self.board[cord[0],cord[1]] == 1: 
            return True, False, self.visit.astype(int)
        elif self.board[cord[0],cord[1]] == -1: 
            return False, True, self.visit.astype(int) 
        else:
            nb = self.neighbour(cord).astype(int)
            for i in range(nb.shape[0]): 
                if nb[i][0] < 0 or nb[i][0] > 18 or nb[i][1] < 0 or nb[i][1] > 18: 
                    return b, w, self.visit.astype(int)
                if self.board[nb[i][0],nb[i][1]] == 0 and self.visit[nb[i][0],nb[i][1]] == 0 and self.board_visit[nb[i][0],nb[i][1]] == 0: 
                    b, w, _ = self.influence(nb[i], b = b, w = w) 
                
                elif self.board[nb[i][0],nb[i][1]] == 1 and self.visit[nb[i][0],nb[i][1]] == 0 and self.board_visit[nb[i][0],nb[i][1]] == 0: 
                    b = True 
                
                elif self.board[nb[i][0], nb[i][1]] == -1 and self.visit[nb[i][0],nb[i][1]] == 0: 
                    w = True
            
        return b, w, self.visit.astype(int)

    def isalive(self, cord, alive = False):
        self.visit[cord[0], cord[1]] = 1 
        nb = self.neighbour(cord).astype(int)
        for i in range(nb.shape[0]): 
            if nb[i][0] < 0 or nb[i][0] > 18 or nb[i][1] < 0 or nb[i][1] > 18: 
                return alive, self.visit.astype(int)
            
            if self.visit[nb[i][0], nb[i][1]] == 0 and self.board[nb[i][0], nb[i][1]] == self.board[cord[0], cord[1]]: 
                alive, _ = self.isalive(nb[i], alive = alive)
                
            elif self.visit[nb[i][0], nb[i][1]] == 0 and self.board[nb[i][0], nb[i][1]] == 0: 
                return True, self.visit.astype(int)
            
        return alive, self.visit.astype(int) 
            
    def neighbour(self, cord): 
        nei = np.array([[np.min([18,cord[0]+1]), cord[1]],
                        [np.max([0,cord[0]-1]), cord[1]],
                        [cord[0], np.min([18, cord[1]+1])],
                        [cord[0], np.max([0, cord[1]-1])]]) 
        neib = np.unique(nei, axis = 0)
        return neib.astype(int)

def e_territory(board): 
    terr = np.zeros([19, 19, 8], dtype = int) 
    empty = np.array(board == 0, dtype = int) 

    ry0 = np.roll(board, 1, axis = 0) 
    ry0[0, :] = np.zeros([1, 19])  

    ry1 = np.roll(board, -1, axis = 0) 
    ry1[-1, :] = np.zeros([1, 19])  

    rx0 = np.roll(board, 1, axis = 1) 
    rx0[:, 0] = np.zeros([19, 1])[:, 0]  

    rx1 = np.roll(board, -1, axis = 1) 
    rx1[:, -1] = np.zeros([19, 1])[:, 0]  

    rx0y0 = np.roll(rx0, 1, axis = 0) 
    rx0y0[0, :] = np.zeros([1, 19])  

    rx0y1 = np.roll(rx0, -1, axis = 0) 
    rx0y1[-1, :] = np.zeros([1, 19])  

    rx1y0 = np.roll(rx1, 1, axis = 0) 
    rx1y0[0, :] = np.zeros([1, 19])  
    
    rx1y1 = np.roll(rx1, -1, axis = 0) 
    rx1y1[-1, :] = np.zeros([1, 19])  

    terr[:,:,0] = ry0*empty 
    terr[:,:,1] = ry1*empty 
    terr[:,:,2] = rx0*empty 
    terr[:,:,3] = rx1*empty 
    terr[:,:,4] = rx0y0*empty 
    terr[:,:,5] = rx0y1*empty 
    terr[:,:,6] = rx1y0*empty 
    terr[:,:,7] = rx1y1*empty 

    return terr 

class calc_liberties():
    def __init__(self, board):
        self.visit = np.zeros([19,19]).astype(int)
        self.num_liberties = np.zeros([19,19]).astype(int)
        self.b = board
        for x in range(19):
            for y in range(19):
                if self.b[x, y] == 0:
                    self.visit[x, y] = 1
                elif self.visit[x, y] == 0:
                    self.recursion_visit = np.zeros([19,19])
                    self.num_liberties[x, y] = np.min([self.recursion(np.array([x, y])), 8])
                    self.assign(np.array([x, y]), self.num_liberties[x, y])

    def assign(self, cord, lib):
        self.visit[cord[0], cord[1]] = 1
        nb = self.neighbour_lib(cord).astype(int)
        for i in range(nb.shape[0]):
            if self.visit[nb[i][0], nb[i][1]] == 0 and self.b[nb[i][0], nb[i][1]] == self.b[cord[0], cord[1]]:
                self.num_liberties[nb[i][0], nb[i][1]]  = self.num_liberties[cord[0], cord[1]]
                self.assign(np.array([nb[i][0], nb[i][1]]), self.num_liberties[cord[0], cord[1]])

    def recursion(self, cord):
        self.recursion_visit[cord[0], cord[1]] = 1
        nb = self.neighbour_lib(cord).astype(int)
        libs_for_stone = 0
        for i in range(nb.shape[0]):
            if self.b[nb[i][0], nb[i][1]] == 0:
                libs_for_stone += 1
            elif self.recursion_visit[nb[i][0], nb[i][1]] == 0 and self.b[nb[i][0], nb[i][1]] == self.b[cord[0], cord[1]]:
                libs_for_stone += self.recursion(nb[i])
        return libs_for_stone


    def neighbour_lib(self, cord):
        nei = np.array([[np.min([18,cord[0]+1]), cord[1]],
                        [np.max([0,cord[0]-1]), cord[1]],
                        [cord[0], np.min([18, cord[1]+1])],
                        [cord[0], np.max([0, cord[1]-1])]])
        neib = np.unique(nei, axis = 0)
        return neib.astype(int)
