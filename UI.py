from Board import *
from Player_Conv import *
import tkinter as tk
import numpy as np 
from copy import deepcopy 

#from Player_Val import *

#Filter mask 
#作用是为了判断真眼，防止自填眼位，四个角需要周边有三颗棋子才能叫真眼，边上需要五颗，中腹需要七颗。
mask = np.ones([19,19], dtype = np.uint8)*7 
mask[0,:] -= 2 
mask[18,:] -= 2 
mask[:,0] -= 2 
mask[:,18] -= 2 

mask3 = np.ones([19,19], dtype = np.uint8)*4 
mask3[0,:] -= 1 
mask3[18,:] -= 1 
mask3[:,0] -= 1 
mask3[:,18] -= 1 

my0 = np.roll(mask-1, 1, axis = 0) 
my0[0, :] = np.zeros([1, 19])
    
my1 = np.roll(mask-1, -1, axis = 0) 
my1[-1, :] = np.zeros([1, 19])
    
mx0 = np.roll(mask-1, 1, axis = 1) 
mx0[:, 0] = np.zeros([19, 1])[:, 0]  
    
mx1 = np.roll(mask-1, -1, axis = 1) 
mx1[:, -1] = np.zeros([19, 1])[:, 0]  
    
mx0y0 = np.roll(mx0, 1, axis = 0) 
mx0y0[0, :] = np.zeros([1, 19])
    
mx0y1 = np.roll(mx0, -1, axis = 0) 
mx0y1[-1, :] = np.zeros([1, 19])
    
mx1y0 = np.roll(mx1, 1, axis = 0) 
mx1y0[0, :] = np.zeros([1, 19])
    
mx1y1 = np.roll(mx1, -1, axis = 0) 
mx1y1[-1, :] = np.zeros([1, 19]) 
###### 

class ClickBoard():
    def __init__(self, game = Go()): 
        self.a = game
        self.hint_pos = None  # 初始化提示位置


        if np.all(self.a.board == np.zeros([19,19])):
            self.rules()

        self.AI_policy = Conv()
        #self.AI_val = Conv_Val()

        self.root = tk.Tk()
        self.root.configure(bg = "#e0c267")
        self.root.title("围棋人机对战")
        self.h = self.root.winfo_screenheight()
        self.hr1 = 0.9 
        self.hr2 = 0.8 

        self.fr0 = tk.Frame(self.root)
        self.fr = tk.Frame(self.root)
        self.fr1 = tk.Frame(self.root)

        self.x0 = np.linspace(0,self.h*self.hr1*self.hr2,19)
        self.stonesize = self.x0[1]-self.x0[0]

        self.txt = tk.StringVar()
        self.txt.set("黑方行棋")
        self.lbl = tk.Label(self.fr0, textvariable = self.txt, font=("微软雅黑", 18, 'bold'), bg = "#e0c267")
        self.prevpass = False 
        # === 防止AI回合期间用户连点造成“黑白一起被人工下掉” ===
        self.ai_busy = False        # True表示AI回合/AI正在思考，禁止用户点击落子
        self.ai_after_id = None     # 记录after任务id，避免重复排队
        self.xshift = self.stonesize/2
        self.yshift = self.stonesize/2

        self.w = tk.Canvas(self.fr, height = self.h*self.hr1*self.hr2+self.stonesize, width = self.h*self.hr1*self.hr2+self.stonesize, bg = "#e0c267")
        self.stones = tk.Canvas(self.w, height = self.h*self.hr1*self.hr2+self.stonesize, width = self.h*self.hr1*self.hr2+self.stonesize)

        self._prev = tk.Button(self.fr1, text = "悔棋", command = self.prev_mov, font=("微软雅黑", 15), bg = '#c09966', bd = 1, pady = 1, padx = 10*1.618, activebackground = "#deb887")
        self._pass = tk.Button(self.fr1, text = "停一手", command = self.pass_btn, font=("微软雅黑", 15), bg = '#c09966', bd = 1, pady = 1, padx = 10*1.618, activebackground = "#deb887")
        self._count = tk.Button(self.fr1, text = "估算分数", command = self.board_count, font=("微软雅黑", 15), bg = '#c09966', bd = 1, pady = 1, padx = 10*1.618, activebackground = "#deb887")
        self._AI_mov = tk.Button(self.fr1, text="落子提示", command=self.show_hint, font=("微软雅黑", 15), bg='#c09966', bd=1, pady=1, padx=10*1.618, activebackground="#deb887")
        #self._AI_mov = tk.Button(self.fr1, text = "召唤小狗", command = self.AI_Move, font=("微软雅黑", 15), bg = '#c09966', bd = 1, pady = 1, padx = 10*1.618, activebackground = "#deb887")
        #self._Value = tk.Button(self.fr1, text = "查看胜率", command = self.AI_Value, font=("微软雅黑", 15), bg = '#c09966', bd = 1, pady = 1, padx = 10*1.618, activebackground = "#deb887")

        self.draw_board()

        self.w.bind("<ButtonRelease-1>", self.callback)
        self.lbl.pack()
        self.w.pack()
        #self._Value.pack(side = 'right')
        self._AI_mov.pack(side = 'right')
        self._count.pack(side = 'right')
        self._pass.pack(side = 'right')
        self._prev.pack(side = 'right')
        self.fr0.pack()
        self.fr.pack()
        self.fr1.pack()

        if np.any(self.a.board != np.zeros([19,19])):
            self.plot_stones()

        self.root.mainloop()

    def draw_board(self, dotsize = 4):
        z0 = np.ones(19)*0
        z1 = np.ones(19)*self.h*self.hr1*self.hr2
        for i in range(19):
            self.w.create_line(self.x0[i]+self.xshift, z0[i]+self.yshift, self.x0[i]+self.xshift, z1[i]+self.yshift, fill = 'black')
            self.w.create_line(z0[i]+self.xshift, self.x0[i]+self.yshift, z1[i]+self.xshift, self.x0[i]+self.yshift, fill = 'black')
        dots = np.array([3,9,15])

        for d0 in dots:
            for d1 in dots:
                self.w.create_oval(self.xshift+self.x0[d0]-dotsize, self.yshift+self.x0[d1]-dotsize,
                                                 self.xshift+self.x0[d0]+dotsize, self.yshift+self.x0[d1]+dotsize, fill = 'black', outline = 'black')

    # ---------- 输入锁：AI回合期间禁止用户落子 ----------
    def _lock_input(self):
        self.ai_busy = True

    def _unlock_input(self):
        self.ai_busy = False
        self.ai_after_id = None

    def _schedule_ai_move(self, delay_ms=500):
        """安排AI走子（并锁定输入，避免after期间用户连点）。"""
        self._lock_input()
        # 避免重复排队（用户误操作/多次触发）
        if self.ai_after_id is not None:
            try:
                self.root.after_cancel(self.ai_after_id)
            except Exception:
                pass
            self.ai_after_id = None
        self.ai_after_id = self.root.after(delay_ms, self.AI_Move)

    def callback(self, event):
        # AI回合/AI正在思考时，忽略所有用户点击
        if self.ai_busy or self.a.turn == 0:
            return
        coorx = event.x - self.xshift
        coory = event.y - self.yshift
        boardx = int(np.round(coorx / self.stonesize)) + 1
        boardy = int(np.round(coory / self.stonesize)) + 1

        if 1 <= boardx <= 19 and 1 <= boardy <= 19:
            indic = self.a.place(np.array([boardx, boardy]))
            if indic == 0:
                self.plot_stones()
                if self.a.turn == 1:
                    self.txt.set("黑方行棋")
                elif self.a.turn == 0:
                    self.txt.set("白方行棋")
                    self._schedule_ai_move(500)

                # 检查用户落子是否与提示位置一致
                if self.hint_pos and (boardx, boardy) != self.hint_pos:
                    self.w.delete("hint")  # 如果落子位置与提示不同，清除提示
                    self.hint_pos = None  # 置空提示位置
            elif indic == -3:
                self.txt.set("打劫禁入点")
            elif indic == -4:
                self.txt.set("禁止自杀")




    def prev_mov(self):
        # 悔棋时如果之前已经安排了AI走子，先取消，避免悔棋后AI还在落子
        if self.ai_after_id is not None:
            try:
                self.root.after_cancel(self.ai_after_id)
            except Exception:
                pass
            self.ai_after_id = None
        self._unlock_input()
        # 如果历史记录中的步数大于2，则撤销最后两步（白棋和黑棋）
        if self.a.hist.shape[0] > 2:
            # 删除历史中的最后两步
            hist = np.delete(self.a.hist, [-1, -2], axis=0)
            self.a = Go()
            # 重新播放从第一步开始的所有历史记录
            for i in range(1, hist.shape[0]):
                self.a.place(hist[i])
            # 强制设定轮到黑棋（假设黑棋用1表示）
            self.a.turn = 1  
            self.txt.set("黑方行棋")
        else:
            self.a = Go()
            self.txt.set("黑方行棋")
        self.plot_stones()


    def pass_btn(self):
        # AI回合/AI正在思考时，忽略停一手（避免把白棋也当成人工下）
        if self.ai_busy or self.a.turn == 0:
            return
        self.a.place(np.array([-1, -1]))
        # 如果游戏结束，直接显示结果
        if self.a.game_end == True:
            bw = ""
            if self.a.wl == 1:
                bw = "黑"
            elif self.a.wl == 0:
                bw = "白"
            self.txt.set("棋局结束：黑棋 {}子，白棋 {}子：{}胜".format(self.a._sum_b, self.a._sum_w, bw))
        else:
            # 如果未结束，根据当前轮次更新提示文字
            if self.a.turn == 1:
                self.txt.set("黑方行棋")
            elif self.a.turn == 0:
                self.txt.set("白方行棋")
                # 延时500毫秒后自动调用 AI_Move 让 AI 落子
                self._schedule_ai_move(500)


    def board_count(self):
        self.a.score(estimate = True)
        self.txt.set("盘面计分：黑棋 {}子，白棋 {}子".format(self.a._sum_b, self.a._sum_w))

    def plot_stones(self):
        self.w.delete("stones")
        for _i in range(19):
            for _j in range(19):
                if self.a.board[_i,_j] == 1:
                    self.w.create_oval(self.x0[_i] + self.xshift-self.stonesize*0.9/2, self.x0[_j] + self.yshift-self.stonesize*0.9/2,
                                       self.x0[_i] + self.xshift+self.stonesize*0.9/2, self.x0[_j] + self.yshift+self.stonesize*0.9/2,
                                       fill = 'black', tag = 'stones')
                if self.a.board[_i,_j] == -1:
                    self.w.create_oval(self.x0[_i] + self.xshift-self.stonesize*0.9/2, self.x0[_j] + self.yshift-self.stonesize*0.9/2,
                                       self.x0[_i] + self.xshift+self.stonesize*0.9/2, self.x0[_j] + self.yshift+self.stonesize*0.9/2,
                                       fill = 'white', tag = 'stones')
    def rules(self):
        self.rule_txt = tk.Tk() 
        self.rule_txt.title("开始窗口")
        self.txtbox = tk.Text(self.rule_txt, font = ("微软雅黑", 15), height = 10, width = 30)
        self.txtbox.insert(tk.END, "围棋人机对战 \n")
        self.txtbox.insert(tk.END, "计分规则：中式数子规则 \n")
        self.txtbox.insert(tk.END, "贴目规则：黑贴7.5目 \n")
        self.txtbox.insert(tk.END, "打劫规则：禁止出现上一步的局面 \n\n")

        self.txtbox.insert(tk.END, "关闭本窗口开始下棋 \n") 

        self.txtbox.pack()
        self.rule_txt.mainloop() 
    
    def show_hint(self):
        self.w.delete("hint")  # 清除之前的提示
        self.hint_pos = None  # 记录提示的位置，初始为空

        AI_Move_inX = AI_prob_calc(self.a)
        self.AI_policy.forward(AI_Move_inX)
        outp = np.array(deepcopy(self.AI_policy.p[0]))

        # 应用过滤逻辑
        mask_self = np.sum(AI_Move_inX[0, 2:21, 2:21, 28:36], axis=2)
        mask_self1 = np.sum(AI_Move_inX[0, 2:21, 2:21, 28:32], axis=2)

        ry0 = np.roll(mask_self, 1, axis=0)
        ry0[0, :] = np.zeros([1, 19])
        ry1 = np.roll(mask_self, -1, axis=0)
        ry1[-1, :] = np.zeros([1, 19])
        rx0 = np.roll(mask_self, 1, axis=1)
        rx0[:, 0] = np.zeros([19, 1])[:, 0]
        rx1 = np.roll(mask_self, -1, axis=1)
        rx1[:, -1] = np.zeros([19, 1])[:, 0]
        rx0y0 = np.roll(rx0, 1, axis=0)
        rx0y0[0, :] = np.zeros([1, 19])
        rx0y1 = np.roll(rx0, -1, axis=0)
        rx0y1[-1, :] = np.zeros([1, 19])
        rx1y0 = np.roll(rx1, 1, axis=0)
        rx1y0[0, :] = np.zeros([1, 19])
        rx1y1 = np.roll(rx1, -1, axis=0)
        rx1y1[-1, :] = np.zeros([1, 19])

        mask2 = (mask_self >= mask - 1) * np.minimum(np.ones([19, 19]),
                    (rx0y0 >= mx0y0) + (rx0y1 >= mx0y1) +
                    (rx1y0 >= mx1y0) + (rx1y1 >= mx1y1)) * (mask_self1 >= mask3)
        mask1 = (mask_self >= mask) * (mask_self1 >= mask3)
        mask_out = 1 - np.reshape(np.minimum(np.ones([19, 19]), mask1 + mask2), [-1])

        outp = outp * mask_out
        if np.sum(outp) == 0:
            return
        outp = outp / np.sum(outp)

        # 寻找合法位置并绘制提示
        for _ in range(361):
            AI_out = np.argmax(outp)
            AI_y = np.mod(AI_out, 19)
            AI_x = int((AI_out - AI_y) / 19)
            AI_place = [AI_x + 1, AI_y + 1]

            # 模拟落子检查合法性
            simulated_go = deepcopy(self.a)
            indic = simulated_go.place(AI_place)
            if indic == 0:
                x_pos = self.x0[AI_x] + self.xshift
                y_pos = self.x0[AI_y] + self.yshift
                self.w.create_oval(
                    x_pos - self.stonesize * 0.4, y_pos - self.stonesize * 0.4,
                    x_pos + self.stonesize * 0.4, y_pos + self.stonesize * 0.4,
                    fill='#00FF00', outline='#00FF00', tag='hint'
                )
                self.hint_pos = (AI_x + 1, AI_y + 1)  # 记录提示的位置
                break
            else:
                outp[AI_out] = 0
                if np.sum(outp) == 0:
                    break
                outp = outp / np.sum(outp)


    def AI_Move(self):
        # AI开始走子：锁定输入，防止用户在AI回合继续落子
        self._lock_input()
        AI_Move_inX = AI_prob_calc(self.a) 
        self.AI_policy.forward(AI_Move_inX) 

        self.outp = np.array(deepcopy(self.AI_policy.p[0]))
        #Filter 
        mask_self = np.sum(AI_Move_inX[0,2:21,2:21,28:36], axis = 2) # shape = (19, 19) 
        mask_self1 = np.sum(AI_Move_inX[0,2:21,2:21,28:32], axis = 2) # shape = (19, 19) 
        
        ry0 = np.roll(mask_self, 1, axis = 0) 
        ry0[0, :] = np.zeros([1, 19])  
    
        ry1 = np.roll(mask_self, -1, axis = 0) 
        ry1[-1, :] = np.zeros([1, 19])  
    
        rx0 = np.roll(mask_self, 1, axis = 1) 
        rx0[:, 0] = np.zeros([19, 1])[:, 0]  
    
        rx1 = np.roll(mask_self, -1, axis = 1) 
        rx1[:, -1] = np.zeros([19, 1])[:, 0]  
    
        rx0y0 = np.roll(rx0, 1, axis = 0) 
        rx0y0[0, :] = np.zeros([1, 19])  
    
        rx0y1 = np.roll(rx0, -1, axis = 0) 
        rx0y1[-1, :] = np.zeros([1, 19])  
    
        rx1y0 = np.roll(rx1, 1, axis = 0) 
        rx1y0[0, :] = np.zeros([1, 19]) 
    
        rx1y1 = np.roll(rx1, -1, axis = 0) 
        rx1y1[-1, :] = np.zeros([1, 19])  
        
        mask2 = np.array(mask_self >= mask-1)*np.min([np.ones([19,19]), 
                                                      np.array(rx0y0 >= mx0y0)+np.array(rx0y1 >= mx0y1)+
                                                      np.array(rx1y0 >= mx1y0)+np.array(rx1y1 >= mx1y1)], axis = 0)*np.array(mask_self1 >= mask3) 
        #* == and, + == or 
        
        mask1 = np.array(mask_self >= mask)*np.array(mask_self1 >= mask3)

        mask_out = 1-np.reshape(np.min([np.ones([19,19]), mask1 + mask2], axis = 0), [-1]) # then multiply 
        self.outp = self.outp*mask_out              
        self.outp = self.outp/np.sum(self.outp) 

        for i_p in range(361): 

            #if self.a.hist.shape[0] > 50: 
            #    self.AI_out = np.argmax(self.outp) 
            #else: 
            #    self.AI_out = np.random.choice(np.arange(361), p = self.outp)
            

            if np.sum(self.outp) == 0:
                self.a.place([-1,-1]) 
                if self.prevpass == False: 
                    self.prevpass = True 
                    if self.a.turn == 1: 
                        self.txt.set("黑方行棋")
                    elif self.a.turn == 0:
                        self.txt.set("白方行棋") 
                else: 
                    self.a.score() 
                    if self.a.game_end == True:
                        bw = ""
                        if self.a.wl == 1:
                            bw = "黑"
                        elif self.a.wl == 0:
                            bw = "白"
                        self.txt.set("棋局结束：黑棋 {}子，白棋 {}子：{}胜".format(self.a._sum_b, self.a._sum_w, bw))
                break 
            self.AI_out = np.argmax(self.outp) #策略走权重最大值 
            #self.AI_out = np.random.choice(np.arange(361), p = self.outp) #随机策略权重
            self.AI_y = np.mod(self.AI_out, 19)
            self.AI_x = int((self.AI_out-self.AI_y)/19)
            self.AI_place = [self.AI_x+1, self.AI_y+1] 
            indic = self.a.place(self.AI_place) 

            if indic == 0:
                self.plot_stones()
                if self.a.turn == 1: 
                    self.txt.set("黑方行棋")
                elif self.a.turn == 0:
                    self.txt.set("白方行棋")
                break
            else:
                self.outp[self.AI_out] = 0
                if np.sum(self.outp) == 0:
                    self.a.place([-1,-1]) 
                    if self.prevpass == False: 
                        self.prevpass = True 
                        if self.a.turn == 1: 
                            self.txt.set("黑方行棋")
                        elif self.a.turn == 0:
                            self.txt.set("白方行棋") 
                    else: 
                        self.a.score() 
                        if self.a.game_end == True:
                            bw = ""
                            if self.a.wl == 1:
                                bw = "黑"
                            elif self.a.wl == 0:
                                bw = "白"
                            self.txt.set("棋局结束：黑棋 {}子，白棋 {}子：{}胜".format(self.a._sum_b, self.a._sum_w, bw))
                    break 
                else: 
                    self.outp = self.outp*mask_out 
                    self.outp = self.outp/np.sum(self.outp) 

        # AI走子结束：解除输入锁
        self._unlock_input()
    """
    def AI_Value(self):
        AI_Move_inX = AI_prob_calc(self.a)
        self.AI_val.forward(AI_Move_inX)

        self.wl_prob = (1+tf.reduce_sum(self.AI_val.v)*(2*self.a.turn-1))/2
        self.txt.set("胜率估算：黑棋 {:.1f}%, 白棋 {:.1f}%".format(100*self.wl_prob, 100-100*self.wl_prob))
    """ 

cpuct = 3
max_rec = 20
max_child_node = 5

class Node():
    def __init__(self, go, parent = None, index = None): #if root node, no index required
        self.parent = parent
        self.N = np.zeros([361])
        self.child = np.zeros([361], dtype = object)
        self.p = AI_prob_calc(go)
        self.v = np.zeros([361])
        self.index = index
        self.go = deepcopy(go)
        self.simulate()
        self.UCB = self.v + cpuct*self.p/(1+self.N)*np.sqrt(np.sum(self.N)) 
        self.backprop(self.wl_sim)

    def simulate(self):
        x = np.arange(1, 20)
        y = np.arange(1, 20)
        cords = np.transpose([np.tile(x, y.shape[0]), np.repeat(y, x.shape[0])])
        np.random.shuffle(cords)
        sim = deepcopy(self.go)
        for i in range(cords.shape[0]):
            sim.place(cords[i])
        sim.score(estimate = True)
        self.wl_sim = 2*sim.wl-1

    def backprop(self, wl):
        if self.parent != None: 
            self.parent.v[self.index] = (self.parent.v[self.index]*self.parent.N[self.index] + wl*(2*self.parent.go.turn-1))/(1+self.parent.N[self.index]) 
            self.parent.N[self.index] += 1
            self.parent.backprop(wl)

    def expand(self, go = None, k = 0):
        if k <= max_rec:
            k += 1
            next_index = np.argmax(self.UCB)
            if go == None:
                nxt_go = deepcopy(self.go)
            else:
                nxt_go = deepcopy(go)
            y = np.mod(next_index, 19)
            x = int((next_index - y) / 19)
            nxt_go.place([x + 1, y + 1])
            if np.sum(self.N != 0) >= max_child_node:
                self.UCB = self.UCB * np.array(self.N != 0, dtype = float)
            else:
                if self.child[next_index] == 0:
                    self.child[next_index] = Node(nxt_go, parent = self, index = next_index)
                else:
                    self.child[next_index].expand(deepcopy(nxt_go), k = k)

class Tree():
    def __init__(self, go):
        self.Node = Node(go)
        for i in range(max_rec):
            self.Node.expand(go)

def AI_prob_calc(a, retinX = False): #a is a Go() type object
    AI_board = np.array([deepcopy(a.board)])
    AI_turn = np.array([deepcopy(a.turn)])
    AI_libs_class = calc_liberties(AI_board[0])
    AI_libs = np.zeros([1, 19, 19])
    AI_terr = np.zeros([1, 19, 19, 8]) 
    AI_libs[0] = AI_libs_class.num_liberties * AI_board[0]
    if a.hist.shape[0] == 1:
        AI_hist_1 = [-5, -5]
    else:
        AI_hist_1 = deepcopy(a.hist[-1])
    if a.hist.shape[0] <= 2:
        AI_hist_2 = [-5, -5]
    else:
        AI_hist_2 = deepcopy(a.hist[-2])
    if a.hist.shape[0] <= 3:
        AI_hist_3 = [-5, -5]
    else:
        AI_hist_3 = deepcopy(a.hist[-3])
    if a.hist.shape[0] <= 4:
        AI_hist_4 = [-5, -5]
    else:
        AI_hist_4 = deepcopy(a.hist[-4])
    if a.hist.shape[0] <= 5:
        AI_hist_5 = [-5, -5]
    else:
        AI_hist_5 = deepcopy(a.hist[-5])
    if a.hist.shape[0] <= 6:
        AI_hist_6 = [-5, -5]
    else:
        AI_hist_6 = deepcopy(a.hist[-6])
    if a.hist.shape[0] <= 7:
        AI_hist_7 = [-5, -5]
    else:
        AI_hist_7 = deepcopy(a.hist[-7])
    if a.hist.shape[0] <= 8:
        AI_hist_8 = [-5, -5]
    else:
        AI_hist_8 = deepcopy(a.hist[-8])

    AI_terr[0] = e_territory(deepcopy(a.board)) 
    AI_hist_board_1 = np.zeros([1, 19, 19])
    AI_hist_board_2 = np.zeros([1, 19, 19])
    AI_hist_board_3 = np.zeros([1, 19, 19])
    AI_hist_board_4 = np.zeros([1, 19, 19])
    AI_hist_board_5 = np.zeros([1, 19, 19])
    AI_hist_board_6 = np.zeros([1, 19, 19])
    AI_hist_board_7 = np.zeros([1, 19, 19])
    AI_hist_board_8 = np.zeros([1, 19, 19])

    if np.all(AI_hist_1 == [-5, -5]):
        pass
    elif np.all(AI_hist_1 == [-1, -1]):
        pass
    else:
        AI_hist_board_1[0, AI_hist_1[0] - 1, AI_hist_1[1] - 1] = 1
    if np.all(AI_hist_2 == [-5, -5]):
        pass
    elif np.all(AI_hist_2 == [-1, -1]):
        pass
    else:
        AI_hist_board_2[0, AI_hist_2[0] - 1, AI_hist_2[1] - 1] = 1
    if np.all(AI_hist_3 == [-5, -5]):
        pass
    elif np.all(AI_hist_3 == [-1, -1]):
        pass
    else:
        AI_hist_board_3[0, AI_hist_3[0] - 1, AI_hist_3[1] - 1] = 1
    if np.all(AI_hist_4 == [-5, -5]):
        pass
    elif np.all(AI_hist_4 == [-1, -1]):
        pass
    else:
        AI_hist_board_4[0, AI_hist_4[0] - 1, AI_hist_4[1] - 1] = 1
    if np.all(AI_hist_5 == [-5, -5]):
        pass
    elif np.all(AI_hist_5 == [-1, -1]):
        pass
    else:
        AI_hist_board_5[0, AI_hist_5[0] - 1, AI_hist_5[1] - 1] = 1
    if np.all(AI_hist_6 == [-5, -5]):
        pass
    elif np.all(AI_hist_6 == [-1, -1]):
        pass
    else:
        AI_hist_board_6[0, AI_hist_6[0] - 1, AI_hist_6[1] - 1] = 1
    if np.all(AI_hist_7 == [-5, -5]):
        pass
    elif np.all(AI_hist_7 == [-1, -1]):
        pass
    else:
        AI_hist_board_7[0, AI_hist_7[0] - 1, AI_hist_7[1] - 1] = 1
    if np.all(AI_hist_8 == [-5, -5]):
        pass
    elif np.all(AI_hist_8 == [-1, -1]):
        pass
    else:
        AI_hist_board_8[0, AI_hist_8[0] - 1, AI_hist_8[1] - 1] = 1

    AI_inX = board_to_in(AI_board, AI_turn, AI_libs, AI_hist_board_1, AI_hist_board_2,
                              AI_hist_board_3,
                              AI_hist_board_4, AI_hist_board_5, AI_hist_board_6,
                              AI_hist_board_7, AI_hist_board_8, AI_terr) 
    if retinX == False: 
        return AI_inX 
    else: 
        return AI_inX, AI_board[0], AI_hist_board_1[0], AI_libs[0]


if __name__ == "__main__":
    ClickBoard()
    pass 