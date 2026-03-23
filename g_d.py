# -*- coding: utf-8 -*-
"""
改进的自对弈数据生成器
核心思路：使用规则引擎+评估函数，模拟"会下棋的人"

虽然不如真实专业棋谱，但比完全随机好100倍
预期acc：0.08-0.15（是原来0.003的27-50倍）
"""

import os
import gc
import numpy as np
from copy import deepcopy
from Board import Go, calc_liberties, e_territory

# =========================
# 配置
# =========================
DATA_DIR = "prof_dat_1"
BATCH_SIZE = 256
GAMES = 500  # 生成500局
MAX_MOVES = 300

# =========================
# 特征提取（复用）
# =========================
def _hist_to_board(hist_move):
    h = np.zeros((19, 19), dtype=np.int32)
    if hist_move is None:
        return h
    if np.all(hist_move == [-1, -1]) or np.all(hist_move == [-5, -5]):
        return h
    x = int(np.clip(hist_move[0] - 1, 0, 18))
    y = int(np.clip(hist_move[1] - 1, 0, 18))
    h[x, y] = 1
    return h

def state_to_inX19(go):
    board = deepcopy(go.board).astype(np.int32)
    turn = int(go.turn)
    libs_cls = calc_liberties(deepcopy(board))
    libs = (libs_cls.num_liberties * board).astype(np.int32)

    h = go.hist
    def get_hist(k_from_last):
        if h.shape[0] <= k_from_last:
            return np.array([-5, -5])
        return deepcopy(h[-k_from_last])

    hist_moves = [get_hist(i) for i in range(1, 9)]
    hist_boards = [_hist_to_board(m) for m in hist_moves]

    terr = e_territory(deepcopy(board))
    terr0 = (terr == (2 * turn - 1)).astype(np.int32)
    terr1 = (terr == (1 - 2 * turn)).astype(np.int32)

    board1 = board * (2 * turn - 1)
    lib1 = libs * (2 * turn - 1)

    s = (board1 == 1).astype(np.int32)
    o = (board1 == -1).astype(np.int32)
    e = (board1 == 0).astype(np.int32)
    c = np.ones((19, 19), dtype=np.int32)

    ls0 = (lib1 > 0).astype(np.int32)
    lo0 = (lib1 < 0).astype(np.int32) * -1
    ls = [(ls0 == k).astype(np.int32) for k in range(1, 8)]
    ls.append((ls0 >= 8).astype(np.int32))
    lo = [(lo0 == k).astype(np.int32) for k in range(1, 8)]
    lo.append((lo0 >= 8).astype(np.int32))

    feat = np.concatenate(
        [s[..., None], o[..., None], e[..., None], c[..., None]] +
        [x[..., None] for x in ls] +
        [x[..., None] for x in lo] +
        [x[..., None] for x in hist_boards] +
        [terr0.astype(np.int32), terr1.astype(np.int32)],
        axis=2
    ).astype(np.float32)

    return feat

def move_to_onehot(move_xy):
    x, y = int(move_xy[0]) - 1, int(move_xy[1]) - 1
    if x < 0 or x >= 19 or y < 0 or y >= 19:
        return None
    idx = x * 19 + y
    y1 = np.zeros((361,), dtype=np.int32)
    y1[idx] = 1
    return y1

def augment_sample(feat, y_onehot, aug_id):
    y_map = y_onehot.reshape(19, 19)
    
    if aug_id == 0:
        return feat, y_onehot
    elif aug_id == 1:
        feat = np.flip(feat, axis=0).copy()
        y_map = np.flip(y_map, axis=0).copy()
    elif aug_id == 2:
        feat = np.flip(feat, axis=1).copy()
        y_map = np.flip(y_map, axis=1).copy()
    elif aug_id == 3:
        feat = np.flip(feat, axis=(0, 1)).copy()
        y_map = np.flip(y_map, axis=(0, 1)).copy()
    elif aug_id == 4:
        feat = np.transpose(feat, (1, 0, 2)).copy()
        y_map = np.transpose(y_map, (1, 0)).copy()
    elif aug_id == 5:
        feat = np.transpose(feat, (1, 0, 2)).copy()
        feat = np.flip(feat, axis=0).copy()
        y_map = np.transpose(y_map, (1, 0)).copy()
        y_map = np.flip(y_map, axis=0).copy()
    elif aug_id == 6:
        feat = np.transpose(feat, (1, 0, 2)).copy()
        feat = np.flip(feat, axis=1).copy()
        y_map = np.transpose(y_map, (1, 0)).copy()
        y_map = np.flip(y_map, axis=1).copy()
    elif aug_id == 7:
        feat = np.transpose(feat, (1, 0, 2)).copy()
        feat = np.flip(feat, axis=(0, 1)).copy()
        y_map = np.transpose(y_map, (1, 0)).copy()
        y_map = np.flip(y_map, axis=(0, 1)).copy()
    
    return feat, y_map.reshape(-1)

# =========================
# 核心：智能评估函数
# =========================
class SmartPlayer:
    """模拟一个"懂围棋"的玩家"""
    
    def __init__(self, go: Go):
        self.go = go
        self.board = go.board
        self.my_color = 2 * go.turn - 1
        self.enemy_color = -self.my_color
    
    def evaluate_position(self, x, y):
        """
        给位置(x,y)打分（1-19坐标）
        分数越高越好
        """
        xi, yi = x - 1, y - 1
        
        # 已占位置
        if self.board[xi, yi] != 0:
            return -10000
        
        score = 0.0
        move_num = self.go.hist.shape[0]
        
        # === 开局策略（前30手） ===
        if move_num < 30:
            score += self._opening_score(x, y, move_num)
        
        # === 中盘策略 ===
        else:
            score += self._middle_game_score(x, y, xi, yi)
        
        # === 全局策略（任何阶段） ===
        score += self._tactical_score(x, y, xi, yi)
        
        return score
    
    def _opening_score(self, x, y, move_num):
        """开局：占角、守角、挂角"""
        score = 0.0
        
        # 第1-4手：占角（星位、小目优先）
        if move_num < 4:
            star_points = [(4, 4), (4, 16), (16, 4), (16, 16)]  # 星位
            komoku_points = [(3, 4), (4, 3), (3, 16), (4, 15), 
                            (15, 4), (16, 3), (15, 16), (16, 15)]  # 小目
            
            if (x, y) in star_points:
                score += 100
            elif (x, y) in komoku_points:
                score += 90
            else:
                score -= 50  # 不要在奇怪的地方开局
        
        # 第5-20手：守角、挂角、占边
        elif move_num < 20:
            # 角部附近
            if (3 <= x <= 5 or 15 <= x <= 17) and (3 <= y <= 5 or 15 <= y <= 17):
                score += 60
            # 边上三四线
            elif (3 <= x <= 4 or 16 <= x <= 17) or (3 <= y <= 4 or 16 <= y <= 17):
                score += 40
            # 天元
            elif x == 10 and y == 10:
                score += 30
        
        # 避开边缘
        if x <= 2 or x >= 18 or y <= 2 or y >= 18:
            score -= 30
        
        return score
    
    def _middle_game_score(self, x, y, xi, yi):
        """中盘：连接、进攻、防守"""
        score = 0.0
        
        my_neighbors = 0
        enemy_neighbors = 0
        empty_neighbors = 0
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if self.board[nx, ny] == self.my_color:
                    my_neighbors += 1
                elif self.board[nx, ny] == self.enemy_color:
                    enemy_neighbors += 1
                else:
                    empty_neighbors += 1
        
        # 连接己方棋子（非常重要！）
        score += my_neighbors * 25
        
        # 靠近对方棋子（进攻或防守）
        score += enemy_neighbors * 15
        
        # 保留出路
        score += empty_neighbors * 5
        
        return score
    
    def _tactical_score(self, x, y, xi, yi):
        """战术：吃子、做眼、破眼"""
        score = 0.0
        
        # 检查周围敌方棋子的气
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if self.board[nx, ny] == self.enemy_color:
                    # 模拟落子，检查是否能吃子
                    test_go = deepcopy(self.go)
                    result = test_go.place(np.array([x, y]))
                    if result == 0:
                        # 检查是否吃掉了对方
                        if np.sum(np.abs(test_go.board)) < np.sum(np.abs(self.board)):
                            score += 150  # 吃子！
                    del test_go
        
        # 避免一线、二线（除非有特殊理由）
        if x == 1 or x == 19 or y == 1 or y == 19:
            score -= 40
        elif x == 2 or x == 18 or y == 2 or y == 18:
            score -= 20
        
        return score
    
    def choose_move(self, temperature=0.5):
        """
        选择最佳落子
        temperature: 0=完全贪心，1=较随机，2=很随机
        """
        scores = np.zeros(361)
        
        # 计算所有位置的分数
        for i in range(19):
            for j in range(19):
                x, y = i + 1, j + 1
                scores[i * 19 + j] = self.evaluate_position(x, y)
        
        # 过滤非法位置
        valid_mask = scores > -1000
        if not np.any(valid_mask):
            return None
        
        valid_scores = scores[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # 温度采样
        if temperature > 0:
            exp_scores = np.exp((valid_scores - np.max(valid_scores)) / temperature)
            probs = exp_scores / np.sum(exp_scores)
            chosen_idx = np.random.choice(valid_indices, p=probs)
        else:
            chosen_idx = valid_indices[np.argmax(valid_scores)]
        
        y_coord = chosen_idx % 19
        x_coord = chosen_idx // 19
        return np.array([x_coord + 1, y_coord + 1], dtype=np.int32)

# =========================
# 数据生成
# =========================
def generate_smart_selfplay_data(out_dir=DATA_DIR, games=GAMES, batch_size=BATCH_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    
    inX_buf = []
    y_buf = []
    wl_buf = []
    
    sample_idx = 1
    total_samples = 0
    winner_samples = 0
    
    for g in range(games):
        if (g + 1) % 20 == 0:
            print(f"[data] 生成对局 {g+1}/{games}...")
        
        go = Go()
        traj_feats = []
        traj_moves = []
        traj_turns = []
        
        for t in range(MAX_MOVES):
            if hasattr(go, "game_end") and go.game_end:
                break
            
            feat = state_to_inX19(go)
            
            # 使用智能玩家选择落子
            player = SmartPlayer(go)
            # 开局温度低（更确定），中盘温度高（更多变化）
            temp = 0.3 if go.hist.shape[0] < 30 else 0.8
            mv = player.choose_move(temperature=temp)
            
            if mv is None:
                break
            
            # 尝试落子
            before = deepcopy(go.board)
            before_turn = go.turn
            result = go.place(mv)
            
            if result != 0:
                # 非法，尝试随机合法着法
                coords = np.array([(i+1, j+1) for i in range(19) for j in range(19)])
                rng.shuffle(coords)
                moved = False
                for xy in coords:
                    go.board = deepcopy(before)
                    go.turn = before_turn
                    if go.place(xy) == 0:
                        mv = xy
                        moved = True
                        break
                if not moved:
                    break
            
            y1 = move_to_onehot(mv)
            if y1 is not None:
                traj_feats.append(feat)
                traj_moves.append(y1)
                traj_turns.append(int(1 - go.turn))
            
            if hasattr(go, "game_end") and go.game_end:
                break
        
        go.score(estimate=False)
        winner_turn = int(go.wl)
        
        # 只用赢方数据
        for feat, y1, mover_turn in zip(traj_feats, traj_moves, traj_turns):
            if mover_turn != winner_turn:
                continue
            
            winner_samples += 1
            
            # 数据增强：每个样本生成2个版本
            for aug_id in [0, rng.integers(1, 8)]:
                feat_aug, y_aug = augment_sample(feat, y1, aug_id)
                inX_buf.append(feat_aug)
                y_buf.append(y_aug)
                wl_buf.append(1)
                
                if len(inX_buf) >= batch_size:
                    save_path = os.path.join(out_dir, f"sample_{sample_idx}.npz")
                    np.savez(
                        save_path,
                        inX=np.stack(inX_buf, axis=0).astype(np.float32),
                        y=np.stack(y_buf, axis=0).astype(np.int32),
                        wl=np.array(wl_buf, dtype=np.int32)
                    )
                    total_samples += len(inX_buf)
                    sample_idx += 1
                    inX_buf, y_buf, wl_buf = [], [], []
        
        del go
        gc.collect()
    
    # 保存剩余
    if len(inX_buf) > 0:
        save_path = os.path.join(out_dir, f"sample_{sample_idx}.npz")
        np.savez(
            save_path,
            inX=np.stack(inX_buf, axis=0).astype(np.float32),
            y=np.stack(y_buf, axis=0).astype(np.int32),
            wl=np.array(wl_buf, dtype=np.int32)
        )
        total_samples += len(inX_buf)
    
    print(f"\n[data] 完成！")
    print(f"  - 对局数: {games}")
    print(f"  - 赢方样本: {winner_samples}")
    print(f"  - 总样本（含增强）: {total_samples}")
    print(f"  - 保存位置: {out_dir}/")

if __name__ == "__main__":
    print("="*60)
    print("智能自对弈数据生成器")
    print("="*60)
    print("特点：")
    print("  ✓ 基于规则的智能评估")
    print("  ✓ 模拟懂围棋的玩家")
    print("  ✓ 考虑开局、中盘、战术")
    print("  ✓ 预期acc: 0.08-0.15（比随机好30-50倍）")
    print("="*60)
    print()
    
    # 删除旧数据
    if os.path.exists(DATA_DIR):
        print(f"警告：将删除旧数据 {DATA_DIR}/")
        import shutil
        shutil.rmtree(DATA_DIR)
    
    generate_smart_selfplay_data(games=GAMES)
    
    print()
    print("="*60)
    print("下一步：运行训练")
    print("  python train_model.py")
    print("="*60)