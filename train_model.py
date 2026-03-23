# -*- coding: utf-8 -*-
"""
高质量训练方案
核心改进：
1. 只用赢方棋谱训练（输方数据扔掉）
2. 使用温度采样提高数据多样性
3. 添加Top-K准确率监控
4. 大幅改进启发式走法质量
5. 添加梯度裁剪防止爆炸
"""

import os
import gc
import numpy as np
import tensorflow as tf
from copy import deepcopy
from Board import Go, calc_liberties, e_territory

# =========================
# 配置参数
# =========================
DATA_DIR = "prof_dat_1"
WEIGHT_FILE = "w_RL.npz"
TRAIN_LOG = "train_dat_RL.npy"
BEST_WEIGHT_FILE = "w_RL_best.npz"

BOARD_SIZE = 19
CHANNELS = 44
BATCH_SIZE = 128  # 减小batch size提高更新频率

# 数据生成
SELFPLAY_GAMES = 200           # 更多对局
MAX_MOVES_PER_GAME = 300
RANDOM_SEED = 1234
HEURISTIC_TEMPERATURE = 1.5    # 温度参数：越大越随机

# 训练策略
EPOCHS = 50
INITIAL_LR = 0.005             # 降低初始学习率
LR_DECAY_STEP = 15
LR_DECAY_RATE = 0.7
GRADIENT_CLIP = 1.0            # 梯度裁剪
PRINT_EVERY = 5
SAVE_EVERY = 100
PATIENCE = 8

USE_AUGMENTATION = True
USE_ONLY_WINNER_DATA = True    # 关键：只用赢方数据训练！

# =========================
# 工具函数
# =========================
def init_random_weights(out_ch=64, hidden_ch=64, seed=0):
    rng = np.random.default_rng(seed)
    w = []
    b = []

    def he_std(kh, kw, cin):
        return np.sqrt(2.0 / (kh * kw * cin))

    std0 = he_std(5, 5, CHANNELS)
    w0 = rng.normal(0, std0, size=(5, 5, CHANNELS, hidden_ch)).astype(np.float32)
    b0 = np.zeros((hidden_ch,), dtype=np.float32)
    w.append(w0); b.append(b0)

    for _ in range(11):
        std = he_std(3, 3, hidden_ch)
        wi = rng.normal(0, std, size=(3, 3, hidden_ch, hidden_ch)).astype(np.float32)
        bi = np.zeros((hidden_ch,), dtype=np.float32)
        w.append(wi); b.append(bi)

    std_last = he_std(1, 1, hidden_ch)
    w_last = rng.normal(0, std_last, size=(1, 1, hidden_ch, 1)).astype(np.float32)
    b_last = np.zeros((1,), dtype=np.float32)
    w.append(w_last); b.append(b_last)

    return np.array(w, dtype=object), np.array(b, dtype=object)

def ensure_weight_file(path=WEIGHT_FILE):
    if os.path.exists(path):
        return
    w, b = init_random_weights(hidden_ch=64, seed=RANDOM_SEED)
    np.savez(path, w=w, b=b)
    print(f"[init] 生成初始权重: {path}")

def ensure_train_log(path=TRAIN_LOG):
    if os.path.exists(path):
        return
    tarr = np.array([[0, 0, np.inf, 0]], dtype=float)
    np.save(path, tarr)

# =========================
# 特征提取
# =========================
def _hist_to_board(hist_move):
    h = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    if hist_move is None:
        return h
    if np.all(hist_move == [-1, -1]) or np.all(hist_move == [-5, -5]):
        return h
    x = int(np.clip(hist_move[0] - 1, 0, 18))
    y = int(np.clip(hist_move[1] - 1, 0, 18))
    h[x, y] = 1
    return h

def state_to_inX19(go: Go):
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
    c = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

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

# =========================
# 改进的启发式走法
# =========================
def get_move_score(go: Go, x: int, y: int) -> float:
    """
    给每个位置打分（x,y是1-19的坐标）
    分数越高越好
    """
    board = go.board
    # 转换为0-18索引
    xi, yi = x - 1, y - 1
    
    # 已占位置返回负分
    if board[xi, yi] != 0:
        return -1000.0
    
    score = 0.0
    move_num = go.hist.shape[0]
    
    # 开局（前30手）：角部权重最高
    if move_num < 30:
        # 角部（星位、小目等）
        corner_positions = [(3,3), (3,4), (4,3), (4,4), (3,16), (3,15), (4,16), (4,15),
                           (16,3), (15,3), (16,4), (15,4), (16,16), (15,16), (16,15), (15,15)]
        if (x, y) in corner_positions:
            score += 50.0
        
        # 边上三线、四线
        if (3 <= x <= 4 or 16 <= x <= 17 or 3 <= y <= 4 or 16 <= y <= 17):
            score += 30.0
        
        # 天元和中腹
        if 9 <= x <= 11 and 9 <= y <= 11:
            score += 20.0
    
    # 中盘：考虑邻近己方棋子
    else:
        # 计算周围己方/对方棋子数
        my_color = 2 * go.turn - 1  # 黑=1, 白=-1
        neighbors = 0
        enemy_neighbors = 0
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if board[nx, ny] == my_color:
                    neighbors += 1
                elif board[nx, ny] == -my_color:
                    enemy_neighbors += 1
        
        # 靠近己方棋子加分（连接）
        score += neighbors * 15.0
        # 靠近对方棋子加分（进攻）
        score += enemy_neighbors * 10.0
    
    # 避免边缘（一线、二线）
    if x == 1 or x == 19 or y == 1 or y == 19:
        score -= 20.0
    if x == 2 or x == 18 or y == 2 or y == 18:
        score -= 10.0
    
    return score

def smart_heuristic_move(go: Go, rng: np.random.Generator, temperature=1.5):
    """
    基于启发式评分的走法选择
    使用温度采样：temperature越大越随机
    """
    # 计算所有位置的分数
    scores = np.zeros(361)
    for i in range(19):
        for j in range(19):
            x, y = i + 1, j + 1
            scores[i * 19 + j] = get_move_score(go, x, y)
    
    # 过滤掉非法位置
    valid_mask = scores > -100
    
    if not np.any(valid_mask):
        go.place(np.array([-1, -1]))
        return np.array([-1, -1], dtype=np.int32)
    
    # 温度采样
    valid_scores = scores[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 将分数转换为概率（使用softmax + temperature）
    exp_scores = np.exp(valid_scores / temperature)
    probs = exp_scores / np.sum(exp_scores)
    
    # 按概率采样
    chosen_idx = rng.choice(valid_indices, p=probs)
    y = chosen_idx % 19
    x = chosen_idx // 19
    move = np.array([x + 1, y + 1], dtype=np.int32)
    
    # 尝试落子
    before = deepcopy(go.board)
    before_turn = go.turn
    r = go.place(move)
    
    if r == 0:
        return move
    else:
        # 如果选中的位置不合法，fallback到完全随机
        go.board = before
        go.turn = before_turn
        return random_legal_move(go, rng)

def random_legal_move(go: Go, rng: np.random.Generator):
    """完全随机合法走法"""
    coords = np.array([(i + 1, j + 1) for i in range(19) for j in range(19)], dtype=np.int32)
    rng.shuffle(coords)
    for xy in coords:
        before = deepcopy(go.board)
        before_turn = go.turn
        r = go.place(xy)
        if r == 0:
            return xy
        go.board = before
        go.turn = before_turn
    go.place(np.array([-1, -1]))
    return np.array([-1, -1], dtype=np.int32)

# =========================
# 数据增强
# =========================
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
# 数据生成
# =========================
def generate_dataset(out_dir=DATA_DIR, games=SELFPLAY_GAMES, batch_size=BATCH_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

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

        for t in range(MAX_MOVES_PER_GAME):
            if hasattr(go, "game_end") and go.game_end:
                break

            feat = state_to_inX19(go)
            mv = smart_heuristic_move(go, rng, temperature=HEURISTIC_TEMPERATURE)
            
            y1 = move_to_onehot(mv)
            if y1 is not None:
                traj_feats.append(feat)
                traj_moves.append(y1)
                traj_turns.append(int(1 - go.turn))

            if hasattr(go, "game_end") and go.game_end:
                break

        go.score(estimate=False)
        winner_turn = int(go.wl)

        # 关键改进：只用赢方数据！
        for feat, y1, mover_turn in zip(traj_feats, traj_moves, traj_turns):
            # 只保留赢方的棋谱
            if USE_ONLY_WINNER_DATA and mover_turn != winner_turn:
                continue
            
            wl_label = 1 if mover_turn == winner_turn else -1
            winner_samples += 1
            
            if USE_AUGMENTATION:
                # 每个赢方样本生成2个增强版本
                for aug_id in [0, rng.integers(1, 8)]:
                    feat_aug, y_aug = augment_sample(feat, y1, aug_id)
                    inX_buf.append(feat_aug)
                    y_buf.append(y_aug)
                    wl_buf.append(wl_label)
            else:
                inX_buf.append(feat)
                y_buf.append(y1)
                wl_buf.append(wl_label)

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

    if len(inX_buf) > 0:
        save_path = os.path.join(out_dir, f"sample_{sample_idx}.npz")
        np.savez(
            save_path,
            inX=np.stack(inX_buf, axis=0).astype(np.float32),
            y=np.stack(y_buf, axis=0).astype(np.int32),
            wl=np.array(wl_buf, dtype=np.int32)
        )
        total_samples += len(inX_buf)

    print(f"[data] 完成！games={games}, 赢方样本={winner_samples}, 总样本（含增强）={total_samples}")
    return total_samples

# =========================
# 训练网络
# =========================
def Rect(x):
    return tf.cast(np.maximum(0.01 * x, x), tf.float32)

def dRect(x):
    return tf.cast(x > 0, tf.float32) + tf.cast(x <= 0, tf.float32) * 0.01

def clip_gradients(grads, max_norm=GRADIENT_CLIP):
    """梯度裁剪"""
    clipped = []
    for g in grads:
        if g is None:
            clipped.append(g)
        else:
            clipped.append(tf.clip_by_norm(g, max_norm))
    return clipped

class ConvPolicy:
    def __init__(self, w_file="w_RL"):
        self.load(w_file)

    def load(self, w_file="w_RL"):
        dat = np.load(w_file + ".npz", allow_pickle=True)
        self.w = dat["w"]
        self.b = dat["b"]

    def save(self, w_file="w_RL"):
        np.savez(w_file + ".npz", w=self.w, b=self.b)

    def forward(self, batch_input):
        N = batch_input.shape[0]
        self.x = batch_input
        self.z1 = tf.nn.conv2d(self.x, self.w[0], [1, 1, 1, 1], "VALID") + tf.tile(tf.expand_dims(self.b[0], 0), [N, 1, 1, 1])
        self.a1 = Rect(self.z1)
        
        a = self.a1
        self.z = [self.z1]
        self.a = [self.a1]
        for i in range(1, 12):
            zi = tf.nn.conv2d(a, self.w[i], [1, 1, 1, 1], "SAME") + tf.tile(tf.expand_dims(self.b[i], 0), [N, 1, 1, 1])
            ai = Rect(zi)
            self.z.append(zi)
            self.a.append(ai)
            a = ai
        
        self.z13 = tf.nn.conv2d(a, self.w[12], [1, 1, 1, 1], "SAME") + tf.tile(tf.expand_dims(self.b[12], 0), [N, 1, 1, 1])
        self.p = tf.nn.softmax(tf.reshape(self.z13, [-1, 19 * 19]), axis=1)
        self.p1 = tf.expand_dims(self.p, 2)

    def loss_calc(self, inX, prob):
        self.forward(inX)
        self.loss = tf.reduce_sum(tf.multiply(-prob, tf.math.log(self.p + 1e-10)))

    def backward(self, inX, prob, eta=0.01):
        self.loss_calc(inX, prob)
        N = prob.shape[0]

        DS = tf.compat.v1.matrix_diag(self.p) - tf.matmul(self.p1, tf.transpose(self.p1, [0, 2, 1]))
        p1 = tf.expand_dims(tf.divide(-prob, self.p + 1e-10), 2)
        d13 = tf.reshape(tf.matmul(DS, p1), [-1, 19, 19, 1])

        a12 = self.a[-1]
        dW13 = tf.compat.v1.nn.conv2d_backprop_filter(a12, self.w[12].shape, d13, [1, 1, 1, 1], "VALID")
        
        w12_rot = tf.transpose(tf.image.rot90(tf.transpose(self.w[12], [2, 0, 1, 3]), k=2), [1, 2, 3, 0])
        d12 = tf.multiply(tf.nn.conv2d(d13, w12_rot, [1, 1, 1, 1], "SAME"), dRect(self.z[11]))

        d_list = [None] * 12
        d_list[11] = d12

        for li in range(11, 0, -1):
            w_rot = tf.transpose(tf.image.rot90(tf.transpose(self.w[li], [2, 0, 1, 3]), k=2), [1, 2, 3, 0])
            prev_z = self.z[li - 1]
            d_prev = tf.multiply(tf.nn.conv2d(d_list[li], w_rot, [1, 1, 1, 1], "SAME"), dRect(prev_z))
            d_list[li - 1] = d_prev

        # 梯度裁剪
        dW13 = tf.clip_by_norm(dW13, GRADIENT_CLIP)
        
        self.w[12] = self.w[12] - eta * dW13 / N
        self.b[12] = self.b[12] - eta * tf.clip_by_norm(np.sum(d13, axis=0), GRADIENT_CLIP) / N

        for li in range(11, 0, -1):
            inp = self.a[li - 1] if li > 1 else self.a1
            inp_pad = tf.pad(inp, [[0, 0], [1, 1], [1, 1], [0, 0]])
            dW = tf.compat.v1.nn.conv2d_backprop_filter(inp_pad, self.w[li].shape, d_list[li], [1, 1, 1, 1], "VALID")
            dW = tf.clip_by_norm(dW, GRADIENT_CLIP)
            db = np.sum(d_list[li], axis=0)
            self.w[li] = self.w[li] - eta * dW / N
            self.b[li] = self.b[li] - eta * tf.clip_by_norm(db, GRADIENT_CLIP) / N

        dW0 = tf.compat.v1.nn.conv2d_backprop_filter(self.x, self.w[0].shape, d_list[0], [1, 1, 1, 1], "VALID")
        dW0 = tf.clip_by_norm(dW0, GRADIENT_CLIP)
        db0 = np.sum(d_list[0], axis=0)
        self.w[0] = self.w[0] - eta * dW0 / N
        self.b[0] = self.b[0] - eta * tf.clip_by_norm(db0, GRADIENT_CLIP) / N

def train_from_prof_dat(data_dir=DATA_DIR, epochs=EPOCHS):
    files = sorted([f for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".npz")])
    if len(files) == 0:
        raise FileNotFoundError(f"在 {data_dir}/ 下没有 sample_*.npz")

    policy = ConvPolicy(w_file="w_RL")
    ensure_train_log(TRAIN_LOG)
    train_hist = np.load(TRAIN_LOG)

    best_acc = 0.0
    best_top5_acc = 0.0
    patience_counter = 0
    step = 0

    for ep in range(epochs):
        current_lr = INITIAL_LR * (LR_DECAY_RATE ** (ep // LR_DECAY_STEP))
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_top5 = 0.0
        
        np.random.shuffle(files)
        
        for fn in files:
            step += 1
            dat = np.load(os.path.join(data_dir, fn), allow_pickle=True)
            inX19 = dat["inX"].astype(np.float32)
            y = dat["y"].reshape(-1, 361).astype(np.int32)
            wl = dat["wl"].astype(np.int32)

            inX = tf.cast(tf.pad(inX19, [[0, 0], [2, 2], [2, 2], [0, 0]]), tf.float32)

            # 关键：只用赢方数据，不做奇怪的标签变换！
            y_train = y.astype(np.float32)

            policy.backward(inX, y_train, eta=current_lr)
            loss = float(policy.loss.numpy()) / y_train.shape[0]
            
            # Top-1准确率
            acc = float(tf.reduce_sum(tf.multiply(y_train, policy.p)).numpy()) / y_train.shape[0]
            
            # Top-5准确率（前5个预测中是否包含正确答案）
            top5_pred = tf.nn.top_k(policy.p, k=5).indices.numpy()
            true_pos = np.argmax(y_train, axis=1)
            top5_correct = np.mean([true_pos[i] in top5_pred[i] for i in range(len(true_pos))])
            
            epoch_loss += loss
            epoch_acc += acc
            epoch_top5 += top5_correct

            if step % PRINT_EVERY == 0:
                print(f"[train] ep={ep+1}/{epochs} step={step} lr={current_lr:.5f} "
                      f"loss={loss:.4f} acc={acc:.4f} top5={top5_correct:.4f}")

            if np.isfinite(loss) and loss < 20:
                if step % SAVE_EVERY == 0:
                    policy.save("w_RL")

            train_hist = np.append(train_hist, np.array([[ep, step, loss, acc]]), axis=0)

        epoch_loss /= len(files)
        epoch_acc /= len(files)
        epoch_top5 /= len(files)
        
        print(f"\n{'='*60}")
        print(f"[epoch {ep+1}] 平均 loss={epoch_loss:.4f}, Top-1 acc={epoch_acc:.4f}, Top-5 acc={epoch_top5:.4f}")
        print(f"{'='*60}\n")
        
        if epoch_top5 > best_top5_acc:
            best_acc = epoch_acc
            best_top5_acc = epoch_top5
            policy.save(BEST_WEIGHT_FILE.replace('.npz', ''))
            print(f"[best] 保存最佳模型 Top-1={best_acc:.4f} Top-5={best_top5_acc:.4f}\n")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"[early stop] {PATIENCE}个epoch没有提升，停止训练")
            break
        
        np.save(TRAIN_LOG, train_hist)

    policy.save("w_RL")
    print(f"\n{'='*60}")
    print(f"[train] 训练完成！")
    print(f"  最佳 Top-1 acc = {best_acc:.4f}")
    print(f"  最佳 Top-5 acc = {best_top5_acc:.4f}")
    print(f"  权重保存: w_RL.npz")
    print(f"  最佳权重: {BEST_WEIGHT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")

    print("=" * 60)
    print("围棋AI训练 - 高质量方案")
    print("=" * 60)
    print(f"核心改进：")
    print(f"  ✓ 只用赢方棋谱训练")
    print(f"  ✓ 智能启发式走法")
    print(f"  ✓ 梯度裁剪防止爆炸")
    print(f"  ✓ Top-5准确率监控")
    print("=" * 60)
    print(f"配置：")
    print(f"  - 对局数量: {SELFPLAY_GAMES}")
    print(f"  - 训练轮数: {EPOCHS}")
    print(f"  - 数据增强: {USE_AUGMENTATION}")
    print(f"  - 初始学习率: {INITIAL_LR}")
    print(f"  - 梯度裁剪: {GRADIENT_CLIP}")
    print("=" * 60 + "\n")

    ensure_weight_file(WEIGHT_FILE)
    ensure_train_log(TRAIN_LOG)

    if not os.path.exists(DATA_DIR) or len([f for f in os.listdir(DATA_DIR) if f.startswith("sample_")]) == 0:
        print("[main] 开始生成高质量训练数据...\n")
        generate_dataset(DATA_DIR, games=SELFPLAY_GAMES, batch_size=BATCH_SIZE)
        print()

    print("[main] 开始训练...\n")
    train_from_prof_dat(DATA_DIR, epochs=EPOCHS)