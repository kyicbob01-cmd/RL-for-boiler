"""
工業級 SCADA 人機介面 - 專業版
風格：Siemens/Rockwell 風格
特點：
1. 嚴格的工業灰階配色
2. 極簡化圖形設計
3. 高對比度文字
4. 狀態顏色僅用於指示燈與關鍵數據
"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'sans-serif']
import numpy as np
import copy
import time

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ==========================================
# 專業工業配色 (Siemens/Rockwell 風格)
# ==========================================
C = {
    # 介面基礎
    'bg_app': '#d4d4d4',      # 應用程式背景 (工業灰)
    'bg_panel': '#e6e6e6',    # 面板背景
    'bg_content': '#ffffff',  # 內容區域背景
    
    # 邊框與分隔
    'border_light': '#ffffff',
    'border_dark': '#808080',
    'border_frame': '#999999',
    
    # 文字
    'text_main': '#000000',
    'text_dim': '#555555',
    'text_light': '#ffffff',
    
    # 狀態指示 (標準工業色)
    'status_run': '#00a000',    # 運行 (深綠)
    'status_stop': '#cccccc',   # 停止 (灰)
    'status_alarm': '#ff0000',  # 警報 (紅)
    'status_warn': '#ffb000',   # 警告 (橘黃)
    'status_active': '#0066cc', # 激活 (藍)
    
    # 圖形物件
    'pipe': '#666666',
    'pipe_flow': '#0066cc',
    'tank_fill': '#e0e0e0',
    'tank_outline': '#333333',
    
    # 比較色
    'human_color': '#b00000',   # 深紅
    'ai_color': '#000080',      # 深藍
}

# 字體配置
F = {
    'h1': ("Microsoft JhengHei", 16, "bold"),
    'h2': ("Microsoft JhengHei", 14, "bold"),
    'body': ("Microsoft JhengHei", 12),
    'num': ("Segoe UI", 12, "bold"),
    'num_big': ("Segoe UI", 24, "bold"),
    'tag': ("Segoe UI", 10),
}

# ==========================================
# 系統邏輯 (保持不變)
# ==========================================
場景 = {
    "場景 1：標準生產": [
        {'name': '反應槽 A', 'target': 150.0, 'duration': 90.0, 'weight': 300.0},
        {'name': '乾燥機 B', 'target': 90.0, 'duration': 500.0, 'weight': 100.0},
        {'name': '預熱器 C', 'target': 100.0, 'duration': 150.0, 'weight': 900.0}
    ],
    "場景 2：高溫作業": [
        {'name': '反應槽 A', 'target': 100.0, 'duration': 150.0, 'weight': 600.0},
        {'name': '反應槽 B', 'target': 150.0, 'duration': 300.0, 'weight': 1000.0}
    ]
}


import torch
import torch.nn as nn
import numpy as np
import os

class Actor(nn.Module):
    """Offline RL Actor Network"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x) * 100.0

class BCModel(nn.Module):
    """Behavior Cloning 神經網路"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x) * 100.0

class RCPolicy(nn.Module):
    """Return-Conditioned Policy Network"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, state, target_cost):
        cost_norm = target_cost / 50.0
        x = torch.cat([state, cost_norm], dim=-1)
        return self.net(x)

class SmartController:
    """SmartController - 優先 RCP > Offline RL > BC > 規則"""
    def __init__(self):
        self.model = None
        self.use_ai = False
        self.model_type = "rules"
        
        # 1. 嘗試載入 RCP (最新策略)
        if os.path.exists("rc_policy.pth"):
            try:
                self.model = RCPolicy()
                self.model.load_state_dict(torch.load("rc_policy.pth"))
                self.model.eval()
                self.use_ai = True
                self.model_type = "rcp"
                print(">>> RCP Agent Loaded (Target Cost=15) <<<")
            except Exception as e:
                print(f"RCP 載入失敗: {e}")

        # 2. 嘗試載入 Offline RL
        if not self.use_ai and os.path.exists("offline_rl_agent.pth"):
            try:
                self.model = Actor()
                self.model.load_state_dict(torch.load("offline_rl_agent.pth"))
                self.model.eval()
                self.use_ai = True
                self.model_type = "offline_rl"
                print(">>> Offline RL Agent Loaded <<<")
            except Exception as e:
                print(f"Offline RL 載入失敗: {e}")
        
        # 3. 嘗試載入 BC
        if not self.use_ai and os.path.exists("bc_agent.pth"):
            try:
                self.model = BCModel()
                self.model.load_state_dict(torch.load("bc_agent.pth"))
                self.model.eval()
                self.use_ai = True
                self.model_type = "bc"
                print(">>> BC Agent Loaded <<<")
            except Exception as e:
                print(f"BC 載入失敗: {e}")
        
        if not self.use_ai:
            print("[SmartController V3] 規則控制器已啟動")

    def decide(self, temp, units):
        """決策函數"""
        # === AI 模型推論 ===
        if self.use_ai and self.model:
            try:
                # 準備觀測狀態
                active_count = 0
                max_target = 0
                total_load = 0
                for u in units.values():
                    if u['state'] != '完成':
                        active_count += 1
                        if u['target'] > max_target: max_target = u['target']
                        gap = max(0, u['target'] - u['current'])
                        total_load += gap * u['weight'] * 1.2
                
                obs = [
                    temp / 300.0,
                    max_target / 300.0,
                    active_count / 4.0,
                    0.0,  # rate simplification
                    total_load / 2000000.0
                ]
                
                # 1. RCP 模型
                if hasattr(self, 'model_type') and self.model_type == 'rcp':
                    obs_t = torch.tensor([obs], dtype=torch.float32)
                    target_t = torch.tensor([[15.0]], dtype=torch.float32) # 目標設定為 15 TWD (非常高效)
                    with torch.no_grad():
                        power = self.model(obs_t, target_t).item() * 100.0
                    return power

                # 2. Offline RL (Actor)
                elif hasattr(self, 'model_type') and self.model_type == 'offline_rl':
                    obs_t = torch.tensor([obs], dtype=torch.float32)
                    with torch.no_grad():
                        power = self.model(obs_t).item() # output is 0-100 already
                    return power

                # 3. BC Model
                else: 
                    obs_t = torch.tensor([obs], dtype=torch.float32)
                    with torch.no_grad():
                        power = self.model(obs_t).item()
                    return power

            except Exception as e:
                print(f"AI Error: {e}")
        
        # === 規則控制 V3 (Fallback) ===
        active = [u for u in units.values() if u['state'] != '完成']
        if not active: 
            return 0.0  # 全部完成，關閉
        
        # 找出還在加熱中的單元 (尚未達標)
        needed_units = [u for u in active if u['current'] < u['target'] - 0.5]
        needed_targets = [u['target'] for u in needed_units]
        
        if not needed_targets:
            # 所有單元都達標，進入保溫模式
            holding = [u['target'] for u in active if u['state'] == '保溫']
            if holding and temp < min(holding) + 3.0: 
                return 30.0  # 維持微火保溫
            return 0.0
        
        # 計算目標鍋爐溫度
        max_target = max(needed_targets)
        current_max_demand = max([u['current'] for u in needed_units])
        
        # 確保足夠的驅動溫差
        min_driving_temp = current_max_demand + 6.0
        target_boiler = max(max_target + 5.0, min_driving_temp)
        
        gap = target_boiler - temp
        
        # 功率決策：階梯式控制
        if gap < -2.0:
            return 0.0   # 過熱，停止
        elif gap > 20:
            return 100.0  # 大差距，全力
        elif gap > 10:
            return 80.0
        elif gap > 5:
            return 50.0
        elif gap > 0:
            return 20.0
        else:
            return 0.0

class Engine:
    def __init__(self): self.reset()
    def reset(self):
        self.temp = 25.0; self.cost = 0.0; self.kw = 0.0; self.units = {}
        self.energy = {'heat': 0.0, 'hold': 0.0, 'loss': 0.0}
        self.history = [] # 記錄數據
        self.start_time = 0
    def load(self, tasks):
        self.reset()
        self.start_time = time.time()
        for i, t in enumerate(tasks):
            self.units[i] = {**t, 'current': 25.0, 'mass': t['weight']*1.2, 
                             'state': '加熱', 'valve': False, 'left': t['duration']}
    def done(self): return self.units and all(u['state'] == '完成' for u in self.units.values())
    def step(self, pwr):
        pwr = max(0, min(100, pwr)); self.kw = pwr
        kwh = pwr * (0.5/3600.0); self.cost += kwh * 4.5
        
        # 記錄狀態 (CSV 格式準備)
        # Time, BoilerTemp, Power, Cost, Unit1_Temp, Unit1_State...
        if self.start_time > 0:
            row = {
                'time': time.time() - self.start_time,
                'boiler_temp': self.temp,
                'power': pwr,
                'cost': self.cost,
            }
            # 簡化記錄單元數據 (只記前3個)
            for i, u in self.units.items():
                if i < 3:
                   row[f'u{i}_temp'] = u['current']
                   row[f'u{i}_state'] = 1 if u['state']=='加熱' else (2 if u['state']=='保溫' else 3)
            self.history.append(row)

        # 能耗統計
        if any(u['state']=='加熱' for u in self.units.values()): self.energy['heat']+=kwh*0.7; self.energy['loss']+=kwh*0.3
        elif any(u['state']=='保溫' for u in self.units.values()): self.energy['hold']+=kwh*0.5; self.energy['loss']+=kwh*0.5
        else: self.energy['loss']+=kwh

        total_out = 0.0
        for u in self.units.values():
            if u['state'] == '完成': continue
            if u['state'] == '加熱' and u['current'] >= u['target']-0.5: u['state'] = '保溫'
            elif u['state'] == '保溫':
                if u['current'] >= u['target']-3.0: u['left'] -= 0.5
                if u['left'] <= 0: u['state'] = '完成'
            
            u['valve'] = u['current'] < u['target']-0.5 and self.temp > u['current']
            if u['valve']:
                delta = max(0, self.temp - u['current'])
                trans = delta * 30.0; loss = (u['current']-25)*0.3
                u['current'] += (trans-loss)/u['mass']*0.5; total_out += trans
            else:
                u['current'] -= ((u['current']-25)*0.2)/u['mass']*0.5
        
        self.temp += (pwr*50*0.9 - (self.temp-25)*4.0 - total_out)/1500.0*0.5

# ==========================================
# 專業繪圖元件
# ==========================================
class IndustrialRender:
    @staticmethod
    def draw_bezel(c, x1, y1, x2, y2, bg=C['bg_content']):
        """繪製工業風格的凹陷邊框"""
        c.create_rectangle(x1, y1, x2, y2, fill=bg, outline=C['border_frame'])
        c.create_line(x1, y2, x2, y2, x2, y1, fill=C['border_light'])
        c.create_line(x1, y2, x1, y1, x2, y1, fill=C['border_dark'])

    @staticmethod
    def draw_tank(c, x, y, w, h, level, color=C['tank_fill']):
        """繪製簡約的工業儲槽"""
        # 槽體
        c.create_rectangle(x, y, x+w, y+h, fill=C['tank_fill'], outline=C['tank_outline'], width=2)
        # 液位
        if level > 0:
            lh = h * level
            c.create_rectangle(x+2, y+h-lh, x+w-2, y+h-1, fill=color, outline='')
        # 刻度線
        for i in range(1, 4):
            ly = y + h * (i/4)
            c.create_line(x, ly, x+10, ly, fill=C['text_dim'])
            c.create_line(x+w-10, ly, x+w, ly, fill=C['text_dim'])

    @staticmethod
    def draw_pump(c, x, y, r, active):
        """繪製泵浦符號"""
        color = C['status_run'] if active else C['bg_panel']
        c.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='black')
        c.create_line(x-r, y, x-r/2, y-r*0.8, x+r/2, y+r*0.8, x+r, y, width=2)

    @staticmethod
    def draw_valve(c, x, y, w, active):
        """繪製工業閥門符號"""
        h = w * 0.6
        fill = C['status_active'] if active else C['bg_panel']
        c.create_polygon(x, y, x+w, y-h, x+w, y+h, fill=fill, outline='black')
        c.create_polygon(x+w, y, x+2*w, y-h, x+2*w, y+h, fill=fill, outline='black')
        # 閥桿
        c.create_line(x+w, y, x+w, y-h-5, width=2)
        c.create_oval(x+w-3, y-h-8, x+w+3, y-h-2, fill='black')

# ==========================================
# 主介面
# ==========================================
class ProfessionalHMI:
    def __init__(self, root):
        self.root = root
        self.root.title("工業鍋爐控制系統 | BOILER CONTROL SYSTEM V2.0")
        self.root.geometry("1920x1080")  # 擴大尺寸
        self.root.resizable(False, False) # 禁止調整
        self.root.configure(bg=C['bg_app'])
        
        self.human = Engine()
        self.ai = Engine()
        self.ctrl = SmartController()
        self.running = False
        
        self._setup_ui()
        self._loop()

    def _setup_ui(self):
        # === 頂部工具列 ===
        toolbar = tk.Frame(self.root, bg=C['bg_app'], height=60, bd=1, relief='raised')
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        tk.Label(toolbar, text="🏭 鍋爐控制系統", font=F['h1'], bg=C['bg_app']).pack(side=tk.LEFT, padx=20)
        
        # 控制區
        ctrl_frame = tk.Frame(toolbar, bg=C['bg_app'])
        ctrl_frame.pack(side=tk.LEFT, padx=50)
        
        tk.Label(ctrl_frame, text="生產批次:", bg=C['bg_app'], font=F['body']).pack(side=tk.LEFT)
        self.scenario_var = tk.StringVar(value=list(場景.keys())[0])
        self.scenario_cb = ttk.Combobox(ctrl_frame, textvariable=self.scenario_var, 
                                        values=list(場景.keys()), font=F['body'], width=20, state='readonly')
        self.scenario_cb.pack(side=tk.LEFT, padx=10)
        
        self.btn_start = tk.Button(ctrl_frame, text="啟動 (START)", bg=C['status_run'], fg='white', 
                             font=F['body'], command=self._start, width=12)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = tk.Button(ctrl_frame, text="重置 (RESET)", bg=C['bg_panel'], 
                             font=F['body'], command=self._reset, width=12)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        # === 主分割區 (50/50 固定比例) ===
        main_content = tk.Frame(self.root, bg=C['bg_app'])
        main_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 使用 Grid 來確保嚴格的 50/50 分割
        main_content.columnconfigure(0, weight=1, uniform="group1")
        main_content.columnconfigure(1, weight=1, uniform="group1")
        main_content.rowconfigure(0, weight=1)
        
        # 左：人類操作站
        self.h_frame = self._create_station(main_content, "操作員站 (OPERATOR)", True, C['human_color'])
        self.h_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        
        # 右：AI 控制站
        self.a_frame = self._create_station(main_content, "自動控制站 (AUTO)", False, C['ai_color'])
        self.a_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

    def _create_station(self, parent, title, is_human, theme_color):
        frame = tk.Frame(parent, bg=C['bg_panel'], bd=2, relief='sunken')
        
        # HEADER
        header = tk.Frame(frame, bg=theme_color, height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=title, bg=theme_color, fg='white', font=F['h2']).pack(side=tk.LEFT, padx=10)
        
        # 內容區域使用 Frame 容器
        content = tk.Frame(frame, bg=C['bg_panel'])
        content.pack(fill=tk.BOTH, expand=True)
        
        # 80/20 垂直分割 (使用 place 進行絕對比例控制)
        
        # 上部: SCADA (80%)
        # 為了有邊距，我們在內部再放一個 frame
        scada_container = tk.Frame(content, bg=C['bg_content'], bd=1, relief='solid')
        scada_container.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.78)
        
        if is_human: self.h_canvas = tk.Canvas(scada_container, bg=C['bg_content'], highlightthickness=0)
        else: self.a_canvas = tk.Canvas(scada_container, bg=C['bg_content'], highlightthickness=0)
        
        canvas = self.h_canvas if is_human else self.a_canvas
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # 下部: 數據面板 (20%)
        dash_container = tk.Frame(content, bg=C['bg_panel'])
        dash_container.place(relx=0.01, rely=0.80, relwidth=0.98, relheight=0.19)
        
        # 狀態顯示
        stat_frame = tk.LabelFrame(dash_container, text="即時數據 (DATA)", font=F['tag'], bg=C['bg_panel'], fg=C['text_main'])
        stat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(stat_frame, text="總能耗成本:", bg=C['bg_panel'], font=F['body']).pack(anchor='w', padx=10, pady=5)
        lbl_cost = tk.Label(stat_frame, text="0.00 TWD", bg=C['bg_panel'], font=F['num_big'], fg=theme_color)
        lbl_cost.pack(anchor='w', padx=10)
        
        # 控制介面 (僅人類) 或 空白填充
        if is_human:
            ctrl_frame = tk.LabelFrame(dash_container, text="手動控制 (MANUAL)", font=F['tag'], bg=C['bg_panel'], fg=C['text_main'])
            ctrl_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
            
            self.pwr_scale = tk.Scale(ctrl_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                    label="鍋爐功率設定 %", font=F['body'], bg=C['bg_panel'], length=200)
            self.pwr_scale.pack(fill=tk.X, padx=20, pady=5)
        else:
            # AI 顯示狀態作為填充
            ai_status_frame = tk.LabelFrame(dash_container, text="AI 狀態", font=F['tag'], bg=C['bg_panel'], fg=C['text_main'])
            ai_status_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
            
            if self.ctrl.use_ai:
                if hasattr(self.ctrl, 'model_type'):
                    if self.ctrl.model_type == "rcp":
                        status_text = "🧠 RCP Agent\n(Return-Conditioned)"
                    elif self.ctrl.model_type == "offline_rl":
                        status_text = "🧠 Offline RL\n(AWR Agent)"
                    else:
                        status_text = "🧠 BC Agent\n(Behavior Cloning)"
                else:
                    status_text = "🧠 AI Agent"
                status_color = C['status_active']
            else:
                status_text = "🔧 Rule-Based V3\n(Smart Controller)"
                status_color = C['status_run']
            
            tk.Label(ai_status_frame, text=status_text, font=F['body'], bg=C['bg_panel'], fg=status_color).pack(expand=True)
        
        if is_human: self.h_lbl_cost = lbl_cost
        else: self.a_lbl_cost = lbl_cost
        
        return frame

    def _draw_system(self, canvas, engine, color):
        canvas.delete("all")
        w = canvas.winfo_width(); h = canvas.winfo_height()
        if w < 100: return
        
        # 1. 繪製主鍋爐 (左側)
        boiler_x, boiler_y = 100, h/2
        
        # 鍋爐外殼
        IndustrialRender.draw_tank(canvas, boiler_x-40, boiler_y-60, 80, 120, 0, C['bg_panel'])
        
        # 火焰/加熱指示
        if engine.kw > 0:
            c_fire = C['status_warn'] if engine.kw < 80 else C['status_alarm']
            canvas.create_oval(boiler_x-15, boiler_y+20, boiler_x+15, boiler_y+50, fill=c_fire, outline='')
            canvas.create_text(boiler_x, boiler_y+35, text="🔥", font=("Segoe UI", 16))
        
        # 溫度表
        canvas.create_rectangle(boiler_x-30, boiler_y-40, boiler_x+30, boiler_y-10, fill='black')
        canvas.create_text(boiler_x, boiler_y-25, text=f"{engine.temp:.1f}", fill='red', font=F['num'])
        
        # 標籤
        canvas.create_text(boiler_x, boiler_y-80, text="B-101\n主鍋爐", font=F['tag'], justify='center')
        
        # 2. 主管線
        pipe_y = boiler_y - 20
        canvas.create_line(boiler_x+40, pipe_y, w-50, pipe_y, width=6, fill=C['pipe'])
        
        # 3. 機台繪製
        units = list(engine.units.values())
        if not units: return
        u_spacing = (w - boiler_x - 100) / len(units)
        
        for i, u in enumerate(units):
            ux = boiler_x + 120 + i * u_spacing
            uy = pipe_y + 80
            
            # 分支管線
            flow = u['valve'] and engine.temp > u['current']
            p_co = C['pipe_flow'] if flow else C['pipe']
            canvas.create_line(ux, pipe_y, ux, uy, width=4, fill=p_co)
            
            # 閥門
            v_color = C['status_active'] if u['valve'] else C['bg_panel']
            IndustrialRender.draw_valve(canvas, ux-10, pipe_y+30, 10, u['valve'])
            
            # 槽體
            fill_c = C['status_active'] if u['state'] == '保溫' else (
                C['status_run'] if u['state'] == '完成' else C['status_warn'])
            
            IndustrialRender.draw_tank(canvas, ux-30, uy, 60, 80, 0.8, fill_c)
            
            # 數據標籤
            canvas.create_text(ux, uy+40, text=f"{u['current']:.1f}\nsp:{u['target']:.0f}", font=F['tag'])
            canvas.create_text(ux, uy-15, text=u['name'], font=F['tag'], fill=C['text_dim'])
            
            # 狀態燈
            s_col = C['status_run'] if u['state']=='完成' else (
                C['status_active'] if u['state']=='保溫' else C['status_warn'])
            canvas.create_oval(ux-40, uy+90, ux-30, uy+100, fill=s_col, outline='black')
            canvas.create_text(ux, uy+95, text=u['state'], anchor='w', font=F['tag'])

    def _start(self):
        tasks = 場景[self.scenario_var.get()]
        self.human.load(copy.deepcopy(tasks)); self.ai.load(copy.deepcopy(tasks))
        self.running = True
        self.btn_start.config(state='disabled'); self.scenario_cb.config(state='disabled')
    
    def _reset(self):
        self.running = False
        self.human.reset(); self.ai.reset()
        self.btn_start.config(state='normal'); self.scenario_cb.config(state='readonly')
        self.h_lbl_cost.config(text="0.00 TWD"); self.a_lbl_cost.config(text="0.00 TWD")

    def _loop(self):
        if self.running:
            try:
                h_pow = self.pwr_scale.get()
                a_pow = self.ctrl.decide(self.ai.temp, self.ai.units)
                
                if not self.human.done(): self.human.step(h_pow)
                if not self.ai.done(): self.ai.step(a_pow)
                
                self._draw_system(self.h_canvas, self.human, C['human_color'])
                self._draw_system(self.a_canvas, self.ai, C['ai_color'])
                
                self.h_lbl_cost.config(text=f"{self.human.cost:.2f} TWD")
                self.a_lbl_cost.config(text=f"{self.ai.cost:.2f} TWD")
                
                if self.human.done() and self.ai.done():
                    self.running = False; self.btn_start.config(state='normal')
                    
                    # 保存玩家數據
                    try:
                        import pandas as pd
                        import os
                        os.makedirs('user_data', exist_ok=True)
                        ts = int(time.time())
                        df = pd.DataFrame(self.human.history)
                        csv_path = f"user_data/human_log_{ts}.csv"
                        df.to_csv(csv_path, index=False)
                        print(f"User data saved to {csv_path}")
                    except Exception as err:
                        print(f"Save data failed: {err}")

                    diff = self.human.cost - self.ai.cost
                    res = "AI 勝出" if diff > 0 else "人類勝出"
                    messagebox.showinfo("完成", f"測試結束\n{res}\n差距: {abs(diff):.2f} TWD\n\n(您的操作數據已保存以供學習)")
            except Exception as e:
                print(e)
                
        self.root.after(50, self._loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalHMI(root)
    root.mainloop()
