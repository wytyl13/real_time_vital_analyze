import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 你的数据
data = [
    'FF 02 53 D7 02 4E CC 02 48 3E 02 48 B5 02 4E 95 02 60 95 02 6A 94 02 71 6D 02 71 3B 02 75 CD 02 78 A6 02 7A 64 02 6E 9C 02 62 2D 02 4D D5 02 41 F2 02 32 EC 02 2A C2 02 1B A1 02 11 0B 02 00 CD 01 FD BB 01 FB 56 02 07 F8 02 0F 40 02 1B F9 02 24 7A 02 31 EC 02 3A 02 02 46 9B 02 4D C1 02 5A F1 02 63 23 02 6F 33 02 6B E6 02 64 F8 02 50 83 02 44 F7 02 39 D4 02 42 4B 02 49 14 02 53 1E 02 53 56 02 54 B5 02 4F 32 02 50 02 02 4D 02 02 4A FE 02 47 70 02 4C 1D 02 4F 51 02 58 B7 02 5B 2B 02 5F FD 02 5D 45 02 5F 71 02 5B 11 02 5A E8 02 57 8F 02 5B ED 02 5C 2C 02 5D D7 02 5E 4E 02 60 52 02 5B E6 02 5C 86 02 57 A2 02 55 D0 02 4F F8 02 4B 55 02 41 CC 02 3E 51 02 3A 92 02 3C A3 02 3E 2A 02 43 74 02 48 AB 02 4E F1 02 54 39 02 59 1A 02 59 B2 02 5E 86 02 61 9C 02 68 69 02 6B E1 02 6E A7 02 69 52 02 63 A3 02 5E 11 02 5B 1F 02 59 53 02 59 D2 02 5C 4B 02 5C E6 02 5B 15 02 5B B8 02 56 BC 02 51 79 02 4F 71 02 4C DB 02 50 91 02 58 8C 02 63 C2 02 6F 0B 02 75 E5 02 79 E0 02 7E E5 02 84 47 02 8B E1 02 8C 7F 02 83 DF 02 76 81 02 66 C5 02 58 A3 02 4B 4F 02 3B D1 02 2C 0B 02 18 F9 02 0A 6F 01 FF 6F 01 FC D6 01 FE 3F 02 0B CE 02 17 8A 02 28 DA 02 38 4E 02 47 8A 02 51 BA',
    'FF 02 5A E2 02 5D 41 02 5E C6 02 5E BD 02 64 7B 02 69 9F 02 6D 90 02 6A 50 02 67 91 02 5C B1 02 4F AA 02 3A 2C 02 25 F7 02 10 71 02 00 FA 01 F0 9B 01 E8 32 01 E6 08 01 EE D1 01 F9 9F 02 08 79 02 12 88 02 22 1F 02 2C 04 02 36 31 02 36 F5 02 37 51 02 32 05 02 39 AE 02 3E DF 02 48 61 02 4A 31 02 4E 09 02 4D 76 02 50 C4 02 4D 38 02 4B 68 02 44 EE 02 47 25 02 46 12 02 4D 6E 02 50 64 02 58 29 02 56 68 02 5A 5D 02 55 AC 02 52 1C 02 48 59 02 3E 42 02 31 E2 02 32 33 02 30 6E 02 36 9D 02 33 0D 02 34 F0 02 30 8E 02 34 0E 02 34 E8 02 38 7B 02 39 20 02 42 E3 02 46 55 02 4E DE 02 4C 81 02 49 F0 02 46 EF 02 50 35 02 5F 17 02 71 E1 02 77 F5 02 7D 31 02 7C 69 02 8A B8 02 99 96 02 A9 D2 02 A9 7F 02 A1 58 02 85 A1 02 6A BE 02 54 CF 02 4D BC 02 48 5E 02 48 0F 02 41 EF 02 3E F0 02 36 AE 02 36 0D 02 3B C8 02 4F 95 02 65 76 02 7C 3E 02 84 FD 02 88 E8 02 88 4F 02 8E 51 02 93 AC 02 9A A8 02 97 62 02 93 4F 02 88 B5 02 7F 8E 02 75 E1 02 72 BC 02 74 10 02 7A F8 02 7C 40 02 7C 26 02 77 6C 02 75 CE 02 73 BA 02 75 70 02 76 B9 02 7B B9 02 80 88 02 8A 05 02 8F 6C 02 94 16 02 95 F9 02 94 F3 02 95 29 02 95 79 02 98 FB 02 9F 0D 02 A2 0F 02 A4 FE 02 A5 95 02 9E 94 02 98 A7',
    'FF 02 93 B9 02 90 8D 02 8F 3C 02 8C E6 02 87 B2 02 85 2E 02 83 58 02 8B AA 02 92 69 02 94 F3 02 92 80 02 8E 48 02 87 7C 02 85 FC 02 83 B2 02 84 4F 02 81 8F 02 7F 97 02 78 DC 02 73 DC 02 6E 51 02 71 81 02 75 23 02 7C E1 02 7C 84 02 7C 61 02 6F 65 02 69 22 02 65 91 02 69 13 02 64 52 02 5D 16 02 49 B9 02 38 65 02 22 17 02 18 3C 02 1A 17 02 2D E6 02 3E A0 02 4E 94 02 51 AF 02 55 0E 02 52 34 02 5A 06 02 5D 7C 02 60 85 02 5A 8E 02 56 34 02 47 F3 02 36 74 02 1B 3C 01 FE B1 01 E1 1F 01 CD 18 01 C1 0F 01 C4 68 01 CB B6 01 DB CE 01 EB 11 01 FB D6 02 07 BE 02 19 08 02 27 2B 02 36 12 02 3F BC 02 47 FC 02 4B 1C 02 50 4F 02 51 4D 02 55 1E 02 56 93 02 57 2F 02 4F C2 02 44 C3 02 3B 41 02 38 75 02 3B C9 02 45 45 02 4A 47 02 4A 07 02 43 97 02 3E CA 02 38 50 02 36 05 02 36 30 02 3D B5 02 43 AC 02 4B B1 02 4E 74 02 4F F8 02 4E B6 02 4E 6D 02 4E 82 02 52 9A 02 55 15 02 56 F7 02 57 3F 02 58 59 02 53 B6 02 51 9D 02 48 C7 02 40 A2 02 38 84 02 36 AC 02 33 C4 02 36 E8 02 39 AA 02 3D 9A 02 3D 9B 02 3E CB 02 3D 00 02 3E AE 02 3F 98 02 49 8F 02 4E 75 02 54 62 02 54 52 02 54 2D 02 52 3F 02 55 59 02 54 AF 02 58 50 02 59 42 02 59 C7 02 56 B5 02 56 E1 02 56 DA 02 59 8C',
    'FF 02 32 37 02 35 0A 02 31 B5 02 34 B6 02 32 F0 02 34 A5 02 34 33 02 35 D6 02 34 B9 02 36 9E 02 35 39 02 37 57 02 36 B3 02 36 51 02 36 03 02 36 ED 02 38 B3 02 39 28 02 36 89 02 39 C5 02 36 B3 02 38 1C 02 38 A1 02 39 01 02 37 93 02 37 D3 02 37 1F 02 39 6D 02 37 91 02 39 F4 02 39 49 02 3B 5D 02 3A 48 02 3B 9D 02 39 46 02 3B A6 02 38 5F 02 3B 2D 02 38 DB 02 3A 75 02 3A 01 02 3B 3D 02 39 1B 02 3B 37 02 39 7D 02 3B F0 02 38 CC 02 3A 6F 02 39 08 02 3A CD 02 39 7D 02 3A F9 02 39 98 02 3C 6D 02 3B 02 02 3D 53 02 3B 81 02 3C D2 02 39 4D 02 3C F5 02 3C 75 02 3E 0C 02 3A 99 02 3D 7A 02 3B 48 02 3D BC 02 3B 14 02 3E 26 02 3B 3A 02 3C 9F 02 3A 5C 02 3D 41 02 3A CB 02 3D A8 02 3B 8F 02 3F 18 02 3B 71 02 3F 2E 02 3C C9 02 3E 54 02 39 05 02 3C 1A 02 39 0B 02 3A 63 02 38 7C 02 3B 42 02 3A 2C 02 3D 0B 02 3A FB 02 3C C5 02 3A A4 02 3C 59 02 39 BA 02 3B 6B 02 3B 5E 02 3C 25 02 3B 36 02 3B E1 02 38 55 02 39 4D 02 37 28 02 3A 1B 02 36 EB 02 3A 95 02 38 99 02 38 51 02 39 8F 02 3C 1F 02 3A BE 02 3B 9F 02 3A DE 02 3A 82 02 38 6D 02 3A 43 02 38 1B 02 3A 0A 02 38 46 02 37 90 02 37 1D 02 37 ED 02 35 BD 02 37 FF 02 37 A5 02 37 72 02 39 0B 02 39 29 02 38 71 02 38 31',
    'FF 02 30 F5 02 2F CC 02 31 B6 02 31 12 02 31 36 02 31 68 02 30 2C 02 31 4E 02 2F D2 02 30 7B 02 30 1A 02 31 24 02 32 C6 02 34 24 02 34 24 02 35 05 02 32 EF 02 35 9E 02 33 A0 02 34 C4 02 31 EB 02 32 39 02 31 A7 02 31 92 02 30 9F 02 32 3B 02 30 15 02 31 3D 02 2F B5 02 31 6B 02 30 92 02 31 C5 02 31 B3 02 32 C9 02 31 F8 02 33 B6 02 32 E9 02 32 61 02 32 AD 02 32 53 02 2F EA 02 32 E3 02 30 A6 02 30 77 02 2F 6A 02 32 00 02 31 17 02 31 85 02 30 67 02 31 73 02 31 23 02 31 F0 02 31 4D 02 32 AE 02 30 54 02 33 29 02 30 BC 02 32 16 02 30 78 02 32 C4 02 2E C0 02 31 35 02 30 9C 02 32 93 02 2E 86 02 31 A1 02 2E F3 02 32 01 02 30 07 02 32 98 02 30 E6 02 34 18 02 31 03 02 33 4F 02 31 7A 02 33 30 02 30 0D 02 33 83 02 30 76 02 32 DB 02 31 23 02 34 4B 02 31 59 02 34 8D 02 30 B8 02 34 18 02 31 F9 02 34 EE 02 32 82 02 34 40 02 31 79 02 35 13 02 31 AD 02 34 05 02 32 14 02 33 26 02 31 7F 02 32 C9 02 30 A6 02 32 1C 02 30 16 02 33 9F 02 31 74 02 32 EA 02 30 DE 02 33 13 02 31 E1 02 33 33 02 32 5A 02 32 32 02 32 4E 02 34 43 02 34 CB 02 36 30 02 33 59 02 34 66 02 33 BD 02 35 2B 02 34 5F 02 36 53 02 32 0C 02 34 3D 02 31 75 02 32 F8 02 33 0B 02 34 02 02 32 23 02 34 44'
]



def parse_bcg_data_continuous_mix(hex_strings):
    """解析体震波数据，保持时间连续性"""
    values = []
    times = []
    markers = []  # 记录每个点的标记类型：0表示02，1表示01
    
    time_counter = 0
    sampling_interval = 0.01  # 10ms
    
    if isinstance(hex_strings, str):
        hex_strings = [hex_strings]
    
    for hex_string in hex_strings:
        hex_values = hex_string.split()
        i = 1  # 跳过FF
        
        while i < len(hex_values):
            if hex_values[i] == '02':
                if i + 2 < len(hex_values):
                    high_byte = int(hex_values[i + 1], 16)
                    low_byte = int(hex_values[i + 2], 16)
                    value = (high_byte << 8) | low_byte
                    if value >= 32768:
                        value -= 65536
                    values.append(value)
                    times.append(time_counter)
                    markers.append(0)  # 标记为02数据
                    time_counter += sampling_interval
                    i += 3
                else:
                    i += 1
            elif hex_values[i] == '01':
                if i + 2 < len(hex_values):
                    high_byte = int(hex_values[i + 1], 16)
                    low_byte = int(hex_values[i + 2], 16)
                    value = (high_byte << 8) | low_byte
                    if value >= 32768:
                        value -= 65536
                    values.append(value)
                    times.append(time_counter)
                    markers.append(1)  # 标记为01数据
                    time_counter += sampling_interval
                    i += 3
                else:
                    i += 1
            else:
                i += 1
    return np.array(values), np.array(times), np.array(markers)


def parse_bcg_data_continuous(hex_strings):
    """解析体震波数据，保持时间连续性"""
    values = []
    times = []
    markers = []  # 记录每个点的标记类型：0表示02，1表示01
    
    time_counter = 0
    sampling_interval = 0.01  # 10ms
    
    for hex_string in hex_strings:
        hex_values = hex_string.split()
        i = 1  # 跳过FF
        
        while i < len(hex_values):
            if hex_values[i] == '02':
                if i + 2 < len(hex_values):
                    high_byte = int(hex_values[i + 1], 16)
                    low_byte = int(hex_values[i + 2], 16)
                    value = (high_byte << 8) | low_byte
                    if value >= 32768:
                        value -= 65536
                    values.append(value)
                    times.append(time_counter)
                    markers.append(0)  # 标记为02数据
                    time_counter += sampling_interval
                    i += 3
                else:
                    i += 1
            elif hex_values[i] == '01':
                if i + 2 < len(hex_values):
                    high_byte = int(hex_values[i + 1], 16)
                    low_byte = int(hex_values[i + 2], 16)
                    value = (high_byte << 8) | low_byte
                    if value >= 32768:
                        value -= 65536
                    values.append(value)
                    times.append(time_counter)
                    markers.append(1)  # 标记为01数据
                    time_counter += sampling_interval
                    i += 3
                else:
                    i += 1
            else:
                i += 1
    
    return np.array(values), np.array(times), np.array(markers)

# 解析数据
# values, times, markers = parse_bcg_data_continuous(data)
values, times, markers = parse_bcg_data_continuous_mix(data)

# 分离01和02数据用于不同颜色绘制
marker_02_indices = markers == 0
marker_01_indices = markers == 1

# 创建美观的图形
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')

# 设置背景色
ax.set_facecolor('#ffffff')

# 绘制连续波形 - 先绘制02数据点（蓝色）
if np.any(marker_02_indices):
    ax.scatter(times[marker_02_indices], values[marker_02_indices], 
               color='#2E86AB', s=1, alpha=0.8, label='02标记数据')

# 再绘制01数据点（红色）
if np.any(marker_01_indices):
    ax.scatter(times[marker_01_indices], values[marker_01_indices], 
               color='#F24236', s=8, alpha=0.9, label='01标记数据', marker='s')

# 绘制连接线（整体波形）
ax.plot(times, values, color='#7f8c8d', linewidth=0.8, alpha=0.6, zorder=0)

# 设置标题和标签
ax.set_title('完整BCG波形图 (16位有符号数据 - 连续时序)', fontsize=18, fontweight='bold', 
             color='#2c3e50', pad=20)
ax.set_xlabel('时间 (秒)', fontsize=14, color='#34495e')
ax.set_ylabel('BCG信号幅度', fontsize=14, color='#34495e')

# 美化网格
ax.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7', linewidth=0.8)
ax.set_axisbelow(True)

# 设置坐标轴样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#7f8c8d')
ax.spines['bottom'].set_color('#7f8c8d')
ax.tick_params(colors='#34495e', labelsize=11)

# 添加图例
legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                   shadow=True, fontsize=12, framealpha=0.9)
legend.get_frame().set_facecolor('#ecf0f1')
legend.get_frame().set_edgecolor('#bdc3c7')

# 计算统计信息
marker_02_count = np.sum(markers == 0)
marker_01_count = np.sum(markers == 1)

# 添加统计信息框
stats_text = f'''数据统计信息:
• 总数据点: {len(values)}
• 数据时长: {max(times):.2f} 秒
• 02标记点数: {marker_02_count}
• 01标记点数: {marker_01_count}
• 平均值: {np.mean(values):.1f}
• 标准差: {np.std(values):.1f}
• 最小值: {np.min(values)}
• 最大值: {np.max(values)}'''

# 创建统计信息框
text_box = ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=12,
                   horizontalalignment='right', verticalalignment='bottom', 
                   bbox=dict(boxstyle='round,pad=1.0', 
                   facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.95),
                   color='#2c3e50')  # 移除 family='monospace'

# 设置x轴范围和刻度
if len(times) > 0:
    ax.set_xlim(0, max(times) * 1.02)
    max_time = max(times)
    if max_time <= 5:
        tick_interval = 0.5
    elif max_time <= 10:
        tick_interval = 1.0
    else:
        tick_interval = 2.0
    ax.set_xticks(np.arange(0, max_time + tick_interval, tick_interval))

# 设置y轴范围（留出一些边距）
y_margin = (np.max(values) - np.min(values)) * 0.1
ax.set_ylim(np.min(values) - y_margin, np.max(values) + y_margin)

# 添加零线
ax.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.5, linewidth=1)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 打印详细统计信息
print("=" * 60)
print("体震波数据解析完成！(连续时序版本)")
print("=" * 60)
print(f"02标记数据点数: {marker_02_count}")
print(f"01标记数据点数: {marker_01_count}")
print(f"总数据点数: {len(values)}")
print(f"数据时长: {max(times):.3f} 秒")
print(f"采样间隔: 10ms")
print("=" * 60)
print("02标记数据统计:")
if marker_02_count > 0:
    marker_02_values = values[markers == 0]
    print(f"  平均值: {np.mean(marker_02_values):.2f}")
    print(f"  标准差: {np.std(marker_02_values):.2f}")
    print(f"  范围: {np.min(marker_02_values)} ~ {np.max(marker_02_values)}")
print("01标记数据统计:")
if marker_01_count > 0:
    marker_01_values = values[markers == 1]
    print(f"  平均值: {np.mean(marker_01_values):.2f}")
    print(f"  标准差: {np.std(marker_01_values):.2f}")
    print(f"  范围: {np.min(marker_01_values)} ~ {np.max(marker_01_values)}")
print("=" * 60)