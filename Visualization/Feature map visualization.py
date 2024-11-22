import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def visualize_tomato_feature_processing(input_size=(608, 608), output_size=(304, 304)):
    """
    可视化番茄图像特征处理过程
    Args:
        input_size: 原始输入尺寸(608x608)
        output_size: 特征输出尺寸(304x304)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 左图：特征分块处理
    ax1.set_title('Feature Block Processing', pad=20)
    
    # 创建2x2基本模式
    base_pattern = np.array([
        [(1,'red'), (2,'yellow'), (1,'red'), (2,'yellow')],
        [(3,'lightgreen'), (4,'blue'), (3,'lightgreen'), (4,'blue')],
        [(1,'red'), (2,'yellow'), (1,'red'), (2,'yellow')],
        [(3,'lightgreen'), (4,'blue'), (3,'lightgreen'), (4,'blue')]
    ])
    
    # 绘制基本分块
    block_size = 1.0
    for i in range(4):
        for j in range(4):
            number, color = base_pattern[i][j]
            # 主块
            rect = Rectangle((j*block_size, -i*block_size), 
                           block_size, block_size,
                           facecolor=color,
                           edgecolor='white',
                           alpha=0.8)
            ax1.add_patch(rect)
            # 添加数字标注
            ax1.text(j*block_size + block_size/2, 
                    -i*block_size - block_size/2,
                    str(number),
                    ha='center', va='center',
                    color='black',
                    fontweight='bold',
                    fontsize=10)
            
            # 添加特征维度标注（在边缘）
            if i == 0 and j == 3:
                ax1.text(j*block_size + block_size*1.2, 
                        0,
                        f'Channel: {output_size[0]}',
                        rotation=0,
                        va='center')
    
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-4.5, 0.5)
    ax1.axis('off')
    
    # 2. 右图：特征堆叠
    ax2.set_title('Feature Maps Stacking', pad=20)
    
    # 绘制特征图堆叠效果
    n_features = 32  # 特征通道数
    stack_offset = 0.05  # 堆叠层的偏移量
    
    # 绘制堆叠的特征图
    for i in range(n_features):
        offset = i * stack_offset
        # 使用不同颜色代表不同特征层
        if i < n_features//4:  # 底层特征(表示边缘等低级特征)
            color = 'blue'
            alpha = 0.1
        elif i < n_features//2:  # 中层特征
            color = 'cyan'
            alpha = 0.1
        elif i < 3*n_features//4:  # 高层特征
            color = 'yellow'
            alpha = 0.1
        else:  # 最高层特征(表示语义特征)
            color = 'red'
            alpha = 0.1
            
        # 绘制特征图
        rect = Rectangle((-1 + offset, -1 + offset), 
                        2, 2,
                        facecolor=color,
                        edgecolor='gray',
                        alpha=alpha,
                        linewidth=0.5)
        ax2.add_patch(rect)
        
        # 在最顶层添加维度标注
        if i == n_features - 1:
            ax2.text(-1 + offset, 1 + offset, 
                    f'{output_size[0]}×{output_size[1]}×{n_features}',
                    fontsize=10)
            # 添加特征数值(1)
            for x in range(2):
                for y in range(2):
                    ax2.text(-0.5 + x + offset, 
                            -0.5 + y + offset,
                            '1',
                            ha='center', va='center',
                            color='black',
                            fontweight='bold')
    
    ax2.set_xlim(-1.5, 2)
    ax2.set_ylim(-1.5, 2)
    ax2.axis('off')
    
    # 添加转换箭头
    plt.arrow(5, -2, 1, 0, 
             head_width=0.2, 
             head_length=0.2, 
             fc='black', 
             ec='black',
             width=0.05)
    
    # 添加说明文本
    fig.text(0.5, -0.05, 
             f'Input: {input_size[0]}×{input_size[1]}×3 → Output: {output_size[0]}×{output_size[1]}×{n_features}',
             ha='center', fontsize=10)
            
    plt.tight_layout()
    return fig

# 创建可视化
fig = visualize_tomato_feature_processing()
plt.show()
