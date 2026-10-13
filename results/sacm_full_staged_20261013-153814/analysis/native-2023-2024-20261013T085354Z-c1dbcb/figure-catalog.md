# 图目录

- trajectory：每seed两条件的末epoch曝光及反馈。无误差条，缺失首轮曝光排除。检查差异随轮次是否扩大、回落或反转。
- utility：Human Target/LLM Target nDCG5原始值及Relative Delta。无误差条；检查方向变化是否来自单侧下降，不能仅依据相对差宣称去偏。

各图同时输出PNG和SVG。
