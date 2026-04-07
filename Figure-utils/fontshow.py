import matplotlib.font_manager as fm

# 获取所有已注册字体名称并排序
fonts = sorted([f.name for f in fm.fontManager.ttflist])

# # 打印所有字体
# for font in fonts:
#     print(font)

# 筛选包含特定关键词的字体
chinese_fonts = [f for f in fonts if 'Hei' in f or 'Song' in f or 'STIX' in f]
print(chinese_fonts)