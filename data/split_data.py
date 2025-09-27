import json
import random

def split_dataset(input_file, output_file1, output_file2, split_by='count', value=0):
    """
    随机将数据集分为两部分。

    参数:
        input_file (str): 输入的JSON文件路径。
        output_file1 (str): 输出的第一部分JSON文件路径。
        output_file2 (str): 输出的第二部分JSON文件路径。
        split_by (str): 分割方式，'count'表示按数量分割，'ratio'表示按比例分割。
        value (int/float): 分割值，数量或比例。
    """
    # 读取数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打乱数据集
    random.shuffle(data)

    # 根据分割方式计算分割点
    if split_by == 'count':
        if not isinstance(value, int) or value <= 0 or value > len(data):
            raise ValueError("当选择按数量分割时，value必须是一个大于0且小于等于数据集大小的整数。")
        split_point = value
    elif split_by == 'ratio':
        if not (0 < value < 1):
            raise ValueError("当选择按比例分割时，value必须是一个0到1之间的浮点数。")
        split_point = int(len(data) * value)
    else:
        raise ValueError("split_by参数必须是'count'或'ratio'。")

    # 分割数据集
    part1 = data[:split_point]
    part2 = data[split_point:]

    # 保存分割后的数据集
    with open(output_file1, 'w', encoding='utf-8') as f:
        json.dump(part1, f, ensure_ascii=False, indent=4)
    with open(output_file2, 'w', encoding='utf-8') as f:
        json.dump(part2, f, ensure_ascii=False, indent=4)

    print(f"数据集已分割完成：{len(part1)}条数据保存到{output_file1}，{len(part2)}条数据保存到{output_file2}。")

# 示例用法
if __name__ == "__main__":
    input_file = r"D:\a-study\KG\MSRA\MSRA\data\train_data\train.json"  # 输入文件路径
    output_file1 = "labeled_10%.json"  # 输出文件1路径
    output_file2 = "unlabeled_90%.json"  # 输出文件2路径

    # 用户定义分割方式
    split_by = input("请输入分割方式（'count'表示按数量，'ratio'表示按比例）：").strip()
    value = input("请输入分割值（数量或比例）：").strip()

    # 根据分割方式解析分割值
    if split_by == 'count':
        value = int(value)  # 将分割值转换为整数
    elif split_by == 'ratio':
        value = float(value)  # 将分割值转换为浮点数

    split_dataset(input_file, output_file1, output_file2, split_by, value)