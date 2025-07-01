import open3d as o3d
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证结果可复现
np.random.seed(42)


def compute_geometric_features(neighborhood_points):
    """
    计算点云的几何特征
    新增特征：
    - 高程 (Z)
    - 高程变化 (Z variation)
    - 强度
    - 密度
   """
    # 当前点的高程
    z_value = neighborhood_points[0, 2]

    # 邻域点的高程变化
    z_variation = np.var(neighborhood_points[:, 2])

    # 强度
    intensity = neighborhood_points[0, 3]

    # 密度
    radius = np.max(np.linalg.norm(neighborhood_points - neighborhood_points[0], axis=1))
    volume = (4 / 3) * np.pi * radius ** 3
    density = len(neighborhood_points) / (volume + 1e-10)  # 防止除以零


    return [z_value, z_variation, intensity, density]


def compute_features_for_point_cloud(point_cloud, k=50):
    #为点云中的每个点计算特征
    points = np.asarray(point_cloud)
    if len(points) == 0:
        return []

    # 使用KDTree加速邻域搜索
    tree = KDTree(points)

    features_list = []

    for i, point in enumerate(points):
        # 搜索k个最近邻点 (包含自身)
        distances, indices = tree.query(point, k=k + 1)
        neighborhood_indices = indices[1:]  # 排除自身
        neighborhood_points = points[neighborhood_indices]

        # 确保有足够的邻域点
        if len(neighborhood_points) < 5:
            # 填充默认值
            features_list.append([0] * 4)
            continue

        # 计算特征 (包含当前点)
        try:
            features = compute_geometric_features(
                np.vstack([point, neighborhood_points])
            )
            features_list.append(features)
        except Exception as e:
            print(f"计算特征时出错: {e}")
            features_list.append([0] * 4)

    return np.array(features_list)

def save_classification_results(points, labels, output_path):
    """保存分类结果"""
    with open(output_path, 'w') as f:
        for i, (x, y, z) in enumerate(points):
            f.write(f"{x} {y} {z} {int(labels[i])}\n")  # 确保标签是整数
    print(f"分类结果已保存至: {output_path}")

def main():
    # 1. 加载训练数据
    print("加载训练数据...")

    data = np.loadtxt(r"Z9_training.txt")
    print(f"训练数据加载成功")

    # 2. 准备训练数据
    print("准备训练数据...")
    # 选取树木点训练数据
    tree_mask = (data[:, -1] > 0)  # 标签大于0表示树木
    tree_points = data[tree_mask]
    print(f"树木点数量: {tree_points.shape[0]}")

    # 选取非树木点训练数据
    non_tree_mask = (data[:, -1] == 0)  # 标签0表示非树木
    non_tree_points = data[non_tree_mask]
    print(f"非树木点数量: {non_tree_points.shape[0]}")

    # 3. 特征计算 (使用随机抽样减少计算量)
    print("计算特征...")

    # 计算特征
    print("计算树木点特征...")
    tree_features = compute_features_for_point_cloud(tree_points)
    print("计算非树木点特征...")
    non_tree_features = compute_features_for_point_cloud(non_tree_points)

    # 创建标签
    y_tree = np.ones(len(tree_features))  # 树木标签为1
    y_non_tree = np.zeros(len(non_tree_features))  # 非树木标签为0

    # 合并训练数据
    X_train = np.vstack([tree_features, non_tree_features])
    y_train = np.concatenate([y_tree, y_non_tree])

    # 4. 训练分类模型
    print("训练分类模型...")
    model = RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 2},  # 让模型更关注非树木（比如墙体）
        random_state=42
    )

    model.fit(X_train, y_train)

    # 加载测试数据
    print("加载测试数据...")


    test_points = np.loadtxt(r"Z9_nonground.txt")

    print("计算测试点特征...")
    test_features = compute_features_for_point_cloud(test_points)

    # 预测
    test_labels = model.predict(test_features)

    # 8. 保存分类结果
    save_classification_results(test_points[:, :3], test_labels, "Z9_classification_results_01.txt")
    #save_classification_results(test_points, test_labels, "Z9_testing_classification_results06.txt")

    print("处理完成!")




if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f}秒")