
import numpy as np
from sklearn.cluster import DBSCAN

def load_txt_point_cloud(txt_path):
    """
    加载 txt 格式的点云文件 (x y z label)
    返回：points (N, 3), labels (N,)
    """
    data = np.loadtxt(txt_path, skiprows=1)
    points = data[:, :3]
    labels = data[:, 3].astype(int)
    return points, labels

def save_txt_point_cloud(output_path, points, labels):
    """
    保存点云为 txt 文件，格式：x y z label
    """
    combined = np.hstack((points, labels.reshape(-1, 1)))
    header = "x y z label"
    np.savetxt(output_path, combined, fmt="%.6f %.6f %.6f %d", delimiter=" ", comments='')
    print(f"结果已保存至: {output_path}")

def process_labels_with_dbscan(points, labels, eps=0.5, min_points=10):
    """
    对 label == 1 的点做 DBSCAN 聚类，并将除最大簇外的所有簇的标签设为 0
    原始 label == 0 的点保持为 0
    """
    # 提取 label == 1 的点
    target_mask = (labels == 1)
    target_points = points[target_mask]
    if len(target_points) == 0:
        print("没有 label == 1 的点，无需聚类")
        return labels
    # DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=min_points).fit(target_points)
    cluster_labels = db.labels_
    # 找出每个簇的大小
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_size = dict(zip(unique_clusters, counts))
    # 排除噪声 (-1)
    valid_clusters = {k: v for k, v in cluster_size.items() if k != -1}
    if not valid_clusters:
        print("DBSCAN 没有找到有效簇")
        return labels
    largest_cluster_label = max(valid_clusters, key=valid_clusters.get)
    print(f"最大簇的 DBSCAN 标签是: {largest_cluster_label}")
    # 构建新的标签数组
    new_labels = labels.copy()
    # 获取所有 label == 1 的索引
    indices = np.where(target_mask)[0]
    # 将最大簇标记为 1，其余标记为 0
    for i, idx in enumerate(indices):
        if cluster_labels[i] == largest_cluster_label:
            new_labels[idx] = 1
        else:
            new_labels[idx] = 0
    return new_labels


if __name__ == "__main__":
    # 输入和输出路径
    input_txt = r"E:\d32\jiguang\09\Code\cluster\building_tree_color.txt"
    output_txt = r"E:\d32\jiguang\09\Code\tree_cluster.txt"
    # 参数设置
    eps = 1.70          # 邻域半径
    min_points = 100    # 构成簇的最小点数
    # 加载数据
    points, labels = load_txt_point_cloud(input_txt)
    print(f"加载了 {len(points)} 个点")
    # 处理标签
    new_labels = process_labels_with_dbscan(points, labels, eps=eps, min_points=min_points)
    # 保存结果
    save_txt_point_cloud(output_txt, points, new_labels)

