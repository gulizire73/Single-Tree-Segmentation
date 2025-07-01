import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


# 加载点云数据
input_path = "E:\\d32\\jiguang\\09\\Code\\Z9_testing_label.txt"
data = np.loadtxt(input_path, skiprows=1)  # 忽略表头行

# 提取 label=0 和 label=1 的点
background_points = data[data[:, 3] == 0]      # label=0 的背景点
tree_points = data[data[:, 3] == 1][:, :4]     # label=1 的树木点 (x,y,z,label)

print(f"共找到 {len(tree_points)} 个树木点")
print(f"共找到 {len(background_points)} 个背景点")


# 高度分布分析（z轴）
heights = tree_points[:, 2]
height_threshold = float(input("请输入用于区分树干和树冠的最大高度阈值（例如 3.0）: "))

trunk_mask = tree_points[:, 2] <= height_threshold
trunk_points = tree_points[trunk_mask][:, :3]        # 只保留 x,y,z
crown_points = tree_points[~trunk_mask][:, :3]

print(f"树干点数量: {len(trunk_points)}")
print(f"树冠点数量: {len(crown_points)}")


#对树干点进行去噪
def remove_outliers_open3d(points, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points)

trunk_points = remove_outliers_open3d(trunk_points, nb_neighbors=20, std_ratio=2.0)
print(f"去噪后树干点数量: {len(trunk_points)}")


# 使用 DBSCAN 对树干点进行聚类
print("正在进行 DBSCAN 聚类...")
dbscan = DBSCAN(eps=0.5, min_samples=15)
trunk_labels = dbscan.fit_predict(trunk_points)

unique_labels = np.unique(trunk_labels)
num_clusters = len(unique_labels[unique_labels != -1])
print(f"检测到 {num_clusters} 个簇（不含噪声点）")
print(f"噪声点数量: {np.sum(trunk_labels == -1)}")

valid_mask = trunk_labels != -1
trunk_points_valid = trunk_points[valid_mask]
trunk_labels_valid = trunk_labels[valid_mask]


# 过滤掉较小的簇
min_points_in_cluster = 100
filtered_trunk_points = []
filtered_trunk_labels = []
current_label = 1  # 从 1 开始编号

for cluster_id in np.unique(trunk_labels_valid):
    cluster_mask = (trunk_labels_valid == cluster_id)
    cluster_points = trunk_points_valid[cluster_mask]
    
    if len(cluster_points) >= min_points_in_cluster:
        filtered_trunk_points.append(cluster_points)
        filtered_trunk_labels.append(np.full(len(cluster_points), current_label))
        current_label += 1

if len(filtered_trunk_points) > 0:
    trunk_points_filtered = np.vstack(filtered_trunk_points)
    trunk_labels_filtered = np.hstack(filtered_trunk_labels)
else:
    trunk_points_filtered = np.empty((0, 3))
    trunk_labels_filtered = np.empty(0, dtype=int)

num_clusters_filtered = current_label - 1  # 因为是从 1 开始编号
print(f"过滤后剩余 {num_clusters_filtered} 个簇（点数 ≥ {min_points_in_cluster}）")


# 定义计算质心的函数
def compute_centroids(points, labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        cluster_points = points[labels == i+1]  # 注意这里 +1
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.full((3,), np.nan))
    return np.array(centroids)


# 将树冠点归类到最接近的树干簇中
if num_clusters_filtered > 0:
    cluster_centers = compute_centroids(trunk_points_filtered, trunk_labels_filtered, num_clusters_filtered)

    # 仅使用 x 和 y 坐标进行曼哈顿距离匹配
    crown_xy = crown_points[:, :2]
    centers_xy = cluster_centers[:, :2]
    distances = cdist(crown_xy, centers_xy, metric='cityblock')
    crown_labels = np.argmin(distances, axis=1) + 1  # 从 1 开始编号

    
    # 合并树干和树冠点云，按簇着色
    all_tree_points = np.vstack([trunk_points_filtered, crown_points])
    all_tree_labels = np.hstack([trunk_labels_filtered, crown_labels])

    # 添加 label 列到点云中
    all_tree_data = np.hstack([
        all_tree_points,
        all_tree_labels.reshape(-1, 1)
    ])

    
    
    # 保存结果
    output_path = "E:\\d32\\jiguang\\09\\Code\\output_with_background.txt"
    np.savetxt(output_path, all_tree_data, fmt='%.6f %.6f %.6f %d', comments='')
    print(f"聚类结果已保存至：{output_path}")

    
    #可视化最终结果
    cmap = plt.get_cmap('nipy_spectral')
    unique_labels_final = np.unique(all_tree_data[:, 3])
    colors = cmap(all_tree_data[:, 3] / max(1, unique_labels_final.max()))[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_tree_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Final Clustering with Background")
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    lookat_point = np.mean(all_tree_data[:, :3], axis=0)
    view_control.set_lookat(lookat_point)
    view_control.set_front([0, -1, -0.3])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.7)
    vis.run()
    vis.destroy_window()

else:
    print("未检测到有效簇（点数 ≥ {}），无法继续归类树冠点。".format(min_points_in_cluster))