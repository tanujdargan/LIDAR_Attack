import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


#%% md
# # POint cloud processing for reference
#%%
point_cloud = np.fromfile('real_08600.bin', dtype=np.float32).reshape(-1, 4)
print(type(point_cloud))
print(point_cloud.shape)
np.savetxt("converted.txt", point_cloud, fmt="%.6f", delimiter=",")
print(len(point_cloud))
#%% md
# ### Visulaisation in matplotlib
#%%
x, y, z, r = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3]
r_norm = (r - r.min()) / (r.max() - r.min())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=r_norm, cmap='viridis', s=0.5)
plt.show()
#%% md
# ## Visualisation in open3d
#%%
pcd=o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
pcd.colors=o3d.utility.Vector3dVector(np.stack([r_norm, np.zeros_like(r_norm), np.zeros_like(r_norm)], axis=1))
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd,
    voxel_size=0.2  # uniform 0.2m grid
)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([voxel_grid])