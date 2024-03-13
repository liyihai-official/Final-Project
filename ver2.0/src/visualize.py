import matplotlib.pyplot as plt
from pandas import read_csv





# df = read_csv("draft_demo3d.txt")
# df.head(10)

# fig2 = plt.figure(dpi = 150, figsize = (10,5))
# plt.plot(df["step"], df["l1_rel"], c = 'r', label = "L1 Error")
# plt.plot(df["step"], df["l2_rel"], c = 'g', label = "L2 Error")
# plt.plot(df["step"], df["li_rel"], c = 'b', label = "Inf Error")
# # plt.ylim([0,2])
# # plt.xlim([0,3e5])
# plt.legend()


# # 生成网格数据
# x = np.linspace(0, 1, 80)
# y = np.linspace(0, 1, 80)
# z = np.linspace(0, 1, 80)

# x, y, z = np.meshgrid(x, y, z)

# # Generate the grid coordinates
# xx = np.linspace(0,1,50)
# yy = np.linspace(0,1,50)

# # Create a meshgrid from the coordinates
# X, Y = np.meshgrid(xx, yy)

# grid_array = np.column_stack((X.ravel(), Y.ravel()))

# # uu = np.sum(np.sin(grid_array), axis=1) + 1
# uu = phi(grid_array) + 2.*T*d


# # Create a 3D plot
# fig = plt.figure(figsize=(15,5), dpi=200)
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot with color based on function values
# scatter = ax.scatter(grid_array[:,0], grid_array[:,1], uu, c=uu, cmap='viridis', marker='+',alpha=0.3)


# # Add colorbar to show the corresponding function values
# cbar = fig.colorbar(scatter, ax=ax, label='u(1, x, y)')

# scatter = ax.scatter(grid_array[:,0], grid_array[:,1], model(grid_array, training=False), alpha=0.2,cmap='viridis', marker='+', c = 'r')

# # Set labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Function Value')
# ax.set_title('u(1,x,y)')

# plt.show()