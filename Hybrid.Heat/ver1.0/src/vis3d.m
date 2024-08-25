fid = fopen('../build/pred.bin', 'rb');

Row = fread(fid, 1, 'uint32');
Col = fread(fid, 1, 'uint32');
Dep = fread(fid, 1, 'uint32');

A = fread(fid, [Row * Col * Dep], 'float32'); 
A = reshape(A, [Dep, Col, Row]);
A = permute(A, [2,3,1]);
% A = A(2:Row-2, 2:Col-2,2:Dep-2);

deltaX = 1 / (Row-1);
deltaY = 1 / (Col-1);
deltaZ = 1 / (Dep-1);


[X,Y,Z] = meshgrid(0:deltaX:1, 0:deltaY:1, 0:deltaZ:1);

% X_norm = (X - 1) / (Row - 1);
% Y_norm = (Y - 1) / (Col - 1);
% Z_norm = (Z - 1) / (Dep - 1);
% 

% 
F = X + Y + Z ...
    - 2 * X .* Y ...
    - 2 * X .* Z ...
    - 2 * Y .* Z ...
    + 4 * X .* Y .* Z;

figure;
h = gcf;


% max((A(2:Row-2, 2:Col-2,2:Dep-2) - F(2:Row-2, 2:Col-2,2:Dep-2)).^2, [], 'all')
diff = mean((A - F ).^2, 'all');

% 绘制数据 A 的切片图
slice(X, Y, Z, A, [0.5 1-2*deltaX], [0.5 1-2*deltaY], [deltaZ 0.5 1-2*deltaZ]);
% hold on; % 保持当前图形

% % 绘制函数 F 的切片图
% slice(X_norm, Y_norm, Z_norm, F, [0.5 1], [0.5 1], [0 0.5 1]);
colormap(jet);
colorbar;

xlabel('Row-axis Label'); % 替换为你想要的 X 轴标签
ylabel('Col-axis Label'); % 替换为你想要的 Y 轴标签
zlabel('Depth-axis Label'); % 替换为你想要的 Z 轴标签
% title("PINN-3D MSE = " +diff);
% saveas(gcf, '../out/main2_3d.png')

% legend('Data A', 'Function F'); % 添加图例区分两者