fid = fopen('build/test_3d.bin', 'rb');

Row = fread(fid, 1, 'uint32');
Col = fread(fid, 1, 'uint32');
Dep = fread(fid, 1, 'uint32');

A = fread(fid, [Row * Col * Dep], 'float32'); 
A = reshape(A, [Dep, Col, Row]);
A = permute(A, [2,3,1]);



[X,Y,Z] = meshgrid(1:Row, 1:Col, 1:Dep);

X_norm = (X - 1) / (Row - 1);
Y_norm = (Y - 1) / (Col - 1);
Z_norm = (Z - 1) / (Dep - 1);

F = (sin(pi * X_norm) .* sin(2 * pi * Y_norm) .* sinh(sqrt(5) * pi * Z_norm)) / sinh(sqrt(5) * pi);

figure;
h = gcf;
title("TEST");

% 绘制数据 A 的切片图
slice(X_norm, Y_norm, Z_norm, A, [0.5 1], [0.5 1], [0 1]);
% hold on; % 保持当前图形

% % 绘制函数 F 的切片图
% slice(X_norm, Y_norm, Z_norm, F, [0.5 1], [0.5 1], [0 0.5 1]);

colormap(jet);
colorbar;

xlabel('Row-axis Label'); % 替换为你想要的 X 轴标签
ylabel('Col-axis Label'); % 替换为你想要的 Y 轴标签
zlabel('Depth-axis Label'); % 替换为你想要的 Z 轴标签

% legend('Data A', 'Function F'); % 添加图例区分两者