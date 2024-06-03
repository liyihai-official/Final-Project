
fid = fopen('mat.bin', 'rb');

rows = fread(fid, 1, 'uint64');
cols = fread(fid, 1, 'uint64');
height = fread(fid, 1, 'uint64');

% A = fread(fid, [rows*cols], 'double');
A = fread(fid, [rows*cols*height], 'double');

A = reshape(A, [height, cols, rows]);
% A = reshape(A, [cols, rows]);

A = permute(A, [2, 3, 1]); % 由于MATLAB和C++的存储顺序不同，需要调整
% A = permute(A, [2, 1]); % 由于MATLAB和C++的存储顺序不同，需要调整


[X, Y, Z] = meshgrid(1:rows, 1:cols, 1:height);
% [X, Y] = meshgrid(1:rows, 1:cols);
% V = A(:);

figure;
% scatter(X(:), Y(:), 28, V, 'filled');
% scatter3(X(:), Y(:), Z(:), 100, V, 'filled');
slice(X, Y, Z, A, [ (rows+1)/2 rows-1], [cols-1], [1+1 height/2 ]);
title('3D Scatter Plot');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');

xlim([2, rows-1]);
ylim([2, cols-1]);
zlim([2, height-1]);
colormap(jet)
colorbar;

% 创建示例数据
% [x, y, z] = meshgrid(-2:0.2:2, -2:0.2:2, -2:0.2:2);
% v = x .* exp(-x.^2 - y.^2 - z.^2);

% 绘制三维热力图
% figure;
% slice(x, y, z, v, [-2 0 2], [], [-2 0 2]); % 在x=-2,0,2和z=-2,0,2处取切片
% shading interp; % 插值着色
% colorbar; % 添加颜色条
% title('三维空间热力图');
% xlabel('X轴');
% ylabel('Y轴');
% zlabel('Z轴');
