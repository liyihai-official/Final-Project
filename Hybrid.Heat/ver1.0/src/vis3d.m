% fid = fopen('../build/test_3d.bin', 'rb');

% Row = fread(fid, 1, 'uint32');
% Col = fread(fid, 1, 'uint32');
% Dep = fread(fid, 1, 'uint32');

% A = fread(fid, [Row * Col * Dep], 'float32'); 
% A = reshape(A, [Dep, Col, Row]);
% A = permute(A, [2,3,1]);



% [X,Y,Z] = meshgrid(1:Row, 1:Col, 1:Dep);

% X_norm = (X - 1) / (Row - 1);
% Y_norm = (Y - 1) / (Col - 1);
% Z_norm = (Z - 1) / (Dep - 1);


% F = X_norm + Y_norm + Z_norm ...
%     - 2 * X_norm .* Y_norm ...
%     - 2 * X_norm .* Z_norm ...
%     - 2 * Y_norm .* Z_norm ...
%     + 4 * X_norm .* Y_norm .* Z_norm;

% figure;
% h = gcf;
% title("TEST");

% % max((A(2:Row-2, 2:Col-2,2:Dep-2) - F(2:Row-2, 2:Col-2,2:Dep-2)).^2, [], 'all')
% % mean((A(2:Row-2, 2:Col-2,2:Dep-2) - F(2:Row-2, 2:Col-2,2:Dep-2)).^2, 'all')

% % 绘制数据 A 的切片图
% slice(X_norm, Y_norm, Z_norm, A, [0.5 1], [0.5 1], [0 1]);
% % hold on; % 保持当前图形

% % % 绘制函数 F 的切片图
% % slice(X_norm, Y_norm, Z_norm, F, [0.5 1], [0.5 1], [0 0.5 1]);
% % saveas(gcf, '../out/main2.png')
% colormap(jet);
% colorbar;

% xlabel('Row-axis Label'); % 替换为你想要的 X 轴标签
% ylabel('Col-axis Label'); % 替换为你想要的 Y 轴标签
% zlabel('Depth-axis Label'); % 替换为你想要的 Z 轴标签

% % legend('Data A', 'Function F'); % 添加图例区分两者



fid = fopen('../build/X.bin', 'rb');
Row = fread(fid, 1, 'uint32');
Col = fread(fid, 1, 'uint32');
A = fread(fid, [Row * Col], 'float32'); 
A = reshape(A, [Col, Row]);
A = permute(A, [2,1]);

fid = fopen('../build/u.bin', 'rb');
Row = fread(fid, 1, 'uint32');
Col=1;
U = fread(fid, [Row * Col], 'float32'); 
U = reshape(U, [Col, Row]);
U = permute(U, [2,1]);


% Extract the x, y, z coordinates
x = A(:, 1);
y = A(:, 2);
z = A(:, 3);

% Create a 3D scatter plot
scatter3(x, y, z, 36, U, 'filled');

% Add labels and title
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Scatter Plot of Points with Values');

% Add a colorbar to represent the values
colorbar;

% Adjust view angle (optional)
view(45, 25);
saveas(gcf, '../out/main2_dataset.png')