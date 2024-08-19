fid = fopen('../build/test.bin', 'rb');

Row = fread(fid, 1, 'uint32');
Col = fread(fid, 1, 'uint32');

% [X,Y]=meshgrid(1:Col, 1:Row);

% X_norm = (X - 1) / (Row - 1);
% Y_norm = (Y - 1) / (Col - 1);

x = linspace(0,1,Row);
y = linspace(0,1,Col);

% 创建网格
[X, Y] = meshgrid(x, y);

% 计算解析解 u(x, y)
U = X + Y - X .* Y;

A = fread(fid, [Row * Col], 'float32');
A = reshape(A, [Col, Row]);
A = permute(A, [2,1]);

figure;
h = gcf;
title("PINN");
% contourf(x, y, U, 10);
contourf(x, y, A', 10);
colormap(jet);

colorbar;
saveas(gcf, '../out/main3.png')
% caxis([-20 +20]);
% pause(0.05);

xlabel('Row-axis Label'); % 替换为你想要的 X 轴标签
ylabel('Col-axis Label'); % 替换为你想要的 Y 轴标签


% figure;
% imagesc(x, y, U);6  % 绘制热图，使用指定的 X 和 Y 坐标
% colormap(jet);

% colorbar;
% xlabel('X');
% ylabel('Y');
% title('解析解 u(x, y) = x + 2y + xy 的二维热分布图');


% mean((A(2:Row-2, 2:Col-2)' - U(2:Col-2, 2:Row-2)).^2, 'all')



% speed = @(v) [v(1)/v(1), v(1)/v(2), v(1)/v(3), v(1)/v(4), v(1)/v(5), v(1)/v(6), v(1)/v(7)];

% proc = [1, 2, 4, 8, 16, 32, 64];
% speed_weak = @(v) [v(1)/v(1)*proc(1), v(1)/v(2)*proc(2), v(1)/v(3)*proc(3), v(1)/v(4)*proc(4), ...
%     v(1)/v(5)*proc(5), v(1)/v(6)*proc(6), v(1)/v(7)*proc(7)];

% % [liy35@callan01 build]$ mpiexec -np 8 ./main1 
% % Total Converge time: 21.4085
% % Iterations: 67669
% % [liy35@callan01 build]$ mpiexec -np 64 ./main1 
% % Total Converge time: 5.0505
% % Iterations: 67669
% % [liy35@callan01 build]$ mpiexec -np 1 ./main1 
% % Total Converge time: 162.064
% % Iterations: 67669
% % [liy35@callan01 build]$ mpiexec -np 4 ./main1 
% % Total Converge time: 42.4305
% % Iterations: 67669
% % [liy35@callan01 build]$ mpiexec -np 16 ./main1 
% % Total Converge time: 11.7459
% % Iterations: 67669
% % [liy35@callan01 build]$ mpiexec -np 32 ./main1 
% % Total Converge time: 6.28171
% % Iterations: 67669

% strong_1002 = [162.064, 81.9909, 42.4305, 21.4085, 11.7459, 6.28171, 5.0505];

% strong_1002 = speed(strong_1002);
% % strong_256 = speed(strong_256);
% % strong_1024_omp = speed(strong_1024_omp);



% % weak = speed_weak(weak);
% % weak_omp = speed_weak(weak_omp);

% figure;
% hold on;
% grid on;

% % 绘制第一个图形
% loglog(proc, strong_1002, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');

% % 绘制第二个图形
% loglog(proc, proc, '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');

% % 设置X轴和Y轴为以2为底的对数刻度
% ax = gca;
% ax.XScale = 'log';
% ax.YScale = 'log';
% ax.XTickLabel = arrayfun(@num2str, 2.^(0:log2(max(proc))), 'UniformOutput', false);
% ax.YTickLabel = arrayfun(@num2str, 2.^(0:log2(max([proc strong_1002]))), 'UniformOutput', false);

% xticks([1,2,4,8,16,32,64,128,256,512,1024]);
% yticks([1,2,4,8,16,32,64,128,256,512,1024]);

% title("Strong Scaling Speedup Ratio");
% hold off;

% % 保存图形
% saveas(gcf, "Strong_main1.png");

% % figure;
% % hold on;
% % grid();
% % % plot(proc, weak, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
% % plot(proc, weak_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
% % plot(proc, [1, 1, 1, 1, 1, 1, 1], '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
% % % legend(["Weak", "With Omp", "Idea"]);
% % title("Efficiency of Weak Scaling");
% % hold off;
% % % saveas(gcf, "Weak.png");