
speed = @(v) [v(1)/v(1), v(1)/v(2), v(1)/v(3), v(1)/v(4), v(1)/v(5), v(1)/v(6), v(1)/v(7)];

proc = [1, 2, 4, 8, 16, 32, 64];
speed_weak = @(v) [v(1)/v(1)*proc(1), v(1)/v(2)*proc(2), v(1)/v(3)*proc(3), v(1)/v(4)*proc(4), ...
    v(1)/v(5)*proc(5), v(1)/v(6)*proc(6), v(1)/v(7)*proc(7)];

% strong_1024 = [
%   7.34037e+06,
%   4.37383e+06,
%   2.90165e+06,
%   1.56588e+06,
%   902118,
%   522227,
%   135117
% ];

strong_1024 = [
  440386,
  227035,
  117144,
  64039.2,
  32454.4,
  19001.2,
  17001.3
];

strong_256 = [
    1.77214e+07,
    9.04541e+06,
    1.02453e+07,
    2.07404e+06,
    1.23055e+06,
    978057,
    795652
];


% strong_1024_omp = [
%   6.60267e+06,
%   2.49456e+06,
%   752795,
%   2.60923e+07
% ];


% weak = [
%   1300.82,
%   5937.18,
%   25125.1,
%   134736
% ];

weak_omp = [
  2526.58,
  9866.88,
  10153.9,
  39050.5,
  42244.3,
  183143,
  2.59875e+07
];

strong_1024 = speed(strong_1024);
strong_256 = speed(strong_256);
% strong_1024_omp = speed(strong_1024_omp);



% weak = speed_weak(weak);
weak_omp = speed_weak(weak_omp);

figure;
hold on;
grid();
plot(proc, strong_1024, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
plot(proc, strong_256, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
% plot(proc, strong_1024_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
plot(proc, proc, '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
% legend(["Strong", "With Omp", "Idea"])
legend(["Strong 128^3", "Strong 256^3", "Idea"])
title("Strong Scaling Speedup Ratio");
hold off;
saveas(gcf, "Strong_3d.png");

% figure;
% hold on;
% grid();
% % plot(proc, weak, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
% plot(proc, weak_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
% plot(proc, [1, 1, 1, 1, 1, 1, 1], '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
% % legend(["Weak", "With Omp", "Idea"]);
% title("Efficiency of Weak Scaling");
% hold off;
% % saveas(gcf, "Weak.png");