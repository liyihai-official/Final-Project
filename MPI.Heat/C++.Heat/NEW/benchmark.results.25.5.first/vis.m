
speed = @(v) [v(1)/v(1), v(1)/v(2), v(1)/v(3), v(1)/v(4), v(1)/v(5), v(1)/v(6), v(1)/v(7)];

proc = [1, 2, 4, 8, 16, 32, 64];
speed_weak = @(v) [v(1)/v(1)*proc(1), v(1)/v(2)*proc(2), v(1)/v(3)*proc(3), v(1)/v(4)*proc(4), ...
    v(1)/v(5)*proc(5), v(1)/v(6)*proc(6), v(1)/v(7)*proc(7)];

strong_1024 = [
  4.5803e+06,
  2.30751e+06,
  1.24368e+06,
  650729,
  333912,
  205066,
  142470
];

% strong_1024_omp = [
%   6.60267e+06,
%   2.49456e+06,
%   752795,
%   2.60923e+07
% ];


weak = [
  1249.95,
  4990.25,
  5765.79,
  21907.1,
  24341.2,
  91958.1,
  140261
];

% weak_omp = [
%   2526.58,
%   9866.88,
%   10153.9,
%   39050.5,
%   42244.3,
%   183143,
%   2.59875e+07
% ];

strong_1024 = speed(strong_1024);
% strong_1024_omp = speed(strong_1024_omp);



weak = speed_weak(weak);
% weak_omp = speed_weak(weak_omp);

figure;
loglog(proc, strong_1024, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');

hold on;
grid();
% plot(proc, strong_1024_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
loglog(proc, proc, '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
legend(["Real Speedup", "Idea Speedup"])
title("Strong Scaling Speedup Ratio");

xlim([1,256]);
ylim([1,256]);

xticks([1,2,4,8,16,32,64,128,256,512,1024]);
yticks([1,2,4,8,16,32,64,128,256,512,1024]);
hold off;
saveas(gcf, "StrongNew1.png");

figure;
loglog(proc, weak, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
% loglog(proc, weak_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');

hold on;
grid();

loglog(proc, [1, 1, 1, 1, 1, 1, 1], '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
% legend(["Weak", "With Omp", "Idea"]);

xlim([1,256]);
ylim([0,1]);

xticks([1,2,4,8,16,32,64,128,256,512,1024]);
yticks([1,2,4,8,16,32,64,128,256,512,1024]);

title("Efficiency of Weak Scaling");
hold off;
saveas(gcf, "WeakNew1.png");