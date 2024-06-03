
speed = @(v) [v(1)/v(1), v(1)/v(2), v(1)/v(3), v(1)/v(4), v(1)/v(5), v(1)/v(6), v(1)/v(7)];

proc = [1, 2, 4, 8, 16, 32, 64];

proc_weak = [1,   4,   9,  16,  25,  36,   49,   64];
speed_weak = @(v) [v(1)/v(1)*proc_weak(1), v(1)/v(2)*proc_weak(2), v(1)/v(3)*proc_weak(3), v(1)/v(4)*proc_weak(4), ...
    v(1)/v(5)*proc_weak(5), v(1)/v(6)*proc_weak(6), v(1)/v(7)*proc_weak(7), v(1)/v(8)*proc_weak(8)];

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
  1266.88,
  5747.55,
  14105.2,
  24398.3,
  37348,
  56610,
  94910.5,
  136904
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
% saveas(gcf, "StrongNew1.png");

figure;
plot(proc_weak, weak, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
% loglog(proc, weak_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');

hold on;
grid();

plot(proc_weak, [1, 1, 1, 1, 1, 1, 1, 1], '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
% legend(["Weak", "With Omp", "Idea"]);

set(gca, 'XScale', 'log');
xlim([1,128]);
ylim([0.5,1]);

xticks([1,2,4,8,16,32,64,128,256,512,1024]);
yticks([0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);

title("Efficiency of Weak Scaling");
hold off;
saveas(gcf, "WeakNew2.png");