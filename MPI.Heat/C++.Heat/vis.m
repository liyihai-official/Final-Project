
speed = @(v) [v(1)/v(1), v(1)/v(2), v(1)/v(3), v(1)/v(4), v(1)/v(5), v(1)/v(6), v(1)/v(7)];

proc = [1, 2, 4, 8, 16, 32, 64];
speed_weak = @(v) [v(1)/v(1)*proc(1), v(1)/v(2)*proc(2), v(1)/v(3)*proc(3), v(1)/v(4)*proc(4), 
    v(1)/v(5)*proc(5), v(1)/v(6)*proc(6), v(1)/v(7)*proc(7)];

strong_1024 = [
  7.34037e+06,
  4.37383e+06,
  2.90165e+06,
  1.56588e+06,
  902118,
  522227,
  135117
];

strong_1024_omp = [
  6.60267e+06,
  2.49456e+06,
  752795,
  2.60923e+07
];


weak = [
  1300.82,
  5937.18,
  25125.1,
  134736
];

weak_omp = [
  1603.24,
  9252.92,
  38278.9,
  2.59875e+07
];

strong_1024 = speed(strong_1024);
strong_1024_omp = speed(strong_1024_omp);



weak = speed_weak(weak);
weak_omp = speed_weak(weak_omp);

figure;
hold on;
grid();
plot(proc, strong_1024, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
plot(proc, strong_1024_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
plot(proc, proc, '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
legend(["Strong", "With Omp", "Idea"])
title("Strong Scaling Speedup Ratio");
hold off;
saveas(gcf, "Strong.png");

figure;
hold on;
grid();
plot(proc, weak, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
plot(proc, weak_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
plot(proc, [1, 1, 1,1], '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
legend(["Weak", "With Omp", "Idea"]);
title("Efficiency of Weak Scaling");
hold off;
saveas(gcf, "Weak.png");