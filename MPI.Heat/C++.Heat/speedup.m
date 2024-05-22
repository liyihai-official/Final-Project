
speed = @(v) [v(1)/v(1), v(1)/v(2), v(1)/v(3), v(1)/v(4)];

proc = [1, 4, 16 ,64];
speed_weak = @(v) [v(1)/v(1)*proc(1), v(1)/v(2)*proc(2), v(1)/v(3)*proc(3), v(1)/v(4)*proc(4)];

strong_1024 = [
  7.31119e+06,
  2.94504e+06,
  891509,
  134817
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
savefig("Strong.fig")

figure;
hold on;
grid();
plot(proc, weak, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'r');
plot(proc, weak_omp, '--gs', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'b');
plot(proc, [1, 1, 1,1], '-o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
legend(["Weak", "With Omp", "Idea"]);
title("Efficiency of Weak Scaling");
hold off;
savefig("Weak.fig")