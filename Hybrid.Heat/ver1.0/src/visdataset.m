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
scatter3(x, y, z, 2, U, 'filled');

% Add labels and title
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Scatter Plot of Generated Dataset');
colormap(jet);
% Add a colorbar to represent the values
colorbar;


% Adjust view angle (optional)
view(45, 25);
saveas(gcf, '../out/main2_dataset.png')


% fid = fopen('../build/grid.bin', 'rb');
% Row = fread(fid, 1, 'uint32');
% Col = fread(fid, 1, 'uint32');
% A = fread(fid, [Row * Col], 'float32'); 
% A = reshape(A, [Col, Row]);
% A = permute(A, [2,1]);

% fid = fopen('../build/pred.bin', 'rb');
% Row = fread(fid, 1, 'uint32');
% Col=1;
% U = fread(fid, [Row * Col], 'float32'); 
% U = reshape(U, [Col, Row]);
% U = permute(U, [2,1]);

% x = A(:, 1);
% y = A(:, 2);
% z = A(:, 3);

% % Create a 3D scatter plot
% scatter3(x, y, z, 36, U, 'filled');

% % Add labels and title
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% title('3D PINN Results Plot of Points with Values');
% colormap(jet);
% % Add a colorbar to represent the values
% colorbar;

% saveas(gcf, '../out/main2_3d.png')
% % Adjust view angle (optional)
% view(45, 25);