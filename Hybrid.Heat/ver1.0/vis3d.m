fid = fopen('build/test_3d.bin', 'rb');

Row = fread(fid, 1, 'uint32');
Col = fread(fid, 1, 'uint32');
Dep = fread(fid, 1, 'uint32');


[X,Y,Z]=meshgrid(1:Col, 1:Row, 1:Dep);


A = fread(fid, [Row * Col * Dep], 'float32'); 
A = reshape(A, [Dep, Col, Row]);

A = permute(A,[3,2,1]);

figure;
h = gcf;
title("TEST");
slice(X, Y, Z, A, [1 Row/2 Row], [Col],  [Dep/2 1]);
colormap(jet);

colorbar;
caxis([0 10]);
% pause(0.05);