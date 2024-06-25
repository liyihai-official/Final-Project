
n=93;
animated(1,1,1,n)=0;

for i=0:n
    fid = fopen(strcat('mat_', num2str(i*1000), '.bin'), 'rb');
    
    rows = fread(fid, 1, 'uint64');
    cols = fread(fid, 1, 'uint64');
    
    A = fread(fid, [rows*cols], 'double');
    A = reshape(A, [cols, rows]);
    
    A = permute(A, [1, 2]);
    [X, Y] = meshgrid(1:rows, 1:cols);
    
    figure(1);
    h = gcf;
    title('2D Scatter Plot');
    imagesc(A');
    colormap(jet);
    shading interp;
    hc=colorbar;
    caxis([0 10]);
    pause(0.005);
    % set(h, 'Position', [100, 100, 200, 160]);

    frame=getframe(figure(1));
    if (i==0)
        [animated, cmap]=rgb2ind(frame.cdata,256,'nodither');
    else
        animated(:,:,1,i)=rgb2ind(frame.cdata,cmap,'nodither');
    end
end

imwrite(animated,cmap,'Heat_2D.gif','DelayTime',0.1,'LoopCount',inf);