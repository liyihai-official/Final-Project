% 
% fid = fopen(strcat('mat_1000', '.bin'), 'rb');
% 
% N = fread(fid, 1, 'uint64');
% A = fread(fid, [N], 'double');
% 
% A = A*ones([1,N]);
% 
% [X,Y] = meshgrid(1:N, 1:N);
% 
% figure(1);
% title('1D Scatter Plot');
% imagesc(A');
% colormap(jet)
% colorbar;

animated(1,1,1,9)=0;

for i=0:40
    fid = fopen(strcat('mat_', num2str(i*100), '.bin'), 'rb');

    N = fread(fid, 1, 'uint64');
    A = fread(fid, [N], 'double');

    A = A*ones([1,N]);

    [X,Y] = meshgrid(1:N, 1:N);

    figure(1);
    title('1D Scatter Plot');
    imagesc(A');
    colormap(jet)
    colorbar;
    caxis([-1 1]);
    pause(0.1);

    frame=getframe(figure(1));
    if (i==0)
        [animated, cmap]=rgb2ind(frame.cdata,256,'nodither');
    else
        animated(:,:,1,i)=rgb2ind(frame.cdata,cmap,'nodither');
    end
end

imwrite(animated,cmap,'Heat_1D.gif','DelayTime',0.1,'LoopCount',inf);