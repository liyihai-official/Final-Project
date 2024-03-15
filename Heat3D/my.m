Nx_0 = 16;
Nx_1 = 16;
Nx_2 = 16;


x0 = zeros(Nx_1, Nx_2, Nx_0);

num_Frame = 20;

step = 0.001;
animated(1,1,1,num_Frame)=0;
char_f = '%f';
for m = 1:Nx_0+1
    char_f = strcat(char_f,' %f');
end

for i=1:num_Frame
    figure(1);
    figure_size = [100, 100, 400, 300]; % [left, bottom, width, height]

    fid=fopen(strcat('outputs/outputSeq',num2str(i),'.dat'),'r');
    
    for k=1:Nx_0
        x = fscanf(fid, char_f, [Nx_2, Nx_1]);
        x0(:, :, k) = transpose(x);
    end
    
    fclose(fid);
    [x1 y1 z1] = meshgrid(1:Nx_2, 1:Nx_1, 1:Nx_0);
    
    slice(x1,y1,z1,x0,[Nx_2/2, Nx_2], [Nx_1/2, Nx_1], [1, Nx_0/2+1]);
    
    colormap(jet);
    shading faceted;
    view([-42,22]);
    hc=colorbar;
    set(gcf, 'Position', figure_size);
    set(hc,'position',[0.932 0.3 0.02 0.6]);
    caxis([0 9]);

    xlim([1 Nx_2]);
    ylim([1 Nx_1]);
    zlim([1 Nx_0]);

    xticks(linspace(1,Nx_2,4)); % 设置特定刻度位置
    xticklabels(linspace(1,Nx_2,4)/Nx_2);

    yticks(linspace(1,Nx_1,4)); % 设置特定刻度位置
    yticklabels(linspace(1,Nx_1,4)/Nx_1);

    zticks(linspace(1,Nx_0,4)); % 设置特定刻度位置
    zticklabels(linspace(1,Nx_0,4)/Nx_0);
    
    pause(step);
    
    frame=getframe(figure(1));
    if (i==1)
      [animated, cmap]=rgb2ind(frame.cdata,256,'nodither');
    else
      animated(:,:,1,i)=rgb2ind(frame.cdata,cmap,'nodither');
    end
end


imwrite(animated,cmap,'Heat_3D.gif','DelayTime',step,'LoopCount',inf);