% Generate animated gif for MPI Heat2D
% function generate_gif_heat2d
% 
% % Load dimensions
% file=textread('param','%s','delimiter','\n');
% % x_dim = width in common convention, into core of code = vertical axis
% x_dim=str2double(file(2));
% % y_dim = height in common convention, into core of code = horizontal axis
% y_dim=str2double(file(4));
% % Number of
% % frames : must be equal to number of output files
% numFrames=100;
% % Time step between 2 frames
% step=0.1;
% animated(1,1,1,numFrames)=0;
% Main loop on number of frames

Nx_0 = 100;
Nx_1 = 100;
numFrames=50;

step = 1/numFrames;
animated(1,1,1,numFrames)=0;
char_f = '%f';
for m = 1:Nx_0+1
    char_f = strcat(char_f,' %f');
end

for i=1:numFrames % Create meshgrid
    disp(i);
    [X,Y]=meshgrid(1:Nx_0,1:Nx_1);
    % Read all values
    Z=load(strcat('outputs4/outputPINN',num2str(i),'.dat'));
    % Surf plot
    figure(1);
    % Set figure size (adjust these values as needed)
    figure_size = [100, 100, 400, 300]; % [left, bottom, width, height]

    set(gcf, 'Position', figure_size);

    surf(X,Y,Z);
    % Parameters for surf plot
    colormap(jet);
    shading interp;
    view([0,0,1]);
    hc=colorbar;
    set(hc,'position',[0.932 0.3 0.02 0.6]);
    set(gcf, 'Color', 'white'); % 或者可以使用 'w'

    caxis([0 10]);
    xlim([1 Nx_0]);
    ylim([1 Nx_1]);
    xlabel('x domain');
    ylabel('y domain');
    zlabel('temperature');
    
    title(i*20);
    % xticks(linspace(0,Nx_0,4)); % 设置特定刻度位置
    % xticklabels(linspace(0,Nx_0,4)/Nx_0);
    % 
    % yticks(linspace(0,Nx_1,4)); % 设置特定刻度位置
    % yticklabels(linspace(0,Nx_1,4)/Nx_1);

    % Pause
    pause(step);
    % Get current frame from figure
    frame=getframe(figure(1));
    if (i==1)
        [animated, cmap]=rgb2ind(frame.cdata,256,'nodither');
    else
        animated(:,:,1,i)=rgb2ind(frame.cdata,cmap,'nodither');
    end
end
% Write final animated gif
imwrite(animated,cmap,'Heat_2D.gif','DelayTime',step,'LoopCount',inf);
% end
