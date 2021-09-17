% Generate Volume matrix
dim=500;
[xi,yi,zi]=ndgrid(0.5:(dim-0.5),0.5:(dim-0.5),0.5:(dim-0.5));
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
mcxvol=ones(size(xi));
mcxvol(dist<625)=2;
mcxvol(dist<529)=3;
mcxvol(dist<100)=4;

load('digimouse.mat');

vol = mcxvol;
%vol = digimouse;

% View each label of the volume matrix

labels1=sort(unique(vol(:)));
% labels(labels==0)=[];

if(length(labels1)>255)
    error('MCX currently supports up to 255 labels for this function');
end

%% loop over unique labels in ascending order

iso=struct('vertices',[],'faces',[]);
tic;
for i=1:length(labels1)
    % convert each label into a binary mask, smooth it, then extract the
    % isosurface using marching cube algorithm (matlab builtin)
    
        volsmooth=smooth3(vol==1,'g',7,1);
        %volsmooth=(vol==i);
    [xi,yi,zi]=ndgrid(1:size(volsmooth,1),1:size(volsmooth,2),1:size(volsmooth,3));
    fv0=isosurface(xi,yi,zi,volsmooth,0.5);
    %figure(i);
    plotmesh(fv0.vertices,fv0.faces);
    %hold on
end
toc;
