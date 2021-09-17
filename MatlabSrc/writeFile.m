 
% Generate the Volume Matrix
dim=60;
[xi,yi,zi]=ndgrid(0.5:(dim-0.5),0.5:(dim-0.5),0.5:(dim-0.5));
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
mcxvol=ones(size(xi));
mcxvol(dist<625)=2;
mcxvol(dist<529)=3;
mcxvol(dist<100)=4;

%load('digimouse.mat');

vol = mcxvol;
%vol2 = digimouse;

% Reshape into the dimensions required
check = reshape(vol,[166*209*223,1]);

% Write into a file
fid = fopen('mouse.dat','w+');

fprintf(fid,"%f\n",check);

fclose(fid);

