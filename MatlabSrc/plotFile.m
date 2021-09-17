A = load("MC_Project/data/verticesATLAS_kernel_7.dat");
B = load("MC_Project/data/indicesATLAS_kernel_7.dat");

no4=reshape(A,[6,length(A)/6])';
el4=reshape(B,[3,length(B)/3])'+1;
figure(4);
plotmesh(no4,el4);
hold on;


