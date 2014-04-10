function Dideal = cpp_direct_ideal(kdata,kx,ky,ktimes,fmap,dideal_offs,lambda_space,lambda_time,lambda_lowrank,max_iter)


%% Inputs
%    kdata  - xres x  frames
%    kx     - xres x  frames
%    ky     - xres x  frames
%    ktimes - xres x  frames
%    fmap   - N x N x frames (N=reconstructed resolution)
%    dideal_offs - S x 1  (S = number of species)
%    lambda_space - scalar spatial regularization --> spatial smoothness
%    lambda_time - scalar temporal regularization --> temporal smoothness
%    max_iter  - number of iterations 

%%Output
%    Dideal - N x N x frames x species 




%% Export Kx
fid = fopen('Kx.dat','w');
fwrite(fid,kx,'float');
fclose(fid);

%% Export Ky
fid = fopen('Ky.dat','w');
fwrite(fid,ky,'float');
fclose(fid);

%% Export Kz
fid = fopen('Kt.dat','w');
fwrite(fid,ktimes,'float');
fclose(fid);

%% Export Kdata
fid = fopen('Kdata.dat','w');
raw(1:2:2*numel(kdata))=real(kdata(:));
raw(2:2:2*numel(kdata))=imag(kdata(:));
fwrite(fid,raw,'float');
fclose(fid);

%% Export Fieldmap
fid = fopen('FieldMap.dat','w');
if size(fmap,3)==1
    fmap = repmat(fmap,[1 1 size(kdata,2)]);
end
fwrite(fid,fmap,'float');
fclose(fid);

%%Run the Recon
rcn = '~kmjohnso/VIPR/RECON/unstable/c13_test/recon_binary';
rcn_sp = [];
for sp = 1:numel(dideal_offs)
    rcn_sp = [rcn_sp sprintf(' -f%d %f ',sp-1,dideal_offs(sp))];
end
c = ['!',rcn,rcn_sp,' -Nt ',num2str(size(kx,2)) ...
    ,' -Nr ',num2str(size(kx,1))...
    ,' -Ns ',num2str(numel(dideal_offs))...
    ,' -Nx ',num2str(size(fmap,1))...
    ,' -Ny ',num2str(size(fmap,2))...
    ,' -lambda_space ',num2str(lambda_space)...
    ,' -lambda_time ',num2str(lambda_time) ...
    ,' -max_iter ',num2str(max_iter)...
    ,' -lambda_lowrank ',num2str(lambda_lowrank)];
eval(c);

%% Import Results
for sp = 1:numel(dideal_offs)
    name = sprintf('Species_%d.dat',sp-1);
    fid = fopen(name);
    raw = fread(fid,'float');
    Dideal(:,:,:,sp) = reshape(raw(1:2:end)+1i*raw(2:2:end),[size(fmap,1) size(fmap,2) size(kx,2)]);
    fclose(fid); 
    %figure
    %montage(reshape(abs(Dideal(:,:,:,sp)),[size(Dideal,1) size(Dideal,2) 1 size(Dideal,3)]),'DisplayRange',[]);
    %colormap('jet');
    %title(num2str(dideal_offs(sp)));
    
end

eval('!rm FieldMap.dat Kdata.dat Kx.dat Ky.dat Kt.dat -f');



