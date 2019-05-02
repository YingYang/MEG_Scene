clear;
clc;

%% load data and SVD
mode_id = 5;
mode = {'Layer1_Layer6', 'Local_contrast_Layer6', ...
    'Layer1_Layer6_nocontrast160', 'Layer1_Layer7_nocontrast160',...
    'Layer1_Layer7'};

X = cell(2,1);
U = cell(2,1);
D = cell(2,1);
UD =cell(2,1);
V = cell(2,1);

regressor_dir = '/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/';
feature_dir = '/home/ying/Dropbox/Scene_MEG_EEG/Features/';

data1 = load('/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/All_images_Layer1_contrast.mat');
contrast_noaspect = data1.contrast_noaspect;
contrast_noaspect = bsxfun(@minus, contrast_noaspect, mean(contrast_noaspect,1));
[u0,d0,v0] = svd(data1.contrast_noaspect, 'econ');
d0 = diag(d0);
var_explained = cumsum(d0.^2)./sum(d0.^2);
figure; plot(var_explained, '-+');

ndim = 160; % 90% variance?
n_all_im = size(u0,1);
u0 = u0(:,1:ndim);
%% most important step, always check
proj = eye(n_all_im)- u0* inv(u0'*u0) * u0';
model_name = 'AlexNet';

% the features should already be zero mean, because the aspect ratio was removed
%% I did the projection in python, so 
if mode_id == 1 || mode_id == 5
    if mode_id == 1
        layer = {'conv1','fc6'};
    else
        layer = {'conv1','fc7'};
    end
    
    for i = 1:2
        data = load( sprintf('%s_Features/%s_%s_noaspect_all_images.mat',model_name, model_name, layer{i}));
        data1 = bsxfun(@minus, data.data, mean(data.data,1));
        X{i} = data1;
    end
elseif mode_id == 2
    layer = { 'localcontrast','fc6'};
    data = load( sprintf('%s_Features/%s_%s_noaspect_all_images.mat',model_name, model_name, layer{2}));
    data1 = bsxfun(@minus, data.data, mean(data.data,1));
    X{2} = data1;
    X{1} = contrast_noaspect;
end

if (mode_id == 1 || mode_id == 2) || mode_id == 5
    layer1 = layer;
    for i = 1:2
        [U{i}, D{i}, V{i}] = svd(X{i}, 'econ');
        UD{i} = U{i}*D{i};
    end
end

if mode_id == 3 || mode_id == 4
    if mode_id == 3
        layer = {'conv1','fc6'};
    else
        layer = {'conv1','fc7'};
    end

    for i = 1:2
        %data = load( sprintf('%s_Features/%s_%s_noaspect_all_images.mat', model_name, model_name, layer{i}));
        %X{i} = proj*data.data;
        data = load(sprintf('%sAlexNet_%s_no_aspect_no_contrast%d_all_im_PCA.mat', regressor_dir, layer{i}, ndim));
        %X{i} = data.data;
        UD{i} = data.X*diag(data.D);     
    end
    layer1 = {sprintf('%s_nc%d', layer{1}, ndim),...
             sprintf('%s_nc%d', layer{2}, ndim)};
end
    
% % debug:
% mat_dict = load(sprintf('%sAlexNet_%s_no_aspect_no_contrast%d_PCA_all_im.mat', ...
%             regressor_dir, 'fc6', 160));
% % more or less match, the later components were anti-correlated
% tmpU = mat_dict.X;
% disp(norm(tmpU(:,1:362)-U{2}(:,1:362))./norm(tmpU));
% disp(norm(mat_dict.data-X{2})./norm(X{2}))

%% CCA
train_start = 363;
r = 362;
% CCA
Xtrain = {UD{1}(train_start:end,1:r), UD{2}(train_start:end,1:r)};
[W1, W2, corrval, Proj1, Proj2, stats] = canoncorr(Xtrain{1}, Xtrain{2});
Proj1 = Xtrain{1}*W1;
Proj2 = Xtrain{2}*W2;

%% Verifying the computation of CCA
if 0
    % try computing it myself using the cross covariance, see wikipedia
    tmp = cov([Xtrain{1}, Xtrain{2}]);
    Sigma_xx = tmp(1:r,1:r);
    Sigma_yy = tmp((r+1):end,(r+1):end);
    Sigma_xy = tmp(1:r,(r+1):end);
    % eigen value decom needed
    [tmpu, tmpd, tmpv] = svd(Sigma_xx);
    tmpu = (tmpu+tmpv)/2;
    Sigma_xx_inv_half = tmpu * diag(1./diag(sqrt(tmpd)))*tmpu';
    Sigma = Sigma_xx_inv_half*((Sigma_xy/Sigma_yy)*Sigma_xy')*Sigma_xx_inv_half;
    Sigma = (Sigma+Sigma')/2;
    [tmp_C,corr] = eigs(Sigma);
    W11 = Sigma_xx_inv_half*tmp_C;
    W22 = Sigma_yy\(Sigma_xy')*Sigma_xx_inv_half*tmp_C;
    i0 = 5;
    subplot(1,2,1); plot(W11(:,i0), W1(:,i0),'.'); grid on;
    subplot(1,2,2); plot(W22(:,i0), W2(:,i0),'.'); grid on;
    % verified, correlated with W1
end

%% variance explained by the projections
error = zeros(r,2);
W = {W1, W2};
Proj = {Proj1, Proj2};
for i = 1:r
    % regress the original data on the projections
    for j = 1:2
        % compute the pseudoinverse 
        [tmpu, tmpd, tmpv] = svd(W{j}(:,1:i), 'econ');
        if size(tmpd,2)>1
            %disp(i);
            disp(size(tmpd));
            tmpd = diag(tmpd);
        end
        tmpd_inv = 1.0./tmpd(1:i);
        % this is equivalent to using pseudoinverse
        tmp_hat = Proj{j}(:,1:i)*tmpv*diag(tmpd_inv)*tmpu';
        %tmp_hat = Proj{j}(:,1:i)* pinv(W{j}(:,1:i));
        %disp(norm(tmp_hat1-tmp_hat)/norm(tmp_hat1)); 
        error(i,j) = sum(sum((Xtrain{j} - tmp_hat).^2))/sum(sum(Xtrain{j}.^2));
    end
end

%% cross validation prediction
rseq = [1:1:362];
n_r = length(rseq);
X1 = Xtrain{1}; X2 = Xtrain{2};
% 6 fold cross validation
n_fold = 6;
n_im0 = size(X1,1);
linear_id = 1:n_im0;
id = reshape(linear_id,6,floor(n_im0/6));
error_rseq = zeros(n_r,2, n_fold);
for i = 1:n_r
    r = rseq(i);
    tmp_error = zeros(1,2);
    for j = 1:n_fold
        tmp_test_id = id(j,:);
        tmp_train_id = setdiff(linear_id, tmp_test_id);
        % cross validation
        error_rseq(i,:,j) = CCA_prediction(X1,X2,r, tmp_train_id, tmp_test_id);
    end
end
error_rseq_mean = squeeze(mean(error_rseq, 3));
h = figure; plot(error_rseq_mean, 'o-'); legend(layer1);
saveas(h,sprintf('CCA_cv_prediction_%s_%s.eps',layer1{1},layer1{2})); grid;
save(sprintf('CCA_cv_error_%s_%s.mat',layer1{1},layer1{2}),'error_rseq','rseq', 'error_rseq_mean');


%p_c = 362;
p_c = 20;
figure; plot(1:p_c,error_rseq_mean(1:p_c,:)*100, '-x', 'LineWidth',1);
set(gca, 'FontSize', 20)
xlabel('p_c' );
ylabel('error %');
if mode_id == 1 || mode_id == 3
    legend('Layer 1 -> 6','Layer 6 -> 1');
else
    legend('Layer 1 -> 7','Layer 7 -> 1');
end

hold on;
plot([0,p_c],[100,100], 'k')

%legend boxoff
%saveas(gca, sprintf('CCA_cv_prediction_%s.eps', mode{mode_id}));

% CV
% less than 3 dimensions


%% visualization of the weights
% weights1 = V{1}(:,1:r)*W1;
% im_id = 363;
% 
% for j = 1:5
%     % visualize the first component
%     weights1_3d = reshape(weights1(:,j), [55,55,96] );
%     n1 = 10; n2 = 10;
%     n0 = 60;
%     weights1_3d_expansion = NaN(n1*n0,n2*n0);
%     for i = 1:96
%         i1 = floor((i-1)/n1)+1;
%         i2 = mod(i-1,n2)+1;
%         tmp = weights1_3d(:,:,i);
%         %tmp = (tmp-min(tmp(:)))/(max(tmp(:))-min(tmp(:)));
%         weights1_3d_expansion(((i1-1)*n0+1): ((i1-1)*n0+55),((i2-1)*n0+1): ((i2-1)*n0+55)) = tmp;
%     end
%     h = imagesc(weights1_3d_expansion); colorbar();
%     saveas(h, sprintf('fig/%s_%s_cca_conv1_r%02d.eps', layer{1}, layer{2}, j));
%     close;
%         
% 
%     % first images
%     im1 = X{1}(im_id,:)';
%     im1_3d = reshape(im1, [55,55,96] );
%     im1_3d_expansion = NaN(n1*n0,n2*n0);
%     for i = 1:96
%         i1 = floor((i-1)/n1)+1;
%         i2 = mod(i-1,n2)+1;
%         tmp = im1_3d(:,:,i)'.*weights1_3d(:,:,i)';
%         tmp = (tmp-min(tmp(:)))/(max(tmp(:))-min(tmp(:)));
%         im1_3d_expansion(((i1-1)*n0+1): ((i1-1)*n0+55),((i2-1)*n0+1): ((i2-1)*n0+55)) = tmp ;
%     end
%     h = imagesc(im1_3d_expansion);colorbar; 
%     saveas(h, sprintf('fig/%s_%s_cca_conv1_r%02d_Im%d.eps', layer{1}, layer{2}, j, im_id));
%     close;
% 
%    
%     im1 = X{1}(im_id,:)';
%     im1_3d = reshape(im1, [55,55,96] );
%     im1_3d_expansion = NaN(n1*n0,n2*n0);
%     for i = 1:96
%         i1 = floor((i-1)/n1)+1;
%         i2 = mod(i-1,n2)+1;
%         tmp = im1_3d(:,:,i)';
%         tmp = (tmp-min(tmp(:)))/(max(tmp(:))-min(tmp(:)));
%         im1_3d_expansion(((i1-1)*n0+1): ((i1-1)*n0+55),((i2-1)*n0+1): ((i2-1)*n0+55)) = tmp;
%     end
%     h = imagesc(im1_3d_expansion);colorbar;
%     saveas(h, sprintf('fig/Conv1_Im%d.png', im_id));
%     close;
% end

%%
% compute the first 10 projections of the testing cca
cca_common = 25;
test_start = 1;
test_end = 362;
Xtest = {UD{1}(test_start:test_end,1:r), UD{2}(test_start:test_end,1:r)};
Proj1_test = Xtest{1}*W1;
Proj2_test = Xtest{2}*W2;

% visulize the correlation
figure;
for i = 1:cca_common
    subplot(5,5,i)
    plot(Proj1_test(:,i), Proj2_test(:,i),'+');
    tmp_corr = corr(Proj1_test(:,i),Proj2_test(:,i));
    title(sprintf('%1.2f', tmp_corr));
end

Proj_test = {Proj1_test, Proj2_test};


%% saving mat files
% how to know how much variance is explained?
% the first 50 dimensions were significant
save(sprintf('CCA_%s_%s.mat', layer1{1}, layer1{2}), ...
    'W1','W2','corrval','Proj1','Proj2','stats');
save(sprintf('CCA_%s_%s_SVD_UD.mat', layer1{1}, layer1{2}), ...
'U','D','UD');
save(sprintf('CCA_%s_%s_CCA_test.mat', layer1{1}, layer1{2}), ...
    'Proj_test', 'Xtest');

save(sprintf('CCA_%s_%s_SVD_V_v7.3.mat', layer1{1}, layer1{2}), 'V', '-v7.3');

    