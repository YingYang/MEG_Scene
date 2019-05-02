function [error] = CCA_prediction(X1, X2, r, train_id, test_id)

    %%
    % train_id, test_id must be integer indices, not true/false
    n_train = length(train_id);
    n_test = length(test_id);
    
    Xtrain = {X1(train_id,:), X2(train_id,:)};
    Xtest = {X1(test_id,:), X2(test_id,:)};
    [W1, W2] = canoncorr(Xtrain{1}, Xtrain{2});
    W = {W1, W2};
    Proj = cell(1,2);
    for j = 1:2
        Proj{j} = Xtrain{j}*W{j};
    end
    error = zeros(1,2);
    for j = 1:2
        id_to_predict = setdiff([1,2],j);
        linear_mapping = zeros(r,2);
        for i = 1:r
            tmpB = regress(Proj{id_to_predict}(:,i), ...
                [Proj{j}(:,i), ones(n_train,1)]);
            linear_mapping(i,:) = tmpB';
        end
        % predicting the testing data
        % ==== predict the CCA projections
        Proj_test_j = Xtest{j}*W{j}(:,1:r);
        Proj_test_to_predict = zeros(n_test,r);
        for i = 1:r
            Proj_test_to_predict(:,i)=Proj_test_j(:,i)*linear_mapping(i,1)...
                                     +linear_mapping(i,2);
        end
        % ==== reconstruct the data
        [tmpu, tmpd, tmpv] = svd(W{id_to_predict}(:,1:r), 'econ');
        if size(tmpd,2)>1
            %disp(i);
            disp(size(tmpd));
            tmpd = diag(tmpd);
        end
        tmpd_inv = 1.0./tmpd(1:i);
        tmp_hat = Proj_test_to_predict*tmpv*diag(tmpd_inv)*tmpu';
        error(j) = sum(sum((Xtest{id_to_predict} - tmp_hat).^2))...
                   /sum(sum(Xtest{id_to_predict}.^2));
    end
end
        
        
        
        
        
            
            
            
    
