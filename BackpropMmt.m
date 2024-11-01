function [W1, W2] = BackpropMmt(W1, W2, X, D)
    alpha = 0.9;
    beta = 0.1;
    mmt1 = zeros(size(W1));
    mmt2 = zeros(size(W2));
    N = 4;
    for k = 1:N
        x = X(k, :)';
        d = D(k);

        v1 = W1 * x;
        y1 = Sigmoid(v1);
        v = W2 * y1;
        y = Sigmoid(v)
        
        e = d - y;
        delta = y .* (1-y) .* e;  %輸出層delta
        
        e1 = W2' * delta;
        delta1 = y1 .* (1-y1) .* e1; %隱藏層delta
        
        dW1 = alpha * delta1 * x'; %隱藏層權重更新量(未加入動量)
        mmt1 = dW1 + beta * mmt1; %隱藏層權重更新量(加入動量)
        W1 = W1 + mmt1;

        dW2 = alpha * delta * y1'; %輸出層權重更新量(未加入動量)
        mmt2 = dW2 + beta * mmt2; %輸出層權重更新量(加入動量)
        W2 = W2 + mmt2;
    end
end

