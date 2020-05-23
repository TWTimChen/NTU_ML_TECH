import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dimension', required=True, type=int, help='[int] dimension for the auto encoder')
ap.add_argument('-c', '--constraint', required=True, type=int, help='[int] add constraint to the encoder')
ap.add_argument('-l', '--learn', required=False, type=float, default=0.1, help='[float] learning rate')
ap.add_argument('-e', '--epoch', required=False, type=int, default=5000, help='[int] epoch ')
args = ap.parse_args()

def main():
    import numpy as np
    import torch

    #################################################################################
    #####################               Load data set               #################
    #################################################################################

    train_file_path = "./zip.train"
    test_file_path = "./zip.test"

    train_file = []
    test_file = []

    with open(train_file_path, "r") as f:
        lines = f.readlines()
        train_file = [[float(numStr) for numStr in line.strip(" \n").split(" ")] for line in lines]

    with open(test_file_path, "r") as f:
        lines = f.readlines()
        test_file = [[float(numStr) for numStr in line.strip(" \n").split(" ")] for line in lines]
        
    train_file = torch.FloatTensor(train_file)   
    test_file = torch.FloatTensor(test_file)

    train_sub = train_file[:,-256:]
    test_sub = test_file[:,-256:]

    #######################################################################################
    #####################            Set hyper Parameters           #######################
    #######################################################################################

    dtype = torch.float
    EPOCH = args.epoch
    RATE = args.learn
    D_in = 256
    D_code = args.dimension

    x = train_sub
    y = train_sub

    x_test = test_sub
    y_test = test_sub
    b_test = torch.ones(test_sub.shape[0],1)

    BATCH_SIZE = x.shape[0]
    b = torch.ones(BATCH_SIZE,1)

    #######################################################################################
    #####################         Initialize layers of weight       #######################
    #######################################################################################

    Encode_in, Encode_out, Decode_in, Decode_out = D_in, D_code, D_code, D_in

    U = (6/(1+D_in+D_code))**(1/2)

    w1 = torch.rand(Encode_in, Encode_out, dtype=dtype)
    w1_b = torch.rand(1, Encode_out, dtype=dtype)
    w2 = torch.rand(Decode_in, Decode_out, dtype=dtype)
    w2_b = torch.rand(1, Decode_out, dtype=dtype)

    if args.constraint == 1:
        w1 = (w1*2-1)*U
        w2 = w1.clone().t()
    elif args.constraint == 0:
        w1 = (w1*2-1)*U
        w2 = (w2*2-1)*U
    
    w1_b = (w1_b*2-1)*U
    w2_b = (w2_b*2-1)*U

    grad_y_pred = torch.zeros(x.shape)
    grad_w2 = torch.zeros(w2.shape)
    grad_w2_b = torch.zeros(w2_b.shape)
    grad_h_tanh = torch.zeros(w2.shape)
    grad_h = torch.zeros(w2.shape)
    grad_w1 = torch.zeros(w1.shape)
    grad_w1_b = torch.zeros(w1_b.shape)

    #######################################################################################
    #####################   Execate iteration of gradient descent   #######################
    #######################################################################################

    for epoch in range(EPOCH):
        h = x.mm(w1) + b.mm(w1_b)
        h_tanh = h.tanh()
        y_pred  = h_tanh.mm(w2) + b.mm(w2_b)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum(axis=1).div(D_in).mean().item()
        if epoch % (EPOCH/10) == (EPOCH/10 - 1):
            print("Epoch: ", epoch+1, "\t", loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y) / D_in
        grad_w2 = h_tanh.t().mm(grad_y_pred)
        grad_w2_b = b.t().mm(grad_y_pred)
        grad_h_tanh = grad_y_pred.mm(w2.t())
        grad_h = grad_h_tanh*(1-h_tanh**2)
        grad_w1 = x.t().mm(grad_h)
        grad_w1_b = b.t().mm(grad_h)

        # Update weights using gradient descent
        if args.constraint == 1:
            wSum = grad_w1+grad_w2.t()
            w1 -= RATE * wSum / BATCH_SIZE
            w2 -= RATE * wSum.t() / BATCH_SIZE
        elif args.constraint == 0:
            w1 -= RATE * grad_w1 / BATCH_SIZE
            w2 -= RATE * grad_w2 / BATCH_SIZE
       
        w1_b -= RATE * grad_w1_b / BATCH_SIZE
        w2_b -= RATE * grad_w2_b / BATCH_SIZE

    print("Ein: ", loss)

    h = x_test.mm(w1) + b_test.mm(w1_b)
    h_tanh = h.tanh()
    y_pred  = h_tanh.mm(w2) + b_test.mm(w2_b)
    loss = (y_pred - y_test).pow(2).sum(axis=1).div(D_in).mean().item()
    print("Eout: ", loss)


if __name__ == "__main__":
    main()
