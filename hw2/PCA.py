import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dimension', required=True, type=int, help='[int] dimension for the auto encoder')
args = ap.parse_args()

def main():
    import numpy as np
    from sklearn.decomposition import PCA

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
        
    train_file = np.array(train_file)   
    test_file = np.array(test_file)

    train_file_mean = train_file.mean(axis=0)
    train_file_center = train_file - train_file_mean
    test_file_mean = test_file.mean(axis=0)
    test_file_center = test_file - test_file_mean

    pca = PCA(n_components=args.dimension)
    pca.fit(train_file_center)

    encode = pca.transform(train_file_center)
    decode = pca.inverse_transform(encode)
    pred = decode + train_file_mean
    loss = np.mean((pred - train_file)**2)
    print("Ein: ", loss)

    encode = pca.transform(test_file_center)
    decode = pca.inverse_transform(encode)
    pred = decode + test_file_mean
    loss = np.mean((pred - test_file)**2)
    print("Eout: ", loss)


if __name__ == "__main__":
    main()
