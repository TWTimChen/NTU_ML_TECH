# For question 11-14
## Required files
* zip.test
* zip.train
* AutoEncoder.py
* PCA.py
## AutoEncoder.py
### usage 
* -d {2, ,4 ,8 ,16, 32, 64, 128}    
    * dimension for the encode space
* -c {1 for true, 0 for false}      
    * with or without constraint 
* -e {default=5000}                 
    * number of epoch 
* -l {default=0.1}                  
    * number of learning rate
### output
* Ten lines for the loss function values
* Ein for the trained model
* Eout for the trained model
### To generate the result
* question 11-12
> for i in 2 4 8 16 32 64 128; do python AutoEncider.py -d $i -c 0; done
* question 13-14
> for i in 2 4 8 16 32 64 128; do python AutoEncider.py -d $i -c 1; done

# For question 15-16
## PCA.py
### usage 
* -d {2, ,4 ,8 ,16, 32, 64, 128}    
    * number of PC for the linear tranform
### output
* Ten lines for the loss function values
* Ein for the trained model
* Eout for the trained model
### To generate the result
* question 15-16
> for i in 2 4 8 16 32 64 128; do python AutoEncider.py -d $i; done