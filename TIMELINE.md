10.27
Learn Pytorch mainly from https://pytorch.org/get-started/locally/

10.28
Learn DRN from https://www.cnblogs.com/wzyuan/p/9880342.html
Unsolved: 
- [ ] why 'labels' in Cifar10 is of size 4 instead of size 10
        
10.29
Learn more details of DRN from https://www.zhihu.com/question/52668301/answer/194998098 & https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
Unsolved: 
- [ ] what can model_zoo & model_urls do?
- [ ] what can 'expansion' do? (why is 4 but not 5, 6)
- [ ] what means 'inplace' in nn.Relu?
- [ ] what is kaiming normal?

10.30
How to import our dataset?
https://blog.csdn.net/woshicao11/article/details/78318156
might be useful: https://blog.csdn.net/Teeyohuang/article/details/82108203
Unsolved:
- [ ] reduce learning rate by factor of 10 every 3000 iterations
- [ ] what are the arguments in optimizer
- [ ] when optimizer is SGD, why would the loss sometimes suddenly larger than 1

11.2
start using mask & exploring FCRN
mask:https://blog.csdn.net/u014453898/article/details/80715121
use sampler to balance classes(labels) weights: 
https://stackoverflow.com/questions/61033726/valueerror-sampler-option-is-mutually-exclusive-with-shuffle-pytorch
knowledge of sampler: https://www.cnblogs.com/marsggbo/p/11308889.html

11.4
Use pretrained model: https://blog.csdn.net/TTdreamloong/article/details/84823705
Flip the image: https://blog.csdn.net/JNingWei/article/details/78753607