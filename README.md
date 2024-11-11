# **Spatiotemporal Trajectory Prediction for Traffic Agents**

## **About**

This project implements several machine learning algorithms relative to traffic agent trajectory prediction.

The most recent model implemented is a Seq2Seq model with a PointNet model used to encode the environment. Though simple, it gives competitive results on the ArgoVerse dataset. 

![Screen visualized](visualize/animation.gif)

## Task List

#### Backbones:
- [x] MLP
- [x] Seq2Seq (LSTM, GRU, etc.)
- [ ] Transformer
- [ ] Graph Neural Network

#### Spatial Encodings:
- [x] PointNet
- [ ] PointNet++
- [ ] Graph Convolutional Network
- [x] ResNet
- [x] CNN
- [x] ConvLSTM

#### Loss Functions:
- [x] ADE/FDE Loss

#### Other Approaches:
- [ ] Social Pooling
- [ ] SocialGAN
- [ ] STGAT
- [ ] Trajectron
- [ ] TPNet

## Contact
Yishai Silver (ssilver@ucsd.edu)