# Zero Shot Image Segmentation using CLIP

Image segmentation is the process of dividing an image into multiple regions, where each region represents a specific object class or possesses certain characteristics. The goal of this task is to perform image segmentation for arbitrary objects based on a text prompt using a subsegment of the Phrasecut dataset. Instead of segmenting images based on a fixed set of classes or static characteristics, we will leverage OpenAI's CLIP to associate text prompts with image segmentation.

## Problem Statement

In this project, we aim to carry out Zero Shot Image Segmentation using OpenAI's CLIP. CLIP is a neural network trained to associate similar text and images while contrasting dissimilar ones through contrastive learning. By leveraging CLIP's capability to produce similar vector representations for related text and images, we can create a model that accepts a text prompt and an image as input. The model will generate an embedding for the text prompt and the input image, which will be used to train a decoder to produce a binary segmentation map.

## Project Tasks

1. **Dataset**: I have used a subsegment of the Phrasecut dataset for training and evaluation. The dataset contains the segmented output in the form of polygons. Firstly, I converted those polygons into binary image masks.

2. **CLIP Encoder**: I have created two model architecture each of which utilize CLIP's encoder form the OpenAI's API to generate embeddings for both the text prompt and the image. A decoder will be trained on top of these embeddings to generate a binary segmentation map.

3. **Transformer Decoder**:  Firstly, I have used the transformers-based Decoder, which is specifiaclly designed for the task of Binary Segmentation based upon the CLIP encoder. I have used Hugging Face's Transformer library to access the decoder model.

4. **Conv Decoder**: Next, I have used a custom architecture for the decoder inspired by the U-Net. Here, I have used the stacks of upsampling and deconvolutional layers. The output from the Encoder is given at various levels to the achitecture, and the information is then passed on to the decoder, which then produces the output of the desired shape.

5. **Evaluation Metrics**: We will evaluate the performance of the model using various evaluation metrics such as accuracy, Dice score, and intersection over union (IoU). By comparing the results obtained from different models with varying complexities and loss functions (BCE and Dice loss), we can analyze their effectiveness in producing accurate segmentation maps.


## Conclusion

The project focuses on achieving Zero Shot Image Segmentation using OpenAI's CLIP. By training a model that combines CLIP's text-image association capabilities with a decoder for generating segmentation maps, we aim to segment images based on arbitrary objects using text prompts. The project involves training models of different complexities and loss functions and evaluating their performance using various metrics. It can be concluded from the findings that the BCE is not the correct metric to evaluate the loss in an image segmentaion task. Also, we see that the U-Net based architecture fails to give proper performance as, probably, because the the CLIP encoder is modelled to give its output to a transformer based architecture which can incorporate self-attention mechanism to give the required output.

### Acknowledgement 
Special thanks to [SAiDL](https://www.saidl.in/) for providing the problem statement.
