# PolygonColorizingUNet
Simple UNet model that fills a polygon with a chosen color.
## If you don't have the patience to clone the repo locally : 
Find a ready to use colab notebook [here](https://colab.research.google.com/drive/1hDgb_ZRUHCauG4U2jyBpYzX2vCfikuIs?usp=sharing).
## To train the model :
1. Clone this repository.
2. Find the dataset [here.](https://drive.google.com/open?id=1QXLgo3ZfQPorGwhYVmZUEWO_sU3i1pHM)
3. Place the downloaded training and validation data in a single folder named data in the root project directory.
4. Then simply run the training using this [notebook](https://github.com/harish-jhr/PolygonColorizingUNet/blob/main/notebooks/train.ipynb).
## To infer using pre-trained weights :
1. Clone this repository.
2. Find pretrained weights [here](https://huggingface.co/Harish-JHR/PolygonColorizingUNet).
3. Simply run infernce usning this [notebook](https://github.com/harish-jhr/PolygonColorizingUNet/blob/main/notebooks/inference.ipynb).


### Visit the wandb for a short report and training dynamics :
Find the report [here](https://wandb.ai/harishjrao615-iiser-bhopal/CustomUNet/reports/PolygonColorizingUNet--VmlldzoxMzg3NDk2NQ?accessToken=4vf90ryjk6r3b281efzy9yc2v6xikyhbmf6c4tdiiliihxrsm4uetg7oupuwb93i).
