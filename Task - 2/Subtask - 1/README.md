# Massachusetts Roads Masking 

In this task, I had to make a deep learning architecture to mask the road on a given image.

## Dataset

The dataset contains images of massachusetts of size (1500,1500) and paired with its corressponding images of road masks. The dataset was not perfect - as it had images of the road chopped but the mask of that area remain same. Moreover, Though the no of images is not very large, due to size of the images it was very difficult to work with. 

Therefore the dataset was resized to (256,256) and was normalised first to \[0,1\] and then to standardised. Moreover, the area whose images were missing were removed from the road true mask.

## Model Architecture

I have tried there different variant of u net :
i) using global average to give better view in residual network and max pooling to deal with the size of the output.
ii) using padding to keep the shape of tensor same after convolution making it more easy to deal with output and residual connections.
iii) using more convolution layer and convolution layer in residual network

## Training

I have used Adam and AdamW optimiser where ever it performed well. And the lr and other weight are mentioned. Mostly all model are trained for 200 epochs.


## Loss funtion and Losses
<ol>
<li> Diceloss 

It works pretty well and have very sharp prediction. In the final model,
<ul>
  <li>Accuracy: 0.9446909427642822
  <li>Recall: 0.4517928659915924
  <li>Presicion: 0.5604997277259827
  <li>IOU: 0.29880213737487793
  <li>diceloss: 0.4601195454597473
</ul>
Accuracy here is not a great evaluator as most of the area is not road. And this can shown in recall and presicion, that the most can only predict road only nearly 50% percentage of the time. And you can see diceloss and iou also show how much area it can predict.

<li> BCELoss

  Due to huge class imbalance, it preforms not very well either.
<ul>
  <li>Accuracy: 0.9529767632484436
  <li>Recall: 0.6953473091125488
  <li>Presicion: 0.08563575893640518
  <li>IOU: 0.13610906898975372
  <li>diceloss: 0.23960565030574799
</ul>
As you can the precision is very low showing that it not very good metric.

<li> Diceloss and BCELoss

  A small percentage of bceloss is added to diceloss, we get best of both loss.
<ul>
  <li>Accuracy: 0.9500315189361572
  <li>Recall: 0.4947252869606018
  <li>Presicion: 0.5395021438598633
  <li>IOU: 0.3122359812259674
  <li>diceloss: 0.4758839011192322
</ul>

As you can see each improved alot, making it better than previous.

<li> Focal Loss
It is a great loss func to manage class imabalance but it very hard to tune the hyperparameters.
<ul>
  <li>Accuracy: 0.7066519856452942
  <li>Recall: 0.1350572556257248
  <li>Presicion: 0.9137591123580933
  <li>IOU: 0.05890076979994774
  <li>diceloss: 0.11124889552593231
</ul>

it great precision but cannot mask all the road very neatly.

<li>DoubleDiceloss

  It is similar to (3) but with another dice loss for non-road area.

<ul>
  <li>Accuracy: 0.946321427822113
  <li>Recall: 0.46392542123794556
  <li>Presicion: 0.5568877458572388
  <li>IOU: 0.30534330010414124
  <li>diceloss: 0.4678359925746918
  </ul>

It is very similar to the third one but computational lighter than (iii).
</ol>

## Inference
The model can clearly catch the road having significant width but ones that are thin are very hard to predict as the images are scaled down.
Moreover, The model are clearly noisy as the model cannot use the connected-ness of the roads which could be solved by adding attention to the residual but as the connected-ness is a local property, the convolution should be able to understand it.

And, double dice loss performed the best overall.

## How other model performed

I have used the pytorch version of DeepLabV3 but it performed poorly may due to the scale of the roads and thus I have not added that in this file.
  
