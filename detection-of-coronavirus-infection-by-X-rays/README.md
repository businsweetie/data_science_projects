# Detection of coronavirus infection by X-rays
The rapid spread of COVID-19 infection has had a huge impact and caused irreparable damage to the lives of many people. To diagnose and detect such infections, computed tomography and radiography are currently used, but even if the necessary devices and medicines are provided, it is not always possible to accurately and correctly diagnose, due to the large flow of patients. The solution to this problem can be fast-growing neural networks that can solve problems related to the analysis and classification of medical images.

## Content
[Description and processing of the studied data](#description)     
[Data Balancing](#balancing)   
[Selection of hyperparameters](#hyperparameters)  
[Model training](#training)   
[Evaluating results](#evaluation)   
[Visualization of results using Grad-CAM](#vizualization)   
[Conclusion](#сonclusion)

<a name="description"><h2>Description and processing of the studied data</h2></a>
In order to review the work of neural networks in the task of recognizing coronavirus infection, a set of data from the site was used [kaggle.com](www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) about 18868 X-ray images of patients, containing [4 classes](#class_img):
- covid, 3615 images (a);
- pneumonia, 1345 images (b);
- lung opacity, 3716 images (c);
- normal, 10192 images (d).

<a name="class_img">![Classes](https://github.com/businsweetie/data_science_projects/blob/main/detection-of-coronavirus-infection-by-X-rays/pic/Dataset.png)</a>

All images have been transformed to an extension $224 \times 224$ the pixel where augmentation (mapping) was applied to the data..

The data set under study was [divided](#sample_table) into training, test, and validation samples.

<a name="sample_table"></a>
Sample type | Covid | Pneumonia | Lung opacity | Normal | Total images
:----------:|:-----:|:---------:|:------------:|:------:|:-----------:
Source      | 3615  | 1345      | 3716         | 10192  | 18868 
Train       | 3315  | 1045      | 3416         | 9892   | 17668
Valid       | 150   | 150       | 150          | 150    | 600
Test        | 150   | 150       | 150          | 150    | 600


<a name="balancing"><h2>Data Balancing</h2></a>
The data set under consideration is [unbalanced](#balancing_img), so the WeightedRandomSampler method was used for correct training.

<a name="balancing_img">![Balancing](https://github.com/businsweetie/data_science_projects/blob/main/detection-of-coronavirus-infection-by-X-rays/pic/DistrData.png)</a>

To use this method, you need to:
- get the value of the number of images in each class;
- calculate weights for each class $\frac{1}{n_i}$, where  $n_i$ is number of images in the classе $i$;
- assign a [corresponding weight factor](#weight_table) to each image in the class.

<a name="weight_table"></a>
Class name   | Number of images in the class | Class weighting factor
:-----------:|:-----------------------------:|:----------------------:
Covid        | $3315$                          | $0.0003$
Pneumonia    | $1045$                          | $0.0010$
Lung opacity | $3416$                          | $0.0003$
Normal       | $9892$                          | $0.0001$

Using this approach will allow you to use approximately the same number of images of each class in each batch. This may mean using the same image several times.

<a name="hyperparameters"><h2>Selection of hyperparameters</h2></a>
**The number of neurons in the hidden layer.** The number of neurons in the hidden layer of the classifier was chosen to be the same for all models: 512.

**Learning rate.** To estimate the effective learning rate, the models were trained at a rate that was initially low, and then exponentially increased with each iteration:
$$lr_{max}=lr_{init}q^n,$$
$$q=\left(\frac{lr_{max}}{lr_{init}}\right)^{\frac{1}{n}},$$
$$lr_i=lr_{init}q^i=lr_{init}\left(\frac{lr_{max}}{lr_{init}}\right)^{\frac{i}{n}},$$
where $lr_{max}$ is final learning rate (upper bound), $lr_{init}$ is initial learning rate (lower bound), $n$ is number of iterations, $lr_i$ is learning rate based at step $i$.

After training the model, an interval of the learning rate is selected at which the value of the error functional decreases most quickly. This interval will be called **_optimal_**. Further training of the model can be carried out in different ways:
1. with the upper value of the optimal interval,
2. with the lower value of the optimal interval,
3. with a value 10 times less than the upper optimal interval (best upper bound),
4. using [cyclical learning rates](https://arxiv.org/abs/1506.01186).

The initial learning rate intervals were chosen to be the same for all architectures: $lr_{max}=1 \times 10^{-7}$, $lr_{init}=1 \times 10^{-1}$. After training all models with the learning rate from the [initial interval](#initial_intervals_img), [optimal intervals](#optimal_intervals_table) and the best upper and lower bounds were selected for each model.

<a name="initial_intervals_img">![initial interval](https://github.com/businsweetie/data_science_projects/blob/main/detection-of-coronavirus-infection-by-X-rays/pic/AllScheduler.jpg)</a>

<a name="optimal_intervals_table"></a>
<table>
    <thead>
        <tr>
            <th>Architecture</th>
            <th>Type of learning rate</th>
            <th>Learning rate value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>ResNet-18</td>
            <td>permanent</td>
            <td>$1 \times 10^{-6}$</td>
        </tr>
        <tr>
            <td>permanent</td>
            <td>$3 \times 10^{-4}$</td>
        </tr>
        <tr>
            <td><em>permanent</em></td>
            <td>$3 \times 10^{-5}$</td>
        </tr>
        <tr>
            <td><em><strong>cyclical</strong></em></td>
            <td>$[5 \times 10^{-6}; 3 \times 10^{-5}]$</td>
        </tr>
      <tr>
            <td rowspan=4>DenseNet-121</td>
            <td><em>permanent</em></td>
            <td>$1 \times 10^{-6}$</td>
        </tr>
        <tr>
            <td><em><strong>permanent</strong></em></td>
            <td>$9.5 \times 10^{-5}$</td>
        </tr>
        <tr>
            <td>permanent</td>
            <td>$9.5 \times 10^{-6}$</td>
        </tr>
        <tr>
            <td>cyclical</td>
            <td>$[1.5 \times 10^{-6};9.5 \times 10^{-5}]$</td>
        </tr>
      <tr>
            <td rowspan=4>EfficientNet-B0</td>
            <td>permanent</td>
            <td>$1 \times 10^{-5}$</td>
        </tr>
        <tr>
            <td><em><strong>permanent</strong></em></td>
            <td>$9 \times 10^{-4}$</td>
        </tr>
        <tr>
            <td><em>permanent</em></td>
            <td>$9 \times 10^{-5}$</td>
        </tr>
        <tr>
            <td>cyclical</td>
            <td>$[1.5 \times 10^{-5};9 \times 10^{-5}]$</td>
        </tr>
    </tbody>
</table>

<a name="training"><h2>Model training</h2></a>

For the trained models, graphs were constructed with the accuracy metric values for each of the [models](#models_comp_img).

<a name="models_comp_img">![Graphs of model comparisons](https://github.com/businsweetie/data_science_projects/blob/main/detection-of-coronavirus-infection-by-X-rays/pic/ModelsComp.png)</a>

Based on the graphs, models were identified for different architectures that achieve maximum accuracy faster. Such models are shown in italics in the [table](#optimal_intervals_table).

<a name="evaluation"><h2>Evaluating results</h2></a>

The performance of different convolutional neural network models was evaluated using the following metrics: accuracy, recall, precision and $F_1$.

Metric values for each class were calculated for each of the models. The [table](#best_metrics_table) shows models for each class that show the best accuracy metric. Models that showed the best predictive power are highlighted in black in the [table](#optimal_intervals_table).

<a name="best_metrics_table"></a>

Class name   | Architecture    | Model               | Accuracy
------------:|:---------------:|:-------------------:|:--------
Covid        | EfficientNet-B0 | $9\times 10^{-4}$   | $1.000000$
Pneumonia    | DenseNet-121    | $9.5\times 10^{-5}$ | $0.996540$
Lung opacity | EfficientNet-B0 | $9\times 10^{-4}$   | $0.968067$
Normal       | DenseNet-121    | $9.5\times 10^{-5}$ | $0.961603$

For the models that showed the best predictive ability, a [graph of the correctness value](#acc_comp_img) was constructed.

<a name="acc_comp_img">![Comparison of models](https://github.com/businsweetie/data_science_projects/blob/main/detection-of-coronavirus-infection-by-X-rays/pic/AccCompCovid.png)</a>

<a name="vizualization"><h2>Visualization of results using GradCAM</h2></a>

The [GradCAM](https://arxiv.org/pdf/1610.02391.pdf) visualization algorithm was applied to the classes for which the EfficientNet-B0 architecture showed the best result in the [figure](#grad_cam_img) (visualization of the EfficientNet-B0 neural network: a) for the covid class, b) for the lung opacity class)

<a name="grad_cam_img">![Visualization](https://github.com/businsweetie/data_science_projects/blob/main/detection-of-coronavirus-infection-by-X-rays/pic/GradCam.jpg)</a>

<a name="сonclusion"><h2>Conclusion</h2></a>

Very high accuracy of predictions was achieved for all metrics for all architectures and for all classes of images. For practical use in a healthcare facility, if a person is suspected of having COVID, the doctor should rely on the prediction that the EfficientNet-B0 architecture model provides.

We can assume that the EfficientNet architecture works slightly better than others, since it has fewer parameters, respectively, fewer weights, and therefore the gradient is less attenuated or explodes, so the network works stably and the prediction accuracy is higher. The other two networks are not much worse than EfficientNet, since their idea is to add intermediate links between layers. DenseNet works more accurately than ResNet, since it transmits not just connections, but entire layers, and at the same time to each subsequent block. Therefore, the final block receives all possible options for the operation of convolutional neural networks and can effectively discard some unnecessary filters and leave only important ones.

It is also worth noting that all architectures have high predictive power for the "covid" and "pneumonia" classes, and slightly lower predictive power for the "lung opacity" and "normal" classes. When using these architectures in practice, the health professional should be more careful when making a diagnosis.
