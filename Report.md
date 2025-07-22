# Study Project: Lane Detection

Author: Joey van der Weijden

## Contents

1. [Introduction](#introduction)
2. [Data](#data)  
    1. [TuSimple](#tusimple)
    2. [Automatic approach](#automatic-approach)
    3. [Manual approach](#manual-approach)  
3. [Model](#model)  
    1. [CLRNet](#clrnet)
    2. [ROIGather](#roigather)
    3. [IoULoss](#line-iou-loss)
4. [Training](#training)  
    1. [Code preperation](#code-preperation)
    2. [Training Process](#training-process)
5. [Results](#results)  
    1. [Quantative results](#quantative-results)
    1. [Qualative results](#qualative-analysis)

## Introduction

Lane detection is a critical component in the development of Advanced Driver Assistance Systems (ADAS). Accurate identification of lane markings allows for lane keeping and safe navigation. The goal of this project was to take an existing neural network structure and train it specifically for a HSE simulation.

In this project, a deep learning approach was used to automatically learn lane features from a dataset of labelled images. Compared to more traditional computer vision techniques, which often rely on edge detection and hand-crafted filters, neural networks should be able to offer a more robust and adaptable system. However, as will be discussed during the result section, does the chosen network struggle with a highly varied environment.

During this report the words "the simulation" will occur multiple times. This refers to a Gazebo simulation environment in which a virtual car can drive around a circuit. This simulation is part of the Autonomous Systems Design course. Below is an example taken from the input images, which will are used to create the annotated dataset.

![Simulation_example](.github/simulation_example.png)  
*Example image taken from the simulation*

## Data

### TuSimple

The TuSimple lane detection benchmark is one of the most used datasets in the field of autonomous driving. It provides a standardized dataset for the evaluation of the performance of lane detection algorithms under real-world driving conditions. Released by the autonomous trucking compnay TuSimple, the dataset has been a key performance indicator of many lane detection models.

The dataset contains over 6,400 clips, each made up of 20 frames (with a resolution of 1280 x 720 pixels). From the 20 frames, the final one is annotated with lane markings, while the other frames are mainly used as context. The labels include precise coordinates of up to four lanes. The lanes are represented using a series of x and y pixels, at fix y-positions (also referred to as horizontal samples, or hsamples.)

The dataset focuses on highway scenarios, with relatively consistent lighting and lane visibility. It mainly contains straight, or very slightly curved lanes, recorded during the daytime with clear weather conditions. This means it offers a real-world yet controlled dataset.

Howevver the dataset has a few limitations. The dataset lacks diverse driving scenarios, such as urban areas, bad weather and complicated curves or intersections.

For this project the TuSimple benchmark served as a reference framework for a custom dataset. Key elements taken for the benchmark are the annotation format and data  structure. This ensures compatibility with common lane detection models. The advantage of creating a custom dataset is a model that is specialised in the HSE simulation environment, plus being able to add more difficulit scenarios to the dataset.

### Automatic approach

The first step of the entire process was to create a dataset. To accomplish this the existing [TuSimple benchmark](https://paperswithcode.com/sota/lane-detection-on-tusimple) was used as a guideline. The goal of the data preparation step of this project was to create an annotated dataset, in the format of TuSimple, using images taken from the simulation.

Since before the emergence of deep learning lane detection was tackled using traditional computer vision techniques, the first approach to creating a dataset was to use these methods to automatically annotate. The reasoning behind this was that these proven techniques should be sufficient to label the images, which then could be used to train a model, which would be a lot more flexible and robust than the traditional computer vision techniques.

**Edge detection** was the first approach. Edge detection is a very standard step in traditional lane detection. The canny edge detector was used abrupt intensity changes in the image that might correspond to lane boundaries. This was typically followed by binary thresholding to isolate high-contrast lines on the road surface. While these methods could highlight lane markings on clean, high-contrast roads, they were highly sensitive to lighting conditions, shadows, road texture, and faint or worn markings. Furthermore, they often detected irrelevant edges from objects like guardrails, road signs, or pavement cracks, making it difficult to distinguish true lane lines from background noise. Expectations for this approach were high, since the simulation environment has a very distinct difference between the pure black road and pure white lines. The edge detector performed very well on the areas which were close to the point of view of the image, but degraded significantly as you moved further to the horizon. The reason for this is that the lines become increasingly blurry as you approach the horizon. Also since there is a sharp difference between the road and the "sky", this was often also detected as a lane.

To remove the issue with the horizon from the equation, a **region of interest (ROI)** mask was used. Usually this is a trapezoidal area in the lower half of the image where the lane lines are expected to appear. The effect of this mask disappointed. Reason for this is that the horizon is not always in the same place. As can be seen in the [example image](#introduction) the horizon sometimes appears right next to the road.

After the edges were extracted with the canny edge decteor and false positives were removed as much as possible using the ROI mask, the **Hough Transform** was used to detect linear features. This technique identifies lines by finding points in the image that align according to certain geometric parameters [e.g., angle and distance from origin in polar coordinates]([4)](#sources). While the Hough lines would reliably dected the lanes for the straight sections of the simulation environment, it was not very good at following curves. Since the simulation environment has a lot of very sharp turn it quickly became evident that this approach was not sufficient.

In one final attempt to automate the proces, an apporach was used in which a lot of assumptions were taken. Assuming the car is always properly in between the right two lanes, first find the lane on the left, and call it lane 0, then do the same for the right lane and the left-most lane. This worked semi-reliably for the areas close to the cameras pov, but became increasingly more difficult as the image moved closer to the horizon, since the perspective made the lanes appear very close together in some of the samples. A perspective transfrom was applied to limit this effect, but due to the quick degredation of lane clarity within the image, this also did not solve the problem.

### Manual approach

After many attempts to get the automatic labeller to work reliably, the choice was made that the time already spent on automating the process, could also have been used to manually label the dataset and therefore the decision was made to change the approach.

#### Labeller

The dataset contained around 400 images and to label these as efficiently as possible the first step was to create a tool which allowed for easy labelling. To create this tool, the python package [TKinter](https://docs.python.org/3/library/tkinter.html) was used, which is the standard Python interface to the Tcl/Tk GUI toolkit.

The tool displays the image to be labeled, along with horizontal reference lines (known as hsamples) that represent the specific y-coordinates used in the TuSimple annotation format. These hsamples are plotted on the image to help the user determine where to place lane points. Unlike the actual TuSimple dataset, the simulated environment includes sharp turns, which can lead to situations where a single lane intersects the same hsample line more than once. This poses a problem for labelling quality, as the TuSimple format allows only one x-coordinate per hsample. If the incorrect intersection point is selected, it can cause significant errors in the shape of the detected lane, potentially distorting the entire trajectory.

This tool can be started with `labelling_tool/manual_labeller`.

![labeller_image](.github/labeller_example.png)  
*Image taken from the labeller, where the two out of three lanes have been labelled*

#### Quality checker

The previously mentioned issue with hsample intersections was only discovered after more than 100 images had already been labeled. This presented a dilemma: either discard all *potentially* faulty labels, or manually review each image to verify their accuracy.

The latter option was chosen to preserve as much usable data as possible. To support this process, a simple quality-checking tool was created. This tool loads each labeled image along with its corresponding labels. The user then has two options: either confirm and keep the label if the hsample intersection issue does not apply, or delete the label so that the image can be re-labeled correctly.

The quality checker tool can be started with `labelling_tool/quality_checker.py`.

#### restructure dataset

The final step in the dataset creation process was to convert the labeled images into the format required by the model. This involved splitting the data into training, validation, and test sets, using a standard 80/10/10 ratio.

To ensure a balanced distribution, all labeled images were first shuffled randomly, preventing any one subset from being biased toward a particular road section. After shuffling, a new directory was created to store all the images, while a separate annotation file was generated. This file included the lane coordinates, the corresponding hsamples, and the file path to each image—structured in accordance with the TuSimple format.

## Model

For this project CLRNet (Cross Layer Refinement Network) was selected as the neural network.

The primary motivation for choosing CLRNet was its high performance on the [TuSimple benchmark](https://paperswithcode.com/sota/lane-detection-on-tusimple). The model's accuracy scores in the top 10 twice (once with a resnet-18 backbone and once with a resnet-34 backbone). If the updated version, CLRNetV2, is also included in counting, the models have 4 out of the top 10 spots.

Another significant advantage of CLRNet was the availability of well-documented and reproducible code via its official [GitHub repository](https://github.com/Turoad/CLRNet?tab=readme-ov-file). This offered a smooth setup, training, and evaluation process without requiring extensive model engineering from scratch. The repository includes training scripts and configuration files, which streamlined the use of the model.

Current CNN-based lane detection approaches generally can be divided into three categories: segmentation-based methods, anchor-based methods and parameter-based methods.

**Segmentation-based methodes:** An approach that generally delivers accurate predictions at the cost of speed is the segmentation-based method. For this, algorithms adopt a pixel-wise prediction formulation, for example by treating the entire lane detection task as a segmentation task. This approach manages to capture good spatial relationships for the lanes, but is much too slow for real-time applications, which lane detection tasks almost always are. The reason that these segmentation-based methods are so time-consuming is that they perform pixel-wise predictions on the entire image, without considering the lanes as a whole unit.

**Anchor-based methods:**: Anchor-based lane detection methods draw inspiration from object detection models such as YOLO[[3]](#sources), which use predefined anchor boxes across the image to localize and classify objects. Instead of performing pixel-wise segmentation, these methods predefine a set of anchor positions, typically fixed rows across the image, and then train the model to predict lane attributes (such as the presence of a lane point) at each anchor.  In practice, this means that for each row (or vertical position), the model predicts whether a lane exists and, if so, the horizontal offset from the anchor point. This representation converts lane detection into a classification and regression task, where each lane is described as a sequence of points at fixed vertical intervals.  A noteable example of an anchor-based model is LaneATT [[1]](#sources). proposed a novel

anchor-based attention mechanism that aggregates global

information. This allowed it to achieve state-of-the-art results and shows

both high efficacy and efficiency. Anchor-based methods tend to be more efficient and compact compared to segmentation-based approaches, as they reduce the output space and post-processing complexity. However, they usually require anchor matching strategies during training, where predicted outputs are assigned to ground truth lanes based on heuristics or distance thresholds — a process that can be brittle or inflexible in scenarios with varying lane counts, occlusions, or curved lanes. spite these challenges, anchor-based models strike a strong balance between accuracy and inference speed, making them well-suited for real-time applications like autonomous driving.

**Parameter-based methods:** Parameter-based methods approach lane detection by directly regressing the mathematical parameters that define a lane’s shape, rather than predicting pixel masks or anchor points. These models treat lanes as parametric curves, often modeled using polynomials (e.g., second- or third-degree) or splines.  The task of lane detection is thus reformulated into a curve-fitting problem, where the neural network predicts the coefficients of these curves based on the input image. A representative example is PolyLaneNet[[2]](#sources), which regresses the parameters of a predefined polynomial to represent each lane line. This approach has several advantages: it produces smooth and compact lane representations, reduces output dimensionality, and simplifies post-processing. However, it also introduces key limitations. Parameter-based methods often assume a fixed number of lanes and a fixed curve form, which can reduce flexibility in complex road scenarios involving lane splits, merges, or occlusions. Additionally, directly regressing curve parameters can be sensitive to noise and initialization, and may struggle to generalize across datasets with varied road geometries. Despite these challenges, parameter-based models are computationally efficient and are particularly well-suited to structured environments like highways, where lanes tend to follow predictable shapes.

*Model description based on the [CLRNet paper](https://arxiv.org/abs/2203.10350)*

### CLRNet

Convolutional Neural Networks (CNNs) have made significant advances in lane detection by learning powerful feature representations. Many recent methods have achieved promising results using CNN-based architectures. However challenges remain. ONe of the fundemental issues is with the multi-scale nature of lane featurues. Lane markings are thin and long objects with a relatively simple appearance, yet distinguishing them from visually similar object requires high-level semantic underatnding and global context.

Low-level CNN features are rich in spatial detail but lack semantic understanding. Conversely, high-level features capture global context but tend to lose fine-grained localization accuracy. This gap creates a trade-off in many lane detection systems: reliance on low-level features can cause confusion between lanes and similar patterns (e.g., mistakenly identifying road landmarks as lane lines), while relying solely on high-level features may lead to imprecise localization of lanes.

Other approaches either model local geometry and merge it into global predictions, or apply fully connected layers using global context. While these methods emphasize either local or global features, they often fail to integrate both effectively, leading to suboptimal detection performance.

To address these challenges, Cross Layer Refinement Network (CLRNet) has been introduced, a novel framework that fully exploits both low-level and high-level features for enhanced lane detection. The network operates in two stages: it first performs coarse lane localization using high-level semantic features, followed by a refinement stage leveraging fine-detail low-level features to achieve precise positioning. This progressive refinement approach significantly improves detection accuracy.

CLRNet is best understood as an anchor-based lane detection model, though it incorporates elements that differentiate it from traditional methods in this category. Like other anchor-based approaches such as LaneATT, CLRNet adopts a row-anchor-based representation, where the vertical positions of lane points are predefined, and the model predicts the horizontal locations and presence of lanes at those anchors. This allows the model to output lane structures as a sequence of coordinate points rather than dense pixel maps. However, CLRNet simplifies the typical anchor-based pipeline by removing complex anchor matching procedures during training. Instead of assigning predictions to predefined anchors through heuristic matching, CLRNet directly supervises lane point coordinates using a simplified training objective. This makes it more flexible and reduces sensitivity to anchor design. The model balances structural precision with efficiency by directly predicting lane points at fixed intervals using a lightweight, refined feature aggregation mechanism. This combination of anchor-based design with direct coordinate regression and multi-scale refinement positions CLRNet as a hybrid solution that improves both performance and practicality for real-time lane detection.

To further enhance robustness in scenarios with weak visual evidence, ROIGather has been added, a module that captures global contextual information by linking lane-region features with the entire feature map. This enables the network to reason about lane structures in a broader context.

Moreover, Line IoU (LIoU) loss was introduced, a new loss function specifically designed for lane detection. Unlike standard loss functions such as smooth-L1, LIoU directly optimizes the alignment of predicted and ground-truth lane lines as whole units, significantly boosting performance.

#### ROIGather

In challenging scenarios, for example when lane markings are partially covered or distorted by lighting, local visual cues may be insufficient for accurate lane detection. To address this, the model's contextual understanding of lane features has been enhanced by aggregating long-range dependencies.It incorporates convolutions along lane priors, enabling each lane pixel to gather information from its surroundings and compensate for missing or degraded visual evidence.

For this purpose ROIGather, a lightweight and efficient module designed to strengthen lane feature representations. It extracts features from lane priors using ROIAlign, uniformly sampling key points along each lane and applying bilinear interpolation. To enrich these features, we perform convolutions along the lane priors and apply a fully connected layer for dimensionality reduction.

To further incorporate global context, ROIGather computes an attention-based relation between the ROI lane features and the entire feature map. This attention mechanism selectively aggregates global features, enhancing the lane prior features with complementary information from the broader scene.

By integrating both local and global context, ROIGather significantly improves the robustness and accuracy of lane detection, particularly under challenging visual conditions.

#### Line IoU loss

Traditional lane detection methods often rely on point-wise distance-based loss functions such as smooth-L1, which treat each point on the lane independently. However, this oversimplified approach overlooks the structural continuity of lane lines, leading to suboptimal regression accuracy.

To better capture the holistic nature of lane structures, a Line IoU (LIoU) loss was introduced that treats the entire lane as a unified geometric entity rather than a collection of discrete points. Inspired by the standard Intersection over Union (IoU) metric used in object detection, LIoU computes the overlap between predicted and ground truth lane segments.

This approach exhibits two main advantages:

1. It is simple and differentiable, which is very easy to implement parallel computations.

2. It predicts the lane as a whole unit, which helps improve the overall performance.

## Training

### Code preperation

Before training could begin, adjustments were required to adapt the original TuSimple configuration to the custom simulation dataset, referred to as SimSimple. While the SimSimple dataset was designed to closely mimic TuSimple in terms of format and structure, there were still some differences—most notably in image resolution and the set of hsamples (the fixed vertical y-positions used for lane point annotations). These changes impacted multiple components of the CLRNet pipeline and required careful tuning to ensure compatibility.

The first step in altering the code was to create a custom configuration file based on the original tusimple.py. This involved updating dataset-specific fields such as the image size, the list of hsamples, the number of lanes, and file paths for training, validation, and testing. Additionally, several components in the CLRNet codebase had hardcoded assumptions about the TuSimple format. These included:

- Input size normalization parameters

- The number of anchor points along the y-axis

- Post-processing logic relying on fixed input dimensions

The most technically involved adjustment was within the non-maximum suppression (NMS) kernel, which had to be rebuilt after modifying its dependencies to support the new image size and sample spacing. Without these changes, the model would incorrectly filter overlapping lane predictions or fail to process outputs altogether.

### Training Process

Training was conducted on three backbone variants of CLRNet: ResNet-18, ResNet-34, and ResNet-101. All models were trained using the same hyperparameters to ensure a fair comparison. The training environment included an NVIDIA RTX 3080 GPU and an Intel i5-12600KF CPU, which provided sufficient compute power to complete training runs efficiently.

Training Parameters:

    Epochs: 60–70

    Batch Size: 16

    Optimizer: AdamW

    Learning Rate: Default from TuSimple config

    Scheduler: Cosine decay with warmup

    Loss Function: Line IoU (as described in the model section)

    Train/Val/Test Split: 80% / 10% / 10%

Training durations varied significantly depending on the model size:

    ResNet-18: ~18 minutes (70 epochs)

    ResNet-34: ~20 minutes (60 epochs)

    ResNet-101: ~1 hour 54 minutes (60 epochs)

During training, all models exhibited fast initial convergence, with validation losses plateauing after approximately 40–50 epochs. ResNet-18, while faster, occasionally showed signs of underfitting — particularly in the form of unstable validation loss and misclassified lanes in high-curvature regions. In contrast, ResNet-101 demonstrated smoother training curves and better generalization, though at the cost of significantly longer training time and higher GPU memory usage.

ResNet-18 is a strong choice for quick prototyping, though its performance on complex curves was limited.

ResNet-34 offered a good balance between training time and predictive accuracy.

ResNet-101 achieved the best performance in terms of F1 score but with diminishing returns relative to the large increase in resource consumption.

## Results

The performance of CLRNet was evaluated on the custom SimSimple dataset using three different backbone architectures: ResNet-18, ResNet-34, and ResNet-101. Both quantitative metrics and qualitative visual analysis were used to assess the effectiveness and limitations of the models.

The models were evaluated using the following standard lane detection metrics:

- F1 Score: The mean of precision and recall, indicating the balance between false positives and false negatives.

- Accuracy (Acc): The percentage of correctly predicted lane points over all predicted and ground truth points.

- False Discovery Rate (FDR): The percentage of lane points predicted by the model that were incorrect.

- False Negative Rate (FNR): The percentage of ground truth lane points that were missed by the model.

### Quantative results

|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][18]     |    97.37    |   94.50  |    2.63  |  2.63      |
| [ResNet-34][34]       |   98.25              |    96.47          |   1.75          |    1.75      |
| [ResNet-101][101]      |   98.25|   96.64  |   1.75   |  1.75  |

[18]: #resnet-18
[34]: #resnet-34
[101]: #resnet-101

The results indicate a clear correlation between backbone depth and detection quality. The ResNet-34 and ResNet-101 backbones significantly outperformed ResNet-18 in all metrics. Interestingly, ResNet-101 did not substantially surpass ResNet-34, despite its deeper architecture and longer training time. This suggests that ResNet-34 offers the best trade-off between performance and computational cost in this simulation context.

### Qualative Analysis

To complement the numerical results, predictions were visualized across a range of scenarios from the test dataset, including:

- Straight lanes

- Mild curves

- Sharp turns

### Resnet-18

Although the ResNet-18 model produced usable results in many cases, it exhibited several consistent weaknesses:

- Lane Omission: In several instances, the model failed to detect one or more lanes, especially the right-most lane, which tended to be less prominent due to camera angle and lighting.

- Drifting: The predicted center lanes occasionally drifted or bent in unnatural directions, particularly near the top of the image.

These issues can reflect the more limited limited capacity of shallower network to integrate semantic and spatial information across large receptive fields.

### Resnet-34

The ResNet-34 model produced substantially more stable and accurate results:

- Consistent Detection: All three lanes were reliably detected in most images, even on curves.

- Improved Horizon Accuracy: While predictions still weakened near the horizon, the degradation was less severe than with ResNet-18.

ResNet-34 appears to be a strong middle ground, delivering high accuracy without the high resource demands of deeper models.

### Resnet-101

As the deepest network tested, ResNet-101 achieved the best validation accuracy and most robust lane shapes:

- Consistent Detection: All three lanes were reliably detected in most images, even on curves.

- Superior Handling of Curves: While not perfect, it better followed the geometry of sharp turns, which were problematic for ResNet-18.

- Longer Prediction Range: The model extended predictions further toward the horizon, although precision still dropped slightly in the far field.

The trade-off, however, was a significant increase in training time and memory usage, with diminishing returns over ResNet-34.

For reference, some example images have been appended to this report, combined with the training charts.

### Additional observations

One particularly interesting observation during qualitative evaluation is that the predicted lane shapes often appear to follow a simple, smooth curve, closely resembling a low-degree polynomial—regardless of the actual geometry of the lane. This behavior was consistent across all model variants, though it was slightly less pronounced in the deeper backbones.

any test samples, especially those involving tight turns or sudden curvature, the model clearly identified the presence of the lane correctly, but its prediction would deviate from the actual path, cutting across the turn or failing to bend sharply enough. This suggests that the model is not struggling with detection per se, but rather with the geometric complexity of the lane trajectory.

A likely explanation for this phenomenon lies in the model's creation. CLRNet was originally designed and benchmarked on datasets such as TuSimple, which predominantly feature straight or gently curved highway lanes. In such settings, approximating lanes with smooth, almost-linear curves is both sufficient and optimal. However, the SimSimple dataset used in this project contains significantly more aggressive curvature, including sharp bends, intersections, and perspective warping—scenarios underrepresented in TuSimple.

As a result, it appears that the model learned a general bias toward simplified lane shapes, possibly as a form of regularization, but at the cost of fidelity in high-curvature regions. This highlights an important limitation: while the model may perform well numerically and in standard conditions, it may oversimplify geometry in more diverse or challenging environments unless retrained on data that reflects those conditions.

### Summary

Some example images have been appended

ResNet-18, while fast, suffered from underfitting. ResNet-101 offered top-tier accuracy but at high computational cost. ResNet-34 appears to be the most practical model, balancing robustness and resource demands — an important consideration for embedded or real-time applications.

The experiments confirm that CLRNet can be successfully adapted to a simulated environment with minimal architectural modifications. However, its real-world performance remains tightly coupled to training data diversity. The strong numerical results (F1 > 98%) should be viewed alongside the qualitative limitations revealed in complex lane geometries and long-range perception.

For future work, performance could likely be improved by:

- Expanding the training dataset to include more curved, occluded, and low-visibility lane types

- Modifying the loss function to penalize geometric deviation more strictly

- Experementing with a different model alltogether

### resnet-18

**Training**  

![training_18](.github/training_resnet18.png)

**Example prediction**  

![training_18](.github/resnet_18_147.png)

![training_18](.github/resnet_18_367.png)

### resnet-34

**Training**  

![training_34](.github/training_resnet34.png)

**Example predictions**  

![training_18](.github/resnet_34_147.png)

![training_18](.github/resnet_34_367.png)

### resnet-101

**Training**  

![training_101](.github/training_resnet101.png)

**Example predictions**  

![training_18](.github/resnet_101_147.png)

![training_18](.github/resnet_101_367.png)

## Conclusion

Working on this project has been very interesting. Initially I had a basic understanding of computer vision approaches to lane detection. By trying to implement I learned a lot more about these approaches, along with their implementation challenges and shortcomings. It was interesting to see that with python the basic implementation can be achieved very easily, but to get it to work on a more complex environment is a completely different story.

The first time I went over the CLRNet paper I struggled a lot with the terminology and had to repeatedly look up certain terms. Over the project this became a lot easier, and notice this as well when looking at different papers on the subject.

Personally I will probably continue to try implementing an automatic approach to label the data, it remains an unresolved challenge I hope to address in future work

## Sources

1. TuSimple. Tusimple benchmark. <https://github>.com/TuSimple/tusimple- benchmark/
2. Tabelini, L., Berriel, R., Paixão, T.M., Badue, C., Souza, A.F.D., Oliveira-Santos, T. (2020). PolyLaneNet: Lane Estimation via Deep Polynomial Regression. arXiv preprint arXiv:2004.10924.
3. Terven, J., Cordova-Esparza, D. (2023). A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS. arXiv preprint arXiv:2304.00501.
4. Vykari, N. (2021). *Understanding Hough Transform With A Lane Detection Model*. DigitalOcean. <https://blog.paperspace.com/understanding-hough-transform-lane-detection/>
5. Zheng, T., Huang, Y., Liu, Y., Tang, W., Yang, Z., Cai, D., He, X. (2022). CLRNet: Cross Layer Refinement Network for Lane Detection. arXiv preprint arXiv:2203.10350.
