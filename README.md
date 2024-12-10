# ID2223_Lab2
Describe in your README.md program ways in which you can improve
model performance are using

(a) model-centric approach - e.g., tune hyperparameters, change the fine-tuning model architecture, etc
- Include hyperparameter-optimization tools such as Optuna. Decide upon a number of parameters and ranges that we
want to test and use for instance TPE or Bayesian Optimization. Seeing that it is time-consuming to 
train LLMs we could omit training each setting for epoch(s) and go for several batches instead. Any
parameter setting that quickly shows bad results within a number of batches probably won't recover
so we can then move on to the next setting. 
- If possible parallelize the model to speed up training, allowing more parameter testing and training
on larger datasets
- Use regularization techniques such as dropout, weight decay, early stopping, etc.


(b) data-centric approach - identify new data sources that enable you to train a better model than the one provided in the blog post
- Scrape websites such as Wikipedia/Twitter or other sites with dense text, divide them up into samples and mask parts of it that 
the then model attempts to fill in during training.
- Use data augmentation to improve the quantity of data to be trained on (such as adding noise, sentence splitting, etc).
- Use uncertainty labelling. Measure somehow the models confidence and then use it to predict on an unlabeled dataset. The
confident predictions can be adopted as training data without manual validation while the lower confidence ones are validated
manually. 

Some additional questions and answers:
- What does it mean to quantize weights?
This means turning high-precision (e.g. 32-bit) weights into a sparser range (e.g. 8-bit). The purpose of doing this is to speed
up training and also make it less computationally demanding. Doing this can however come at a cost of accuracy and precision,
fine nuances in the model's prediction won't be captured to the same extent if we make weights more coarse or clip them. 

- Why did you choose this specific model
The smaller model what chosen due to time limit. Training the model for one epoch took several days to picking a heavier one would
not have been feasible

- What does Gradient Accumulation?
This technique is useful when our computational resources are limited and we wish to use some computationally intensive batch-
size. We can divide the desired batch-size into mini-batches, and then compute a forward and backward pass for each such batch.
The gradient steps for each mini-batch are then accumulated which then is used to update the model's weights, effectively
mimicking a larger batch size.

- What are Warm-up steps?
Warm-up steps is a way of easing into a desired learning rate. We start with a small learning rate and then gradually increase
it. The purpose of doing this is to avoid taking steps in the loss function based on noisy gradients (which can be common in the
early phase of training).

- How is gradient and model-checkpointing done?
In SFTTrainer(), the arguments will do this:
        gradient_checkpointing=True,
        max_steps = 0,
        output_dir = "/content/drive/MyDrive/...",
        resume_from_checkpoint = "/content/drive/MyDrive/ID2223_Lab2/...",
        save_steps=100,

In this example, the model is saved every 100th epoch. Once the training resumes at some later time, the model will continue from
the epoch specified in "resume_from_checkpoint" and use the latest weights. 
