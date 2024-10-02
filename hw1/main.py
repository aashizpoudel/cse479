import tensorflow as tf 
import model
import utils
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import os

#Create logs folder because everything will be stored in logs folder.
os.makedirs("./logs",exist_ok=True)

#Set random seed of 1024 for reproducibility
tf.random.set_seed(1024)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)



models = {1 : model.model_1, 2 : model.model_2}

#hyper parameters
learning_rates = {'0.01':0.01,"0.001":0.001}
batch_sizes = {'batch_8':8, 'batch_32':32}
regularizers = {'regularizer':True,'no_regularizer': False}

results_csv = []
for model_num,model_fn in models.items(): #iterate over model functions
    for batch_arg,batch_size_val in batch_sizes.items():
        for lr_arg,lr_val in learning_rates.items():
            for regularizer_arg,use_regularizer in regularizers.items():
                single_result = {'model':model_num,'learning_rate':lr_arg,'batch_size':batch_size_val,'regularizer':use_regularizer}
                
                print(f"Working on model arch {model_num},{batch_arg},{lr_arg},{regularizer_arg}]")
                prefix = f"{model_num}_{batch_arg}_{lr_arg}_{regularizer_arg}"
                
                exp_model = model_fn(use_regularizer=use_regularizer)                              #load model architecture
                exp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_val),loss=loss_fn,metrics=['accuracy'])
                
                train_ds = utils.get_training_set(batch_size=batch_size_val)                            # get train dataset
                val_ds = utils.get_val_set(batch_size=batch_size_val)  
                # get val dataset
                
                csv_logger_cb = tf.keras.callbacks.CSVLogger(filename=f"logs/{prefix}_experiment_log.csv")   # store logs in csv file
                start_time = time.time()
                exp_model.fit(train_ds,epochs=2,validation_data=val_ds,callbacks=[early_stopping_cb,csv_logger_cb])
                end_time = time.time() 
                
                duration = end_time - start_time
                
                exp_model.save_weights(f"logs/{prefix}_best_weight.weights.h5") #saving the best model for this run
                
                evaluation_results = utils.evaluate(exp_model,val_ds)   #perform evaluation on val dataset 
                
                print(f"Model arch {model_num},[{batch_arg},{lr_arg},{regularizer_arg}]=>Accuracy:{evaluation_results[0]:.3f},F1:{evaluation_results[1]:.3f}") # printing
                
                single_result['val_accuracy'] = evaluation_results[0]
                single_result['val_f1_score'] = evaluation_results[1]
                single_result['converged_on']= pd.read_csv(f"logs/{prefix}_experiment_log.csv").tail(1)['epoch'].item()
                single_result['prefix'] = prefix
                single_result['duration'] = duration
                results_csv.append(single_result)
                
                train_history = pd.read_csv(f"logs/{prefix}_experiment_log.csv")
                fig,axs = plt.subplots()
                axs.plot(train_history['loss'],label="Training loss")
                axs.plot(train_history['val_loss'],label="Validation loss")
                plt.title(f"Training history - Model:{model_num}\n Hyps: batch size={batch_size_val}, learning_rate={lr_val}, regularizer={use_regularizer}\n Val accuracy {evaluation_results[0]:.3f}")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(f"logs/{prefix}_training_history.pdf")
                # plt.show()
                plt.close()
                
                del exp_model #deleting model variable to recreate new model
            
final_result = pd.DataFrame(results_csv)
final_result.to_csv("logs/exp1.csv")
                      
#find best model 
final_result = pd.read_csv("logs/exp1.csv")
best_model_hyps = final_result[final_result['val_f1_score']==final_result['val_f1_score'].max()]
prefix = best_model_hyps['prefix'].item()
print("Best model is:", prefix)

best_model_weight = f"logs/{prefix}_best_weight.weights.h5"
if prefix[0] == '1':
    model_fn = model.model_1
else:
    model_fn = model.model_2

selected_model = model_fn()
selected_model.load_weights(best_model_weight)

#Load test set for final evaluation
test_set= utils.get_test_set(batch_size=1)
predictions = selected_model.predict(test_set)
labels = np.concatenate([t for _,t in test_set],axis=0).argmax(axis=-1)
predictions = predictions.argmax(axis=-1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


cm = confusion_matrix(labels, predictions)
print("Confusion matrix")
print(cm)
print("--------")
fig,ax = plt.subplots(figsize=(20,20))
# ax = fig.add_subplot(111)
cax = ax.imshow(cm)
plt.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
ax.set_xticks(np.arange(10), labels=class_names)
ax.set_yticks(np.arange(10), labels=class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="w")
        
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig("logs/confusion_matrix.pdf")
plt.close()
print("----------")
print("Final model results")
print("F1 Score(macro)",f1_score(labels,predictions,average="macro"))
print("Recall Score(macro)",recall_score(labels,predictions,average="macro"))
print("Precision Score(macro)",precision_score(labels,predictions,average="macro"))
print("Accuracy Score",accuracy_score(labels,predictions))

print("--------")
# Get random dataset and output image, prediciton and label.
random_three = test_set.shuffle(2).take(3)
count = 1
for image,labels in random_three:
    single_image = image
    single_label = tf.math.argmax(labels[0]).numpy()
    prediction = tf.math.argmax(selected_model(single_image),-1).numpy()[0]
    plt.imshow(single_image[0])
    plt.title(f"GT: {class_names[single_label]}, Prediction: {class_names[prediction]}")
    plt.savefig(f"logs/example_{count}.pdf")
    count += 1
    plt.close()