import tensorflow as tf 
import model
import utils
import pickle
import pandas as pd
import matplotlib.pyplot as plt


tf.random.set_seed(1024)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

# Experiment with first architecture
models = {1 : model.model_1, 2 : model.model_2}
learning_rates = {'0.01':0.01,"0.001":0.001}
batch_sizes = {'batch_8':8, 'batch_32':32}
regularizers = {'regularizer':True,'no_regularizer': False}
results_csv = []
for model_num,model_fn in models.items():
    for batch_arg,batch_size_val in batch_sizes.items():
        for lr_arg,lr_val in learning_rates.items():
            for regularizer_arg,use_regularizer in regularizers.items():
                single_result = {'model':model_num,'learning_rate':lr_arg,'batch_size':batch_size_val,'regularizer':use_regularizer}
                
                print(f"Working on model arch 1,{batch_arg},{lr_arg},{regularizer_arg}]")
                prefix = f"1_{batch_arg}_{lr_arg}_{regularizer_arg}"
                
                exp_model = model_fn(use_regularizer=use_regularizer)                              #load model architecture
                exp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_val),loss=loss_fn,metrics=['accuracy'])
                
                train_ds = utils.get_training_set(batch_size=batch_size_val)                            # get train dataset
                val_ds = utils.get_val_set(batch_size=batch_size_val)                                   # get val dataset
                if utils.dir_check(f"logs/{prefix}_experiment_log.csv"): 
                    csv_logger_cb = tf.keras.callbacks.CSVLogger(filename=f"logs/{prefix}_experiment_log.csv")   # store logs in csv file
                exp_model.fit(train_ds,epochs=200,validation_data=val_ds,callbacks=[early_stopping_cb,csv_logger_cb])
                if utils.dir_check(f"logs/{prefix}_best_weight.weights.h5"):
                    exp_model.save_weights(f"logs/{prefix}_best_weight.weights.h5") #saving the best model
                
                evaluation_results = utils.evaluate(exp_model,val_ds)   #perform evaluation on val dataset 
                
                print(f"Model arch 1,[{batch_arg},{lr_arg},{regularizer_arg}]=>Accuracy:{evaluation_results[0]:.3f},F1:{evaluation_results[1]:.3f}") # printing
                single_result['val_accuracy'] = evaluation_results[0]
                single_result['val_f1_score'] = evaluation_results[1]
                if utils.dir_check(f"logs/{prefix}_experiment_log.csv"):
                    single_result['converged_on']= pd.read_csv(f"logs/{prefix}_experiment_log.csv").tail(1)['epoch'].item()
                results_csv.append(single_result)
                if utils.dir_check(f"logs/{prefix}_experiment_log.csv"):
                    train_history = pd.read_csv(f"logs/{prefix}_experiment_log.csv")
                fig,axs = plt.subplots()
                axs.plot(train_history['loss'],label="Training loss")
                axs.plot(train_history['val_loss'],label="Validation loss")
                plt.title(f"Training history for batch size={batch_size_val}, learning_rate={lr_val}, regularizer={use_regularizer}.\n val accuracy {evaluation_results[0]:.3f})")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(f"logs/{prefix}_training_history.pdf")
                # plt.show()
                
                
                del exp_model #deleting model variable to recreate new model
            
final_result = pd.DataFrame(results_csv)
final_result.to_csv("logs/exp1.csv")
                      
#find best model 
# best_model_hyps = final_results[final_result['val_f1_score']==final_results['val_f1_score'].max()]






# Experiment with second architecture




# Experiment with third architecture



