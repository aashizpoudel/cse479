import tensorflow as tf 
import model
import utils
import pickle

tf.random.set_seed(1024)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Experiment with first architecture
optimizers = {'adam':"adam",'rmsprop':"rmsprop"}
batch_sizes = {'batch_8':8, 'batch_32':32}
regularizers = {'regularizer':True,'no_regularizer': False}

for batch_arg,batch_size_val in batch_sizes.items():
    for optim_arg,optimizer_var in optimizers.items():
        for regularizer_arg,use_regularizer in regularizers.items():
            print(f"Working on model arch 1,[{batch_arg},{optim_arg},{regularizer_arg}]")
            prefix = f"{batch_arg}_{optim_arg}_{regularizer_arg}"
            
            exp_model = model.model_1(use_regularizer=use_regularizer)                              #load model architecture
            exp_model.compile(optimizer=optimizer_var,loss=loss_fn,metrics=['accuracy'])
            
            train_ds = utils.get_training_set(batch_size=batch_size_val)                            # get train dataset
            val_ds = utils.get_val_set(batch_size=batch_size_val)                                   # get val dataset
            csv_logger_cb = tf.keras.callbacks.CSVLogger(filename=f"logs/{prefix}_experiment_log.csv")   # store logs in csv file
            exp_model.fit(train_ds,epochs=100,validation_data=val_ds,callbacks=[early_stopping_cb,csv_logger_cb]) # model training
            
            exp_model.save_weights(f"logs/{prefix}_best_weight.weights.h5") #saving the best model
            
            evaluation_results = utils.evaluate(exp_model)   #perform evaluation on test dataset 
            
            pickle.dump(evaluation_results,open(f"logs/{prefix}_evaluation_results.dat","wb"))  #save confusion matrix in file for future plotting
            print(f"Model arch 1,[{batch_arg},{optim_arg},{regularizer_arg}]=>Accuracy:{evaluation_results[0]:.3f},F1:{evaluation_results[1]:.3f}") # printing
            del exp_model #deleting model variable to recreate new model
            
            




# Experiment with second architecture




# Experiment with third architecture



