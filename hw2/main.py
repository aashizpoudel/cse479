import tensorflow as tf
import util 
import model
import matplotlib.pyplot as plt
import os 
import datetime
import json
import pandas as pd
import numpy as np


def run_experiment(model_fn, train_set, val_set, batch_sizes=[32],learning_rates=[0.01], result_dir="./results", patience=10,weight_decay=0): 
    os.makedirs(result_dir,exist_ok=True)
    max_epochs=200
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    exp_results = []
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            folder = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            folder_path = f"{result_dir}/{folder}"
            
            os.makedirs(folder_path,exist_ok=True)

            
            exp_model = model_fn()
            train_ds = train_set.map(util.preprocess_data).map(util.augment_data_for_resnet).batch(batch_size)
            val_ds = val_set.map(util.preprocess_data).batch(batch_size)

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience,monitor="val_loss",restore_best_weights=True)

            exp_model.compile(optimizer=optimizer,loss=loss_fn)

            training_history = exp_model.fit(train_ds,validation_data=val_ds, epochs=max_epochs,callbacks=[early_stopping])

            plt.plot(training_history.history['loss'],label="training loss")
            plt.plot(training_history.history['val_loss'],label="validation loss")
            plt.legend()
            plt.savefig(f"{folder_path}/training_history.jpg")
            plt.close()

            exp_model.save(f"{folder_path}/best.hd5") 

            acc,f1, _ = util.evaluate(exp_model,val_ds,top_k=5)

            exp_results.append({'val_acc':acc,'val_f1':f1,'folder':folder_path,'batch_size':batch_size,'learning_rate':learning_rate})
            del exp_model



    result_df = pd.DataFrame(exp_results)
    if(os.path.exists(f"{result_dir}/exp_result.csv")):
        previous_df = pd.read_csv(f"{result_dir}/exp_result.csv")
        result_df = pd.concat([previous_df,result_df],ignore_index=True, sort=False)

    result_df.to_csv(f"{result_dir}/exp_result.csv",index=False)

#     best_model_hyps = result_df[result_df['val_acc']==result_df['val_acc'].max()]
#     best_folder = best_model_hyps['folder'].item()
#     print("Best model")
#     print(best_model_hyps)

#     best_folder_path = f"{best_folder}"

#     best_model = tf.keras.models.load_model(f"{best_folder_path}/best.hd5")

#     acc,f1,cf = util.evaluate_on_test(best_model,top_k=5)
#     print("Accuracy(top-5) and F1 for best model")
#     print(acc, f1)
    
    return result_df

