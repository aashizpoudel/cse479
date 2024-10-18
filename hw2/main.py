import tensorflow as tf
import util 
import model
import matplotlib.pyplot as plt
import os 
import datetime
import json
import pandas as pd
import numpy as np

os.makedirs("./results",exist_ok=True)
train_set,val_set = util.get_train_val_set()

batch_sizes=[32]
learning_rates=[0.0001]
patience=10
min_train_epochs=50
max_epochs=100

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
exp_results = []

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        folder = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        os.makedirs(f"./results/{folder}",exist_ok=True)
        folder_path = f"./results/{folder}"
                
        exp_model = model.vanilla_model()
        train_ds = train_set.map(util.preprocess_data).batch(batch_size).map(util.augment_data)
        val_ds = val_set.map(util.preprocess_data).batch(batch_size)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience,monitor="val_loss",restore_best_weights=True)

        exp_model.compile(optimizer=optimizer,loss=loss_fn)


#         training_history_first = exp_model.fit(train_ds, epochs=min_train_epochs)
        
#         plt.plot(training_history_first.history['loss'],label="training loss")
#         plt.legend()
#         plt.savefig(f"{folder_path}/training_history_first.jpg")
#         plt.close()
#         print("Done first set of minimal training")
        training_history_second = exp_model.fit(train_ds,validation_data=val_ds, epochs=max_epochs,callbacks=[early_stopping])
        
        plt.plot(training_history_second.history['loss'],label="training loss")
        plt.plot(training_history_second.history['val_loss'],label="validation loss")
        plt.legend()
        plt.savefig(f"{folder_path}/training_history_second.jpg")
        plt.close()
        
        exp_model.save(f"{folder_path}/best.keras") 
        
        acc,f1, _ = util.evaluate(exp_model,val_ds,top_k=5)
        
        exp_results.append({'val_acc':acc,'val_f1':f1,'folder':folder,'batch_size':batch_size,'learning_rate':learning_rate})
        
    


result_df = pd.DataFrame(exp_results)     
result_df.to_csv("./results/exp_result.csv",index=False)

best_model_hyps = result_df[result_df['val_acc']==result_df['val_acc'].max()]
best_folder = best_model_hyps['folder'].item()
print("Best model")
print(best_model_hyps)

best_folder_path = f"./results/{best_folder}"

best_model = tf.keras.models.load_model(f"{best_folder_path}/best.keras")

acc,f1,cf = util.evaluate_on_test(best_model,top_k=5)
print("Accuracy(top-5) and F1 for best model")
print(acc, f1)



                           
