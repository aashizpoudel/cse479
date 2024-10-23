import tensorflow as tf
import util 
import model
import matplotlib.pyplot as plt
import os 
import datetime
import json
import pandas as pd
import numpy as np
import shutil
from sklearn.metrics import ConfusionMatrixDisplay



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
    return result_df

if __name__ == "__main__":
    print("Running the experiments")
    os.makedirs("./tmp",exist_ok=True)
    batch_sizes = [128, 256]
    learning_rates = [0.001, 0.01]
    #exp 1 vgg_net r, resnet with regularizer
    model.L2_REG = tf.keras.regularizers.L2(0.0001)


    # For vggnets 
    all_results = []
    train_set, val_set = util.get_train_val_set()
    for pf,fn in [("vgg_lr",model.vggnet_l), ("vgg_sr",model.vggnet_s),("resr", model.resnet_model)]:
        results = run_experiment(fn,train_set,val_set,learning_rates=learning_rates,batch_sizes=batch_sizes,result_dir=f"./tmp/results_{pf}", patience=15)
        all_results.append(results)

    all_results_df = pd.concat(all_results, ignore_index=False, sort=False)
    all_results_df.to_csv("./tmp/all_results_exp1.csv",index=False)
    
    #exp 2 vgg_net r, resnet without regularizer
    model.L2_REG = None
    all_results = []
    train_set, val_set = util.get_train_val_set()
    for pf,fn in [("vgg_l",model.vggnet_l), ("vgg_s",model.vggnet_s),("res", model.resnet_model)]:
        results = run_experiment(fn,train_set,val_set,learning_rates=learning_rates,batch_sizes=batch_sizes,result_dir=f"./tmp/results_{pf}", patience=15)
        all_results.append(results)

    all_results_df = pd.concat(all_results, ignore_index=False, sort=False)
    all_results_df.to_csv("./tmp/all_results_exp2.csv",index=False)

    results_1 = pd.read_csv("./tmp/all_results_exp1.csv")
    results_2 = pd.read_csv("./tmp/all_results_exp2.csv")

    #make figures folder
    print("Saving to figures folder")
    os.makedirs("./tmp/figures",exist_ok=True)
    for results in [results_1,results_2]:
        for id,result in results.iterrows():
            folder = result['folder']
            model = folder.split("/")[2].split("_")[1:]
            model_str = "_".join(model)
            history_file = os.path.join(result['folder'],"training_history.jpg")
            new_file = f"{model_str}_{result['learning_rate']:.0e}_{result['batch_size']}_th.jpg"
            destination_file = os.path.join("./tmp/figures",new_file)
            shutil.copyfile(history_file, destination_file)
            print(destination_file)

    #Evaluation on unseen dataset
    all_results = pd.concat([results_1,results_2], ignore_index=True, sort=False)
    best_result = all_results[all_results['val_acc']==all_results['val_acc'].max()]
    print("Best result")
    print(best_result)
    best_model_path = best_result['folder'].item() + "/best.hd5"
    print(best_model_path)
    best_model = tf.keras.models.load_model(best_model_path)


    test_set = util.get_test_set()
    test_ds = test_set.map(util.preprocess_data).batch(32)
    predictions = best_model.predict(test_ds).argmax(-1)
    truths = np.concatenate([tf.argmax(labels,axis=-1) for _,labels in test_ds],axis=0)

    #Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
       truths, predictions)
    plt.savefig("./tmp/figures/confusion_matrix.jpg")
    plt.close()  


    print("Best model evaluation")
    acc,f1,_ = util.evaluate_on_test(best_model,top_k=5)
    print("Top-5 accuracy for the best model", acc)
