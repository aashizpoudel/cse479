#experiment 1
import pandas as pd
from pathlib import Path 
import model
import util
import tensorflow as tf

def run_experiment(model_fn,train_ds, val_ds, loss_fn, vectorizer,logs="./results"): 
# model_fn = model.get_bidirectional_lstm_attention
    result_dir = Path(logs)
    exp_results = []
    for batch_size in [32,64]:
        for learning_rate in [0.01,0.001,0.0001]:
            for regularizer in [True, False]:
                s_result = {'batch_size':batch_size,'learning_rate':learning_rate,'regularizer':regularizer}
                folder_name = f"{batch_size}_{learning_rate}_{regularizer}"
                result_path = result_dir.joinpath(folder_name)
                result_path.mkdir(parents=True,exist_ok=True)
                print("Working on",folder_name)
                reg = tf.keras.regularizers.L1L2(0.001,0.01) if regularizer else None
                model = model_fn(vectorizer=vectorizer,kernel_regularizer=reg,use_dropout=regularizer)
                print(model.summary())
                early_callback = tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_binary_accuracy", restore_best_weights=True)
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                history = util.train_model(model, train_ds, val_ds, loss_fn, batch_size=batch_size, optimizer=optimizer, callbacks=[early_callback],epochs=200)
                accuracy = util.evaluate_model(model, val_ds,result_path=result_path)
                util.plot_training_graphs(history, result_path)
                print("For",folder_name,"accuracy",accuracy,"early_stopping", early_callback.best)
                s_result['accuracy'] = accuracy
                s_result['logs'] = history.history 
                model.save_weights(result_path.joinpath("best_weights.hd5"))
                model.save(result_path.joinpath("best_model.hd5"))
                # print("Accuracy on validation", accuracy)
                # break
                s_result['model'] = result_path.joinpath("best_model.hd5")

                exp_results.append(s_result)
                del model
    final_results=pd.DataFrame(exp_results)
    final_results.to_csv(f"{logs}/exp_results.csv")
    return final_results


if __file__=="__main__":
    model_fn = model.get_bidirectional_lstm_attention
    train_ds, val_ds = util.get_train_val_ds()
    loss_fn = tf.keras.losses.BinaryCrossEntropy(from_logits=True)
    vocab = util.get_for_vocab_ds()
    vectorizer = util.get_text_vectorizer(vocab, vocab_size=30000)
    results_l = run_experiment(model_fn, train_ds, val_ds, loss_fn, vectorizer, "./results_lstm_")
    model_fn = model.get_bidirectional_gru_attention
    results_g = run_experiment(model_fn, train_ds, val_ds, loss_fn, vectorizer, "./results_gru_")
    combined_results = pd.concat([results_l,results_g],reset_index=True, sort=False)
    with_regularizer = combined[combined['regularizer']==True]
    best= with_regularizer[with_regularizer['accuracy']==with_regularizer['accuracy'].max()]
    best_config = best['model'].iloc[0]
    if "gru" in best_config:
        best_model = model.get_bidirectional_gru_attention(vectorizer)
    else:
        best_model = model.get_bidirectional_lstm_attention(vectorizer)
    best_model.load_weights(best_config)
    test_ds= util.get_test_ds()
    accuracy = util.evaluate_model(best_model, test_ds,Path("./"))
    print("Best model")
    print(best)
    print("Test accuracy", accuracy)