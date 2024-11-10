# Add attention layer to the deep learning network
import keras.backend as K
import tensorflow as tf
class Attention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context =K.sum(context, axis=1)
        return context
    
    
def get_bidirectional_lstm_attention(vectorizer,kernel_regularizer=None, use_dropout=False):
    input = tf.keras.layers.Input(shape=[None],dtype=tf.string)
    vocab_len = len(vectorizer.get_vocabulary())
    x = vectorizer(input)
    x = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=64, mask_zero=True,name="embedding")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True,kernel_regularizer=kernel_regularizer), name="bi_lstm_0")(x)
    # x, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True, return_state=True), name="bi_lstm_0")(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True,kernel_regularizer=kernel_regularizer), name="bi_lstm_1")(x)
    x = Attention()(x)
    # x = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=kernel_regularizer )(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer)(x)
    if use_dropout:
        x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=input,outputs=x)


def get_bidirectional_gru_attention(vectorizer,kernel_regularizer=None, use_dropout=False):
    input = tf.keras.layers.Input(shape=[None],dtype=tf.string)
    vocab_len = len(vectorizer.get_vocabulary())
    x = vectorizer(input)
    x = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=64, mask_zero=True,name="embedding")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True,kernel_regularizer=kernel_regularizer), name="bi_gru_0")(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True,kernel_regularizer=kernel_regularizer), name="bi_lstm_0")(x)
    # x, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True, return_state=True), name="bi_lstm_0")(x)
    
    x = Attention()(x)
    
    # x = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=kernel_regularizer )(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer)(x)
    if use_dropout:
        x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=input,outputs=x)

    