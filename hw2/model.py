import tensorflow as tf

L2_REG=None
def vggnet_s(config=[8,'M',16, 'M',32,32,'M'], output_class=100):
    global L2_REG
    l2_reg=L2_REG
    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(output_class,activation='softmax',kernel_regularizer=l2_reg)
    ])
    
    
    features = tf.keras.Sequential([
        tf.keras.Sequential([tf.keras.layers.Conv2D(c,(3,3),padding='same',use_bias=False,kernel_regularizer=l2_reg),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU()]) if c!='M' else tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")
                             for c in config])
    
    model = tf.keras.Sequential([features, tf.keras.layers.GlobalAveragePooling2D(keepdims=(3,3)), classifier])
    return model

def vggnet_l(config=[8,'M',16, 'M',32,32,'M',64,64,'M',64,64,'M'], output_class=100):
    global L2_REG
    l2_reg=L2_REG
    print("L2 Reg", L2_REG)
    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(output_class,activation='softmax',kernel_regularizer=l2_reg)
    ])
    
    
    features = tf.keras.Sequential([
        tf.keras.Sequential([tf.keras.layers.Conv2D(c,(3,3),padding='same',use_bias=False,kernel_regularizer=l2_reg),tf.keras.layers.BatchNormalization(),tf.keras.layers.ReLU()]) if c!='M' else tf.keras.layers.MaxPooling2D((2,2),strides=2,padding="same")
                             for c in config])
    
    model = tf.keras.Sequential([features, tf.keras.layers.GlobalAveragePooling2D(keepdims=(3,3)), classifier])
    return model



class ResBlock(tf.keras.Model):
    def __init__(self,channels, l2_reg=None):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels,(3,3),padding="same",use_bias=False, kernel_regularizer=l2_reg)        
        self.conv2 = tf.keras.layers.Conv2D(channels,(3,3),padding="same",use_bias=False, kernel_regularizer=l2_reg)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
    def call(self, inputs,training=False):
        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)
        
        return self.activation(x + inputs)
    

def resnet_model():
    global L2_REG
    l2_reg=L2_REG
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),padding='same',use_bias=False,kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        ResBlock(16,l2_reg),
        tf.keras.layers.Conv2D(32,(3,3),padding='same',strides=2,use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        ResBlock(32,l2_reg),
        tf.keras.layers.Conv2D(64,(3,3),padding='same',strides=2,use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        ResBlock(64,l2_reg),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(100,kernel_regularizer=l2_reg, activation='softmax')])
    return model
        

