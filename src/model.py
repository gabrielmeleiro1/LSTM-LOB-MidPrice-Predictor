from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

def create_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with increased units
    x = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Store for residual connection
    residual = x
    
    # Multi-Head Attention layer 
    attention = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention)
    
    # Second LSTM layer 
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Add residual connection
    x = Add()([x, residual])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Another Multi-Head Attention layer
    attention = MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention)
    
    # Third LSTM layer
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Dense layers with and regularization
    x = Dense(64, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Dense(32, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    outputs = Dense(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Using AdamW optimizer
    optimizer = AdamW(learning_rate=0.001, weight_decay=0.0001)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model


