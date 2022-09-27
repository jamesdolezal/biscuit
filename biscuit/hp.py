import slideflow as sf

nature2022 = sf.model.ModelParams(
    model='xception',
    tile_px=299,
    tile_um=302,
    batch_size=128,
    epochs=[1],         # epochs 1, 3, 5, 10 used for initial sweep
    early_stop=True,
    early_stop_method='accuracy',
    dropout=0.1,
    uq=False,           # to be enabled in separate sub-experiments
    hidden_layer_width=1024,
    optimizer='Adam',
    learning_rate=0.0001,
    learning_rate_decay_steps=512,
    learning_rate_decay=0.98,
    loss='sparse_categorical_crossentropy',
    normalizer='reinhard_fast',
    include_top=False,
    hidden_layers=2,
    pooling='avg',
    augment='xyrjb'
)