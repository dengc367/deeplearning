import tensorflow as tf
from tensorflow.python.keras.engine.training import Model


class BaseModel():
    """base model to get the public methods
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def train(self, train_dataset, test_dataset, epochs, steps_per_epoch=None, checkpoint_path=None, checkpoint_frequency='epoch', restore_latest=False, monitor='val_loss', mode='min'):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=2, restore_best_weights=True)]
        if checkpoint_path is not None:
            if restore_latest:
                self.load_weights(checkpoint_path, restore_latest)
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq=checkpoint_frequency, monitor=monitor, mode=mode,
                                                               verbose=1, save_weights_only=True, save_best_only=True)
            callbacks.append(checkpoint_cb)

        history = self.model.fit(train_dataset,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=test_dataset,
                                 validation_steps=100,
                                 callbacks=callbacks)
        return history

    def compile(self, optimizer="adam"):
        self.model.compile(optimizer=optimizer)

    def save_model(self, saved_model_path, version=None, model=None):
        model = model if model else self.model
        if version:
            saved_model_path = saved_model_path + "/" + version
        # model.save(saved_model_path, signatures=model.call)
        model.save(saved_model_path)

    def load_weights(self, checkpoint_path, latest=False):
        if latest:
            checkpoint_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
            latest_path = tf.train.latest_checkpoint(checkpoint_dir)
            print('latest checkpoint dir: ', latest_path)
            self.model.load_weights(latest_path)
        else:
            self.model.load_weights(checkpoint_path)

    def save_weights(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

    def get_model(self):
        return self.model

    def summary(self):
        self.model.summary()

    def get_serving_model(self):
        raise NotImplementedError

    def save_serving_model(self, saved_model_path, version=None):
        serving_model = self.get_serving_model()
        self.save_model(saved_model_path, version=version, model=serving_model)
