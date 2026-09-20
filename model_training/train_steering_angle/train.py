import os
import sys

# Add project root directory to sys.path to allow proper imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore [missing-import]
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

# pyrefly: ignore [missing-import]
from src.models import model
from model_training.train_steering_angle import driving_data

class DataLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.summary_writer = tf.compat.v1.summary.FileWriter(log_path, graph=tf.get_default_graph())

    def log(self, summary, step):
        self.summary_writer.add_summary(summary, step)

    def close(self):
        self.summary_writer.close()

class Trainer:
    def __init__(self, model, log_dir, logger, l2_norm_const=0.001, learning_rate=0.0001):
        self.log_dir = log_dir
        self.l2_norm_const = l2_norm_const
        self.logger = logger

        self.session = tf.InteractiveSession()
        self.loss, self.train_step = self._build_training_graph(model, learning_rate)

        self.saver = tf.compat.v1.train.Saver(write_version=tf.train.SaverDef.V2)
        self.session.run(tf.compat.v1.global_variables_initializer())

        self.merged_summary_op = tf.compat.v1.summary.merge_all()

    def _build_training_graph(self, model, learning_rate):
        train_vars = tf.compat.v1.trainable_variables()
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y_pred))) + \
                     tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * self.l2_norm_const
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        tf.compat.v1.summary.scalar('loss', loss)
        return loss, train_step

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            self._train_one_epoch(epoch, batch_size)
            print(f"Epoch {epoch + 1}/{epochs} completed.")

    def _train_one_epoch(self, epoch, batch_size):
        for i in range(int(driving_data.num_images / batch_size)):
            xs, ys = driving_data.LoadTrainBatch(batch_size)
            self.train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.5})

            if i % 10 == 0:
                self._log_training_progress(epoch, i, batch_size)

            if i % batch_size == 0:
                self._save_checkpoint(epoch, i)

    def _log_training_progress(self, epoch, step, batch_size):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        loss_value = self.loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss_value}")

        summary = self.session.run(self.merged_summary_op, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        self.logger.log(summary, int(epoch * (driving_data.num_images / batch_size) + step))

    def _save_checkpoint(self, epoch, step):
        os.makedirs(self.log_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.log_dir, 'model.ckpt')
        self.saver.save(self.session, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path} (Epoch {epoch + 1}, Step {step})")

        # Also persist checkpoints to saved_models directory
        target_dir = 'saved_models/steering_angle'
        try:
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, 'model.ckpt')
            self.saver.save(self.session, target_path)
        except Exception:
            pass

    def close(self):
        self.session.close()
        self.logger.close()


if __name__ == "__main__":
    LOG_DIR = 'saved_models/steering_angle'
    LOGS_PATH = 'src/training/train_steering_angle/logs'
    EPOCHS = 30
    BATCH_SIZE = 100

    logger = DataLogger(LOGS_PATH)
    trainer = Trainer(model, LOG_DIR, logger)

    try:
        trainer.train(EPOCHS, BATCH_SIZE)
    finally:
        trainer.close()

    print("Run the command line:\n"
          "--> tensorboard --logdir=src/training/train_steering_angle/logs\n"
          "Then open http://0.0.0.0:6006/ in your web browser")
