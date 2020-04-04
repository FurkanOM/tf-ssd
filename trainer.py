import tensorflow as tf
import helpers
import augmentation
import ssd

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

batch_size = 32
epochs = 200
load_weights = False
ssd_type = "ssd300"
hyper_params = helpers.get_hyper_params(ssd_type)

VOC_train_data, VOC_info = helpers.get_dataset("voc/2007", "train+validation")
VOC_val_data, _ = helpers.get_dataset("voc/2007", "test")
VOC_train_total_items = helpers.get_total_item_size(VOC_info, "train+validation")
VOC_val_total_items = helpers.get_total_item_size(VOC_info, "test")
step_size_train = helpers.get_step_size(VOC_train_total_items, batch_size)
step_size_val = helpers.get_step_size(VOC_val_total_items, batch_size)
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
img_size = helpers.SSD[ssd_type]["img_size"]

VOC_train_data = VOC_train_data.map(lambda x : helpers.preprocessing(x, img_size, img_size, augmentation.apply))
VOC_val_data = VOC_val_data.map(lambda x : helpers.preprocessing(x, img_size, img_size))

padded_shapes, padding_values = helpers.get_padded_batch_params()
VOC_train_data = VOC_train_data.shuffle(batch_size*4).padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
VOC_val_data = VOC_val_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
#
ssd_model = ssd.get_model(hyper_params)
ssd_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                  loss=[ssd.loc_loss_fn, ssd.conf_loss_fn])
ssd.init_model(ssd_model)
#
ssd_model_path = helpers.get_model_path(ssd_type)
if load_weights:
    ssd_model.load_weights(ssd_model_path)
ssd_log_path = helpers.get_log_path(ssd_type)
# We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
prior_boxes = ssd.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_train_feed = ssd.generator(VOC_train_data, prior_boxes, hyper_params)
ssd_val_feed = ssd.generator(VOC_val_data, prior_boxes, hyper_params)

custom_callback = helpers.CustomCallback(ssd_model_path, monitor="val_loss", patience=epochs)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=ssd_log_path)

ssd_model.fit(ssd_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=ssd_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[custom_callback, tensorboard_callback])
