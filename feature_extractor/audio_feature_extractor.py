import os
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
import argparse


def extract(checkpoint_path, num_secs, audio_dir, npy_dir):
    # Paths to downloaded VGGish files.
    # pca_params_path = 'vggish_pca_params.npz'
    # freq = 1000
    # sr = 44100

    # path of audio files and AVE annotation
    lis = os.listdir(audio_dir)
    len_data = len(lis)
    # audio_features = np.zeros([len_data, 10, 128])

    # i = 0
    for n in range(len_data):
        '''feature learning by VGG-net trained by audioset'''
        audio_index = os.path.join(audio_dir, lis[n])  # path of your audio files

        input_batch = vggish_input.wavfile_to_examples(audio_index)
        np.testing.assert_equal(
            input_batch.shape,
            [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

        # Define VGGish, load the checkpoint, and run the batch through the model to
        # produce embeddings.
        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: input_batch})
            # print('VGGish embedding: ', embedding_batch[0])
            # print(embedding_batch.shape)
            # audio_features[i, :, :] = embedding_batch
            # i += 1
            # print(i)
            # save npy file
            np.save(os.path.join(npy_dir, os.path.splitext(lis[n])[0] + '.npy'), embedding_batch)

    # save the audio features into one .h5 file.
    # If you have a very large dataset, you may save each feature into one .npy file

    # with h5py.File('.../audio_embedding.h5', 'w') as hf:
    #     hf.create_dataset("dataset",  data=audio_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', dest='audio_dir', type=str, default='data/LLP_dataset/audio')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/feats/vggish')
    parser.add_argument('--ckpt_path', dest='ckpt_path', type=str, default='feature_extractor/vggish_model.ckpt')
    parser.add_argument('--num_secs', dest='num_secs', type=int, default=10)
    parser.add_argument("--gpu", dest='gpu', type=str, default='9')

    args = parser.parse_args()
    audio_dir = args.audio_dir
    out_dir = args.out_dir
    ckpt_path = args.ckpt_path
    num_secs = args.num_secs

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # set gpu number

    extract(ckpt_path, num_secs, audio_dir, out_dir)
