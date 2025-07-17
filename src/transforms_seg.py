import torch 
import numpy as np
from skimage import transform

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size  # e.g., (256, 256)

    def __call__(self, sample):
        image = sample['image']  # shape: (1, D, H, W)
        D = image.shape[1]

        if D == 0:
            raise ValueError("El video no tiene frames o está mal leído.")

        resized = np.zeros((1, D, self.output_size[0], self.output_size[1]), dtype=np.float32)
        for i in range(D):
            try:
                resized_frame = transform.resize(
                    image[0, i],
                    self.output_size,
                    preserve_range=True,
                    anti_aliasing=True,
                    mode='reflect'
                ).astype(np.float32)
                # Recorte defensivo
                resized_frame = resized_frame[:self.output_size[0], :self.output_size[1]]
                resized[0, i] = resized_frame
            except Exception as e:
                print(f"Error resizing frame {i}: {e}")
                raise

        sample['image'] = resized

        if 'pred' in sample:
            pred = sample['pred']  # shape: (D, H, W)
            resized_pred = np.zeros((D, self.output_size[0], self.output_size[1]), dtype=np.float32)
            for i in range(D):
                try:
                    resized_pred_frame = transform.resize(
                        pred[i],
                        self.output_size,
                        preserve_range=True,
                        anti_aliasing=True,
                        mode='reflect'
                    ).astype(np.float32)
                    resized_pred_frame = resized_pred_frame[:self.output_size[0], :self.output_size[1]]
                    resized_pred[i] = resized_pred_frame
                except Exception as e:
                    print(f"Error resizing pred frame {i}: {e}")
                    raise
            sample['pred'] = resized_pred

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image_var_torch = torch.from_numpy(image).float() / 255.0
        sample_out = {'image': image_var_torch}

        if 'pred' in sample:
            pred = sample['pred']
            pred_tensor = torch.from_numpy(pred).float() / 255.0
            sample_out['pred'] = pred_tensor

        return sample_out