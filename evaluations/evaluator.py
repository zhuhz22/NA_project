import argparse
import io
import os
import json
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import requests

# import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

# from prdc import compute_prdc
from pprint import pprint

INCEPTION_V3_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
)
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"

from evaluations.feature_extractor import build_feature_extractor
from cleanfid.resize import build_resizer
import torch

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio, MeanSquaredError
from torchmetrics.functional import structural_similarity_index_measure

MODE = "legacy_tensorflow"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_batch", help="path to reference batch npz file")
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    parser.add_argument("--metric", default="fid", help="metric to compute")
    args = parser.parse_args()

    if args.metric == "fid":
        get_fid(args)
    elif args.metric == "is":
        get_is(args)
    elif args.metric == "lpips":
        get_ssim_lpips(args)


def get_is(args):
    model = build_feature_extractor(MODE, torch.device("cuda"), use_dataparallel=True)
    evaluator = Evaluator(model, batch_size=1024)

    print("computing sample batch activations...")
    _, sample_logits = evaluator.read_activations(args.sample_batch)

    print("Computing evaluations...")
    inception = evaluator.compute_inception_score(sample_logits)
    metrics = {}

    metrics["inception"] = inception
    pprint(metrics)

    save_dir = os.path.dirname(os.path.realpath(args.sample_batch))
    with open(os.path.join(save_dir, "is.json"), "w") as f:
        json.dump(metrics, f)


def get_fid(args):
    nearest_k = 3
    model = build_feature_extractor(MODE, torch.device("cuda"), use_dataparallel=True)
    # model=None
    # print("warming up TensorFlow...")
    # config = tf.ConfigProto(
    #     allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    # )
    # config.gpu_options.allow_growth = True
    evaluator = Evaluator(model, batch_size=1024)

    # This will cause TF to print a bunch of verbose stuff now rather
    # than after the next print(), to help prevent confusion.
    # evaluator.warmup()

    print("computing reference batch activations...")
    obj = np.load(args.ref_batch)

    if "pool_3" in obj:
        ref_acts = obj["pool_3"]
    else:
        if not ("mu" in obj and "sigma" in obj and "act" in obj):
            ref_acts, _ = evaluator.read_activations(args.ref_batch)
            # ref_acts = evaluator.read_activations(args.ref_batch)
        else:
            ref_acts = obj["act"]

    print("computing/reading reference batch statistics...")
    ref_stats = evaluator.read_statistics(args.ref_batch, ref_acts, store=False)
    print("computing sample batch activations...")
    sample_acts, sample_logits = evaluator.read_activations(args.sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats = evaluator.read_statistics(args.sample_batch, sample_acts)

    print("Computing evaluations...")
    fid = sample_stats.frechet_distance(ref_stats)
    inception = evaluator.compute_inception_score(sample_logits)
    metrics = {}

    metrics["fid"] = fid
    metrics["inception"] = inception
    pprint(metrics)

    save_dir = os.path.dirname(os.path.realpath(args.sample_batch))
    with open(os.path.join(save_dir, "fid.json"), "w") as f:
        json.dump(metrics, f)


def get_ssim_lpips(args):

    print("Computing SSIM, LPIPS")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to("cuda")
    psnr = PeakSignalNoiseRatio().to("cuda")
    mse = MeanSquaredError().to("cuda")
    ref_batch = np.load(args.ref_batch)["arr_0"]
    sample_batch = np.load(args.sample_batch)["arr_0"]
    assert ref_batch.shape == sample_batch.shape

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(ref_batch).permute(0, 3, 1, 2), torch.from_numpy(sample_batch).permute(0, 3, 1, 2)
    )

    dataloader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=1)

    ssim_all = []
    lpips_all = []
    mse_all = []
    psnr_all = []
    for i, (ref, sample) in enumerate(tqdm(dataloader)):
        ref = ref.to("cuda").float() / 255 * 2 - 1
        sample = sample.to("cuda").float() / 255 * 2 - 1

        with torch.no_grad():
            if ref.shape[-1] < 256:
                ref = torch.nn.functional.interpolate(ref, size=224, mode="bilinear")
                sample = torch.nn.functional.interpolate(sample, size=224, mode="bilinear")
            ssim_score = structural_similarity_index_measure(sample, ref)
            lpips_score = lpips(ref, sample)
            psnr_score = psnr(sample, ref)
            mse_score = mse(sample.flatten(1), ref.flatten(1))
            ssim_all.append(ssim_score)
            lpips_all.append(lpips_score)
            mse_all.append(mse_score)
            psnr_all.append(psnr_score)

    ssim_all = torch.mean(torch.tensor(ssim_all))
    lpips_all = torch.mean(torch.tensor(lpips_all))
    mse_all = torch.mean(torch.tensor(mse_all))
    psnr_all = torch.mean(torch.tensor(psnr_all))

    metrics = {}
    metrics["ssim"] = ssim_all.item()
    metrics["lpips"] = lpips_all.item()
    metrics["mse"] = mse_all.item()
    metrics["psnr"] = psnr_all.item()
    pprint(metrics)

    save_dir = os.path.dirname(os.path.realpath(args.sample_batch))
    with open(os.path.join(save_dir, "ssim_lpips.json"), "w") as f:
        json.dump(metrics, f)


class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(
        self,
        model,
        batch_size=1024,
        softmax_batch_size=512,
    ):
        self.model = model
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.fn_resize = build_resizer(MODE)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.

        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                 dimension. The tuple is (pool_3, spatial).
        """
        preds = []
        logits = []
        for batch in tqdm(batches):
            if MODE == "legacy_tensorflow":
                batch = torch.from_numpy(batch.transpose([0, 3, 1, 2])).float()
            else:
                batch = self.resize_batch(batch)

            # batch = batch.astype(np.float32)
            pred, logit = self.model(batch)
            preds.append(pred.detach().cpu().numpy().reshape([pred.shape[0], -1]))
            logits.append(logit.detach().cpu().numpy().reshape([logit.shape[0], -1]))
            # spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(logits, axis=0),
        )

    def resize_batch(self, img_batch):
        batch_size = img_batch.shape[0]
        l_resized_batch = []
        for idx in range(batch_size):
            img_np = img_batch[idx]

            img_resize = self.fn_resize(img_np)
            l_resized_batch.append(torch.tensor(img_resize.transpose((2, 0, 1))).unsqueeze(0))
        resized_batch = torch.cat(l_resized_batch, dim=0)
        return resized_batch

    def read_statistics(
        self,
        npz_path: str,
        activations: Tuple[np.ndarray, np.ndarray],
        store=False,
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()) and "sigma" in list(obj.keys()) and "act" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"])

        FIDstats = self.compute_statistics(activations)
        if store:
            np.savez(npz_path, arr_0=obj["arr_0"], mu=FIDstats.mu, sigma=FIDstats.sigma, act=activations)
        return FIDstats

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, logits: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(logits), self.softmax_batch_size):
            logits_batch = logits[i : i + self.softmax_batch_size]
            softmax_out.append(torch.nn.functional.softmax(torch.from_numpy(logits_batch), dim=-1).numpy())
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_prec_recall(self, activations_ref: np.ndarray, activations_sample: np.ndarray) -> Tuple[float, float]:
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(activations_ref, radii_1, activations_sample, radii_2)
        return (float(pr[0][0]), float(pr[1][0]))


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix)
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L50-L63
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


if __name__ == "__main__":
    main()
