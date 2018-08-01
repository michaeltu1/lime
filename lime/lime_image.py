"""
Functions for explaining classifiers that use Image data.
"""
import copy
import time
import itertools

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from ray.rllib.utils.timer import TimerStat

from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0., outline=False):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: TODO

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        elif outline:
            fs = [x[0] for x in exp
                  if x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue
                    # temp[segments == f, cp] *= 0.5
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose, random_state=self.random_state)
        self.random_seed = None
        self.data = None
        self.labels = None
        self.top = None
        
        self.times = {"Bernoulli Sampling Time":            [],
                      "Perturbed Data Point Creation Time": [],
                      "Classification Time":                []}

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         timed=False,
                         time_classification=False,
                         use_bandits=False,
                         epsilon=0.1):

        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        self.random_seed = random_seed

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        features_to_use = None
        if use_bandits:
            bandit = self.data_labels(image, fudged_image, segments, 
                                      classifier_fn, num_samples,
			              batch_size=batch_size,
				      timed=timed,
				      time_classification=time_classification,
				      num_features=num_features,
                                      use_bandits=True,
                                      epsilon=epsilon)
            data, labels = bandit.arrangements, bandit.perturbed_labels
            features_to_use = bandit.features
        else:
            data, labels = self.data_labels(image, fudged_image, segments,
                                            classifier_fn, num_samples,
          	                            batch_size=batch_size,
               	                            timed=timed,
                                            time_classification=time_classification)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        if top_labels == 1 and use_bandits:
            ret_exp.segments = segments
            ret_exp.local_exp[top[0]] = np.array([(ft, bandit.q_vals[ft]) for ft in features_to_use])
            return ret_exp
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
                timed=timed,
	        used_features=features_to_use)

        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    timed=False,
                    time_classification=False,
		    use_bandits=False,
		    epsilon=0.1,
		    num_features=5):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        st = time.time()
	
        # used as num_siuperpixels
        n_features = np.unique(segments).shape[0]
	
        if use_bandits:
            bandit = EpsilonGreedyDataLabels(image, fudged_image, segments, 
                                             classifier_fn, n_features,
                                             num_features, num_samples, epsilon, timed=timed)
            return bandit
	    
	
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        self.times["Bernoulli Sampling Time"].append(time.time() - st)
        labels = []
        data[0, :] = 1
        imgs = []

        for row in data:
            s = time.time()
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            self.times["Perturbed Data Point Creation Time"].append(time.time() - s)
            if len(imgs) == batch_size:
                s = time.time()
                preds = classifier_fn(np.array(imgs))
                self.times["Classification Time"].append(time.time() - s)
                labels.extend(preds)
                imgs = []           
        if len(imgs) > 0:
            s = time.time()
            preds = classifier_fn(np.array(imgs))
            self.times["Classification Time"].append(time.time() - s)
            labels.extend(preds)
        if time_classification:
            print("Average Classification Time: {} seconds".format(np.mean(self.times["Classification Time"])))
        if timed:
            avg_data_time = np.mean(self.times["Perturbed Data Point Creation Time"])
            avg_classification_time = np.mean(self.times["Classification Time"])
            print("Bernoulli Sampling Time: {} seconds".format(np.mean(self.times["Bernoulli Sampling Time"])))
            print("Average Perturbed Data Point Creation Time: {} seconds".format(avg_data_time))
            print("Average Classification Time: {} seconds".format(avg_classification_time))
            print("Average Time per Loop: {} seconds".format((avg_data_time + avg_classification_time) / data.shape[0]))
            print("data_labels Function: {} seconds".format(time.time() - st))
        return data, np.array(labels)


class EpsilonGreedyDataLabels(object):
    """
    Replace feature selection + neighborhood data generation
    """
    def __init__(self, image, fudged_image, segments, classifier_fn, num_superpixels, num_features, num_samples, epsilon, timed=False):
        """
        gt_pred: ground truth prediction made by the original model
        classifier_fn: classifier function of the original model
        num_superpixels: number of superpixels in the image
        num_features: number of most impactful superpixels asked for
        num_samples: number of data points to generate using epsilon greedy exploration
                     (used internally to determine superpixels to use, not actual neighborhood data)
        epsilon: constant influencing the amount of exploration vs. exploitation
        """
        self.image = image
        self.fudged_image = fudged_image
        self.segments = segments
        self.gt_pred = classifier_fn(image[np.newaxis,])
        self.classifier_fn = classifier_fn
        self.num_superpixels = num_superpixels or num_features
        self.num_features = num_features
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.timed = timed
        self.eps_greedy_data = []
        self.eps_greedy_imgs = []
        self.eps_greedy_preds = None
        self.rewards = None
        self.counts = np.array([0.0] * num_superpixels)
        self.q_vals = np.array([epsilon] * num_superpixels)
        self.arrangements = []
        self.perturbed_data = []
        self.perturbed_labels = []
        self.features = []
        self.times = {"Epsilon-Greedy Image Creation Time": [],
                      "Perturbed Data Point Creation Time": [],
                      "Epsilon-Greedy 10 Image Classification Time": [],
                      "Reward, Count, and Q-Value Eval Time": []}


        self.run()

        if timed:
            self.times["Average Epsilon-Greedy Image Creation Time"] = np.mean(self.times["Epsilon-Greedy Image Creation Time"])
            self.times["Average Perturbed Data Point Creation Time"] = np.mean(self.times["Perturbed Data Point Creation Time"])
            self.times["Average Epsilon-Greedy 10 Image Classification Time"] = np.mean(self.times["Epsilon-Greedy 10 Image Classification Time"])
            self.times["Average Reward, Count, and Q-Value Eval Time"] = np.mean(self.times["Reward, Count, and Q-Value Eval Time"])

            del self.times["Epsilon-Greedy Image Creation Time"]
            del self.times["Perturbed Data Point Creation Time"]
            del self.times["Epsilon-Greedy 10 Image Classification Time"]
            del self.times["Reward, Count, and Q-Value Eval Time"]

            for k, v in self.times.items():
                print("{}: {} seconds".format(k, v))

    # Used internally to decide how to generate the neighborhood
    def generate_data(self, num_samples):

        s = time.time()
        for __ in range(num_samples):
            self.eps_greedy_data.append(np.array([0 if np.random.random() < eps else 1 for eps in self.q_vals]))
        self.eps_greedy_data = np.array(self.eps_greedy_data)
        self.times["Epsilon-Greedy Sampling Time"] = time.time() - s

        for i in range(num_samples):
            s = time.time()
            arr = self.eps_greedy_data[i]
            temp = copy.deepcopy(self.image)
            zeros = np.where(arr == 0)[0]
            mask = np.zeros(self.segments.shape).astype(bool)
            for z in zeros:
                mask[self.segments == z] = True
            temp[mask] = self.fudged_image[mask]
            self.eps_greedy_imgs.append(temp)
            self.times["Epsilon-Greedy Image Creation Time"].append(time.time() - s)
        self.eps_greedy_imgs = np.array(self.eps_greedy_imgs)

    # Creates returned neighborhood data
    def generate_neighborhood_and_labels(self):
        binary_permutations = ["".join(seq) for seq in itertools.product("01", repeat=self.num_features)]
        binary_permutations.reverse()
        for bp in binary_permutations:
            s = time.time()
            sample = np.array([1] * self.num_superpixels)
            for i in range(self.num_features):
                if bp[i] == "0":
                    sample[self.features[i]] = 0
            self.arrangements.append(sample)
            self.times["Perturbed Data Point Creation Time"].append(time.time() - s)
        self.arrangements = np.array(self.arrangements)

        for arr in self.arrangements:
            temp = copy.deepcopy(self.image)
            zeros = np.where(arr == 0)[0]
            mask = np.zeros(self.segments.shape).astype(bool)
            for z in zeros:
                mask[self.segments == z] = True
            temp[mask] = self.fudged_image[mask]
            self.perturbed_data.append(temp)

        self.perturbed_data = np.array(self.perturbed_data)

        s = time.time()
        self.perturbed_labels = self.classifier_fn(self.perturbed_data)
        self.times["Perturbed Data Classification Time"] = time.time() - s

    def run(self):
        for i in range(10):
            self.generate_data(10)

            s = time.time()
            self.eps_greedy_preds = self.classifier_fn(self.eps_greedy_imgs)
            self.times["Epsilon-Greedy 10 Image Classification Time"].append(time.time() - s)

            s = time.time()
            same_pred_samples = np.where(np.argmax(self.eps_greedy_preds, axis=1) == np.argmax(self.gt_pred[0]))
            self.rewards = self.eps_greedy_data[same_pred_samples].sum(axis=0)
            self.counts += self.eps_greedy_data.sum(axis=0)
            self.q_vals = (np.divide(self.counts - 1, self.counts, out=np.zeros_like(self.counts), where=self.counts!=0).T * self.q_vals + \
                           np.divide(np.ones_like(self.counts), self.counts, out=np.zeros_like(self.counts), where=self.counts!=0).T * self.rewards).T

            self.eps_greedy_data = []
            self.eps_greedy_imgs = []

            self.times["Reward, Count, and Q-Value Eval Time"].append(time.time() - s)

        self.features = np.argsort(self.q_vals)[-self.num_features:]

        st = time.time()
        self.generate_neighborhood_and_labels()
        self.times["data_labels Time"] = time.time() - st

