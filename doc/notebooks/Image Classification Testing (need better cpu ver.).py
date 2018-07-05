
# coding: utf-8

# Here is a simpler example of the use of LIME for image classification by using Keras (v2 or greater)

# In[1]:


import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

print('Notebook run using keras:', keras.__version__)


# # Using Inception
# Here we create a standard InceptionV3 pretrained model and use it on images by first preprocessing them with the preprocessing tools

# In[2]:


inet_model = inc_net.InceptionV3()


# In[3]:


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


# ## Let's see the top 5 prediction for some image

# In[4]:


images = transform_img_fn([os.path.join('data','cat_mouse.jpg')])
# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
plt.imshow(images[0] / 2 + 0.5)
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)


# (class, description, probability)

# ## Explanation
# Now let's get an explanation

# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image


# In[6]:


explainer = lime_image.LimeImageExplainer()


# hide_color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels. Here, we set it to 0 (in the representation used by inception model, 0 means gray)

# In[7]:


get_ipython().run_cell_magic('time', '', '# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels\nexplanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)')


# Follows same steps as LimeTextExp -- i.e. get neighborhood, train model to be like og model  
# Perturb - on/off superpixels  

# Image classifiers are a bit slow. Notice that an explanation on my Surface Book dGPU took 1min 29s

# ### Now let's see the explanation for the top class ( Black Bear)

# We can see the top 5 superpixels that are most positive towards the class with the rest of the image hidden

# In[8]:


from skimage.segmentation import mark_boundaries


# Image Segmentation - process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics

# In[9]:


temp, mask = explanation.get_image_and_mask(295, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

print(explanation.top_labels)


# Or with the rest of the image present:

# In[10]:


temp, mask = explanation.get_image_and_mask(295, positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# We can also see the 'pros and cons' (pros in green, cons in red)

# In[11]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# Or the pros and cons that have weight at least 0.1

# In[12]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# # Testing Section!!

# jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb  (<-- to convert to python script)

# In[109]:


import types
from lime.utils.generic_utils import has_arg
from skimage.segmentation import felzenszwalb, slic, quickshift

class BaseWrapper(object):
    """Base class for LIME Scikit-Image wrapper


    Args:
        target_fn: callable function or class instance
        target_params: dict, parameters to pass to the target_fn


    'target_params' takes parameters required to instanciate the
        desired Scikit-Image class/model
    """

    def __init__(self, target_fn=None, **target_params):
        self.target_fn = target_fn
        self.target_params = target_params

        self.target_fn = target_fn
        self.target_params = target_params

    def _check_params(self, parameters):
        """Checks for mistakes in 'parameters'

        Args :
            parameters: dict, parameters to be checked

        Raises :
            ValueError: if any parameter is not a valid argument for the target function
                or the target function is not defined
            TypeError: if argument parameters is not iterable
         """
        a_valid_fn = []
        if self.target_fn is None:
            if callable(self):
                a_valid_fn.append(self.__call__)
            else:
                raise TypeError('invalid argument: tested object is not callable,                 please provide a valid target_fn')
        elif isinstance(self.target_fn, types.FunctionType)                 or isinstance(self.target_fn, types.MethodType):
            a_valid_fn.append(self.target_fn)
        else:
            a_valid_fn.append(self.target_fn.__call__)

        if not isinstance(parameters, str):
            for p in parameters:
                for fn in a_valid_fn:
                    if has_arg(fn, p):
                        pass
                    else:
                        raise ValueError('{} is not a valid parameter'.format(p))
        else:
            raise TypeError('invalid argument: list or dictionnary expected')

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        Args:
            **params: Dictionary of parameter names mapped to their values.

        Raises :
            ValueError: if any parameter is not a valid argument
                for the target function
        """
        self._check_params(params)
        self.target_params = params

    def filter_params(self, fn, override=None):
        """Filters `target_params` and return those in `fn`'s arguments.
        Args:
            fn : arbitrary function
            override: dict, values to override target_params
        Returns:
            result : dict, dictionary containing variables
            in both target_params and fn's arguments.
        """
        override = override or {}
        result = {}
        for name, value in self.target_params.items():
            if has_arg(fn, name):
                result.update({name: value})
        result.update(override)
        return result


class SegmentationAlgorithm(BaseWrapper):
    """ Define the image segmentation function based on Scikit-Image
            implementation and a set of provided parameters

        Args:
            algo_type: string, segmentation algorithm among the following:
                'quickshift', 'slic', 'felzenszwalb'
            target_params: dict, algorithm parameters (valid model paramters
                as define in Scikit-Image documentation)
    """

    def __init__(self, algo_type, **target_params):
        self.algo_type = algo_type
        if (self.algo_type == 'quickshift'):
            BaseWrapper.__init__(self, quickshift, **target_params)
            kwargs = self.filter_params(quickshift)
            self.set_params(**kwargs)
        elif (self.algo_type == 'felzenszwalb'):
            BaseWrapper.__init__(self, felzenszwalb, **target_params)
            kwargs = self.filter_params(felzenszwalb)
            self.set_params(**kwargs)
        elif (self.algo_type == 'slic'):
            BaseWrapper.__init__(self, slic, **target_params)
            kwargs = self.filter_params(slic)
            self.set_params(**kwargs)

    def __call__(self, *args):
            return self.target_fn(args[0], **self.target_params)


def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 15))
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()


# In[113]:


# # Loop for cell below:
# # Testing: Decreasing / increasing influence of superpixel
temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
gamma = 1.0
prediction = 'American_black_bear'

figures = {}

while gamma <= 1.5 and prediction == 'American_black_bear':
    print("\n" + str(gamma))
    index = np.where(mask != 0)
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[index] = alter_image[index] * gamma
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    prediction = tup[1]
    for x in decode_predictions(preds)[0]:
        print(x)
    gamma += .1
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex = lime_image.LimeImageExplainer()
    expl = ex.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl.top_labels))
    temp, mask = expl.get_image_and_mask(expl.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[121]:


# Testing: Decreasing influence of superpixel
temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)

gamma = 1.00
prediction = 'American_black_bear'

figures = {}

while gamma > .4 and prediction == 'American_black_bear':
    print("\n" + str(gamma))
    index = np.where(mask != 0)
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[index] = alter_image[index] * gamma
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    prediction = tup[1]
    for x in decode_predictions(preds)[0]:
        print(x)
    gamma -= .1

    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_0 = lime_image.LimeImageExplainer()
    expl_0 = ex_0.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_0.top_labels))
    temp, mask = expl_0.get_image_and_mask(expl_0.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[123]:


# Testing: Decreasing influence of all regions other than the superpixel
temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)

figures = {}

gamma = 1.0
prediction = 'American_black_bear'
while gamma <= 1 and prediction == 'American_black_bear':
    print("\n" + str(gamma))
    index = np.where(mask == 0)
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[index] = alter_image[index] * gamma
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    prediction = tup[1]
    for x in decode_predictions(preds)[0]:
        print(x)
    gamma -= .1
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_1 = lime_image.LimeImageExplainer()
    expl_1 = ex_1.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_1.top_labels))
    temp, mask = expl_1.get_image_and_mask(expl_1.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[125]:


# Testing: Increasing influence of all regions other than the superpixel
temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)

figures = {}

gamma = 1.0
prediction = 'American_black_bear'
while gamma <= 2 and prediction == 'American_black_bear':
    print("\n" + str(gamma))
    index = np.where(mask == 0)
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[index] = alter_image[index] * gamma
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    prediction = tup[1]
    for x in decode_predictions(preds)[0]:
        print(x)
    gamma += .1
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_2 = lime_image.LimeImageExplainer()
    expl_2 = ex_2.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_2.top_labels))
    temp, mask = expl_2.get_image_and_mask(expl_2.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[21]:


# Testing: Adding small shifts to the image, checking corresponding decisions

# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}
print("Testing Vertical Shifts ...\n")
# Using negative gammas / pixel shifts (downwards is the positive direction)
for gamma in [x for x in range(-1, -19, -1)]:
    print("gamma = " + str(gamma))
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[:gamma] = alter_image[-gamma:]
    alter_image = alter_image[-gamma:]
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    for x in decode_predictions(preds)[0]:
        print(x)
    print()

    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_3 = lime_image.LimeImageExplainer()
    expl_3 = ex_3.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_3.top_labels))
    temp, mask = expl_3.get_image_and_mask(expl_3.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[56]:


# Testing: Adding small shifts to the image, checking corresponding decisions

# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}
print("Testing Vertical Shifts ...\n")
# Positive gammas / pixel shifts downwards (downwards is the positive direction)
gamma = 1
prediction = "American_black_bear"
while prediction == "American_black_bear":
    print("gamma = " + str(gamma))
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[gamma:] = alter_image[:-gamma]
    alter_image = alter_image[gamma:]
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    for x in decode_predictions(preds)[0]:
        print(x)
    print()
    prediction = tup[1]
    gamma += 1
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_4 = lime_image.LimeImageExplainer()
    expl_4 = ex_4.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_4.top_labels))
    temp, mask = expl_4.get_image_and_mask(expl_4.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[75]:


# Testing: Adding small shifts to the image, checking corresponding decisions
gamma = -4 # Corresponds to size of pixel shift

# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}

print("Testing Horizontal Shifts ...\n")
# With negative gammas
gamma = -1
prediction = "American_black_bear"
while prediction == "American_black_bear":
    print("gamma = " + str(gamma))
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[:, :gamma] = alter_image[:, -gamma:]
    alter_image = alter_image[:, -gamma:]
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    prediction = tup[1]
    for x in decode_predictions(preds)[0]:
        print(x)
    print()
    gamma -= 1
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_5 = lime_image.LimeImageExplainer()
    expl_5 = ex_5.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_5.top_labels))
    temp, mask = expl_5.get_image_and_mask(expl_5.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[51]:


# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}

print("Testing Horizontal Shifts ...\n")
# With positive gammas
gamma = 1
result = "American_black_bear"
while result == "American_black_bear":
    print("gamma = " + str(gamma))
    alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
    alter_image[:, gamma:] = alter_image[:, :-gamma]
    alter_image = alter_image[:, gamma:]
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    result = tup[1]
    gamma += 1
    for x in decode_predictions(preds)[0]:
        print(x)
    print()
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_7 = lime_image.LimeImageExplainer()
    expl_7 = ex_7.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_7.top_labels))
    temp, mask = expl_7.get_image_and_mask(expl_7.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[32]:


# Testing: Rotations

# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}

print("Testing 90 Degree Rotations ...\n")
for i in range(1, 3):
    print("Rotated: " + str(i * 90) + " degrees counterclockwise")
    alter_image = np.rot90(alter_image)
    preds = inet_model.predict(np.array([alter_image]))
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    for x in decode_predictions(preds)[0]:
        print(x)
    print()
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_8 = lime_image.LimeImageExplainer()
    expl_8 = ex_8.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_8.top_labels))
    temp, mask = expl_8.get_image_and_mask(expl_8.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[37]:


# Testing: Rotations

# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}

print("Testing 90 Degree Rotations ...\n")
for i in range(1, 4):
    print("Rotated: " + str(i * 90) + " degrees counterclockwise")
    alter_image = np.rot90(alter_image)
    preds = inet_model.predict(np.array([alter_image]))
    preds = inet_model.predict(np.array([alter_image]))
    tup = decode_predictions(preds)[0][0]
    for x in decode_predictions(preds)[0]:
        print(x)
    print()
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_9 = lime_image.LimeImageExplainer()
    expl_9 = ex_9.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_9.top_labels))
    temp, mask = expl_9.get_image_and_mask(expl_9.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")


# In[41]:


# Testing: Zoom

# Original prediction
alter_image = transform_img_fn([os.path.join('data','cat_mouse.jpg')])[0]
preds = inet_model.predict(np.array([alter_image]))
print("Original Prediction:")
for x in decode_predictions(preds)[0]:
    print(x)
print()

figures = {}

print("Testing Zoom In ...\n")
# zoom should be < (299/2)
zoom, result = 1, "American_black_bear"
while zoom < 299 / 2 and result == "American_black_bear":
    print("zoom = " + str(zoom))
    zoomed = alter_image[zoom:-zoom, zoom:-zoom]
    preds = inet_model.predict(np.array([zoomed]))
    preds = inet_model.predict(np.array([zoomed]))
    tup = decode_predictions(preds)[0][0]
    result = tup[1]
    for x in decode_predictions(preds)[0]:
        print(x)
    print()
    zoom += 1
    
    alt = alter_image.copy()
    alt[np.where(alt > 1)] = 1
    alt[np.where(alt < -1)] = -1

    figures[str(round(gamma, 2))] = alt / 2 + .5
    
    ex_10 = lime_image.LimeImageExplainer()
    expl_10 = ex_10.explain_instance(alt, inet_model.predict, top_labels=5, hide_color=0, num_samples=10)
    print("\n" + str(expl_10.top_labels))
    temp, mask = expl_10.get_image_and_mask(expl_10.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    figures[str(round(gamma, 2)) + ",contrib"] = mark_boundaries(temp / 2 + 0.5, mask)

plot_figures(figures, 2, len(pics.items()) // 2)


# In[130]:


temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print("Contributing features in original:")

